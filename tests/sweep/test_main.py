"""`sweep-run` (spec sections 7 and 8; Review Focus 1 and 2).

`main` runs with the fake clock and fake pool of tests/sweep/conftest.py: no process is spawned,
nothing sleeps. Prices give 0.01 USD per pod-second (GPU 36 USD/h; the 20 GB volume adds
0.0019 USD/h). Grid and budget overrides keep calibration to about 1.5 USD of fake time.
"""

import dataclasses
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from resetsweep.ppo.update import START_CONFIG
from resetsweep.sweep.arms import arm_space
from resetsweep.sweep.budget import BudgetGuard
from resetsweep.sweep.calibration import FamilyPlan, Plan
from resetsweep.sweep.main import (
    BudgetTooSmallError,
    SettingsError,
    SweepLoop,
    load_settings,
    main,
    project_full_ladder,
    size_rungs,
    size_smoke_rung,
)
from resetsweep.sweep.pool import RunnerConfig
from resetsweep.sweep.scheduler import ConflictingResultError, Job
from resetsweep.train import RunResult, RunSpec

ARM_UNIFORM = next(
    a for a in arm_space("rung0") if a.label() == "pendulum_sparse/uniform/-"
)
ARM_VISITED = next(
    a
    for a in arm_space("rung0")
    if a.label() == "pendulum_sparse/visited[p_task=0.5]/-"
)


def _env(tmp_path: Path, **kw: str) -> dict[str, str]:
    env = {
        "SWEEP_CAP_USD": "20",
        "SWEEP_GPU_USD_PER_H": "36",
        "SWEEP_STATE_DIR": str(tmp_path / "sweep"),
        "WANDB_MODE": "disabled",
        "SWEEP_NUM_ENVS_GRID": "64,128,256",
        "SWEEP_WORKERS_GRID": "1,2,4",
        "SWEEP_ROLLOUT_LEN": "8",
        "SWEEP_CALIB_STEPS": "4000",
        "SWEEP_THROUGHPUT_ITERS": "2",
        "SWEEP_EVAL_N": "16",
        # The fake world's runs are tiny; the 2M default (a paid-run guard) would refuse them.
        "SWEEP_MIN_R0_STEPS": "512",
    }
    return {**env, **kw}


class RecordingSink:
    def __init__(self) -> None:
        self.tables: dict[str, Any] = {}
        self.summaries: list[dict[str, Any]] = []
        self.logs: list[tuple[float, dict[str, float]]] = []
        self.closed = False

    def log(self, elapsed_s: float, data: dict[str, float]) -> None:
        self.logs.append((elapsed_s, data))

    def log_table(
        self, name: str, columns: list[str], rows: list[list[object]]
    ) -> None:
        self.tables[name] = (columns, rows)

    def summary(self, data: dict[str, Any]) -> None:
        self.summaries.append(data)

    def close(self) -> None:
        self.closed = True


def _run(
    env: dict[str, str],
    clock: Any,
    pool_factory: Any,
    sink: RecordingSink | None = None,
) -> int:
    sink = sink or RecordingSink()
    return main(
        [],
        environ=env,
        clock=clock,
        pool_factory=pool_factory,
        report_sink_factory=lambda project, group, mode: sink,
        device="cpu",
    )


def _sweep_jobs(world: Any, session: int | None = None) -> list[Any]:
    return [
        s
        for s in world.submissions
        if s.job.rung >= 0 and (session is None or s.session == session)
    ]


def _ledger(state_dir: Path) -> list[dict[str, float]]:
    return [
        json.loads(line)
        for line in (state_dir / "ledger.jsonl").read_text().splitlines()
    ]


def _results(state_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (state_dir / "results.jsonl").read_text().splitlines()
    ]


# --- settings ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["SWEEP_CAP_USD", "SWEEP_GPU_USD_PER_H"])
def test_settings_missing_price_names_the_variable(tmp_path: Path, name: str) -> None:
    env = _env(tmp_path)
    del env[name]

    with pytest.raises(SettingsError, match=f"^{name} is not set"):
        load_settings(env)


@pytest.mark.parametrize("value", ["abc", "-3", "0", "nan", "inf", ""])
def test_settings_invalid_price_names_the_variable_not_the_value(
    tmp_path: Path, value: str
) -> None:
    env = _env(tmp_path, SWEEP_GPU_USD_PER_H=value)

    with pytest.raises(
        SettingsError, match="^SWEEP_GPU_USD_PER_H must be a finite number > 0$"
    ):
        load_settings(env)


def test_settings_defaults(tmp_path: Path) -> None:
    env = {"SWEEP_CAP_USD": "100", "SWEEP_GPU_USD_PER_H": "2.5"}

    s = load_settings(env)

    assert (s.state_dir, s.wandb_project, s.wandb_mode, s.smoke) == (
        Path("/workspace/sweep"),
        "reset-curriculum-sweep",
        "online",
        False,
    )
    assert (s.num_envs_grid, s.workers_grid, s.rollout_len, s.r0_steps) == (
        (2048, 4096, 8192),
        (1, 2, 4),
        24,
        None,
    )


def test_settings_reject_unknown_wandb_mode(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="^WANDB_MODE must be one of"):
        load_settings(_env(tmp_path, WANDB_MODE="dryrun"))


def test_main_returns_2_on_bad_settings(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    env = _env(tmp_path)
    del env["SWEEP_CAP_USD"]

    assert _run(env, clock, pool_factory) == 2


# --- rung sizing -------------------------------------------------------------------------------


def _family(name: str, steps_per_s: float, overhead_s: float = 4.0) -> FamilyPlan:
    return FamilyPlan(
        family=name,
        ppo=START_CONFIG,
        num_envs=64,
        rollout_len=8,  # one rollout = 512 steps, the smallest r0 size_rungs accepts
        steps_per_s=steps_per_s,
        overhead_s=overhead_s,
        calibrated=True,
    )


PLAN = Plan(
    n_workers=2,
    families={"a": _family("a", 1000.0), "b": _family("b", 500.0)},
    fallback=_family("fallback", 500.0),
)


# Keyed (family, method): every arm here is uniform, so each job counts x1.
A, B = ("a", "uniform"), ("b", "uniform")
COUNTS = ({A: 10.0, B: 5.0}, {A: 4.0, B: 2.0}, {A: 2.0, B: 1.0})


def test_size_rungs_fits_the_whole_ladder_in_90_percent_of_remaining() -> None:
    # Per-arm cost in r0 units: seeds x budget = 1x1, 3x3, 5x9 = 1, 9, 45.
    # Pod-s per r0 step: rung 0 10/1000 + 5/500 = 0.02; rung 1 9 * (4/1000 + 2/500) = 0.072;
    #   rung 2 45 * (2/1000 + 1/500) = 0.18; total 0.272.
    # Overhead (4 s per job over 2 workers): rung 0 15 * 1 * 2 = 30; rung 1 6 * 3 * 2 = 36;
    #   rung 2 3 * 5 * 2 = 30; total 96 s.
    # Budget 0.9 * 10 USD / 0.01 USD/s = 900 s -> r0 = floor((900 - 96) / 0.272) = 2955.
    sizing = size_rungs(
        arms_per_rung=COUNTS,
        plan=PLAN,
        remaining_usd=10.0,
        usd_per_s=0.01,
        min_r0_steps=1000,
    )

    assert sizing.rungs == (2955, 8865, 26595)
    # USD per rung: (2955 * 0.02 + 30) * 0.01, (2955 * 0.072 + 36) * 0.01, (2955 * 0.18 + 30) * 0.01
    assert sizing.cost_usd == pytest.approx((0.891, 2.48760, 5.619))
    assert sum(sizing.cost_usd) <= 0.9 * 10.0


# Smoke-patch item 2: full runs weight each job by its measured probe slowdown. "walk" on family
# a runs 2x uniform, so 5 walk arms on a cost what 5 uniform arms on b (half a's speed) do in
# COUNTS, and the hand-computed r0 = 2955 carries over. Ignoring the factor would give
# r0 = floor((900 - 96) / (0.015 + 0.054 + 0.135)) = 3941.
W = ("a", "walk")
FACTOR_PLAN = dataclasses.replace(PLAN, method_factors={W: 2.0})
COUNTS_W = ({A: 10.0, W: 5.0}, {A: 4.0, W: 2.0}, {A: 2.0, W: 1.0})


def test_size_rungs_applies_measured_method_factors_in_full_runs() -> None:
    sizing = size_rungs(
        arms_per_rung=COUNTS_W,
        plan=FACTOR_PLAN,
        remaining_usd=10.0,
        usd_per_s=0.01,
        min_r0_steps=1000,
    )

    assert sizing.rungs == (2955, 8865, 26595)
    assert sizing.cost_usd == pytest.approx((0.891, 2.48760, 5.619))


def test_size_rungs_counts_an_unprobed_method_as_uniform_in_full_runs() -> None:
    # "visited" has no probe: x1 in a full run (the loop's learned correction covers it), so
    # this is COUNTS with b's uniform arms relabelled, and r0 stays 2955.
    v = ("b", "visited")
    counts = ({A: 10.0, v: 5.0}, {A: 4.0, v: 2.0}, {A: 2.0, v: 1.0})

    sizing = size_rungs(
        arms_per_rung=counts,
        plan=FACTOR_PLAN,
        remaining_usd=10.0,
        usd_per_s=0.01,
        min_r0_steps=1000,
    )

    assert sizing.rungs[0] == 2955


def test_full_ladder_projection_sizes_the_full_ladder_at_the_hypothetical_cap() -> None:
    # remaining = cap 25 - 5 margin - 10 calibration cap = 10 USD: the size_rungs case above.
    proj = project_full_ladder(
        arms_per_rung=COUNTS_W,
        plan=FACTOR_PLAN,
        cap_usd=25.0,
        usd_per_s=0.01,
        min_r0_steps=3000,
    )

    assert proj.remaining_usd == pytest.approx(10.0)
    assert proj.arms_per_rung == pytest.approx((15.0, 6.0, 3.0))
    assert proj.r0 == 2955
    assert proj.cost_usd == pytest.approx((0.891, 2.48760, 5.619))
    assert proj.would_exit_5 is True  # 2955 < 3000
    ok = project_full_ladder(
        arms_per_rung=COUNTS_W,
        plan=FACTOR_PLAN,
        cap_usd=25.0,
        usd_per_s=0.01,
        min_r0_steps=1000,
    )
    assert ok.would_exit_5 is False


def test_full_ladder_projection_with_nothing_left_after_calibration_exits_5() -> None:
    proj = project_full_ladder(
        arms_per_rung=COUNTS_W,
        plan=FACTOR_PLAN,
        cap_usd=12.0,  # 12 - 5 - 10 < 0
        usd_per_s=0.01,
        min_r0_steps=1,
    )

    assert (proj.r0, proj.would_exit_5) == (0, True)
    assert proj.remaining_usd == pytest.approx(-3.0)


def test_settings_project_cap_default_and_override(tmp_path: Path) -> None:
    assert load_settings(_env(tmp_path)).project_cap_usd == 100.0
    s = load_settings(_env(tmp_path, SWEEP_PROJECT_CAP_USD="60"))
    assert s.project_cap_usd == 60.0
    with pytest.raises(SettingsError, match="SWEEP_PROJECT_CAP_USD"):
        load_settings(_env(tmp_path, SWEEP_PROJECT_CAP_USD="lots"))


def test_smoke_logs_the_full_ladder_projection_everywhere(
    tmp_path: Path,
    clock: Any,
    pool_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = tmp_path / "sweep"
    sink = RecordingSink()

    assert _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory, sink) == 0

    out = capsys.readouterr().out
    [line] = [x for x in out.splitlines() if x.startswith("full-ladder projection")]
    assert "smoke-calibrated" in line and "cap 100.00 USD" in line
    report = json.loads((state / "report.json").read_text())
    extra = report["extra"]
    assert extra["full_projection_cap_usd"] == 100.0
    assert extra["full_projection_min_r0_steps"] == 512.0
    assert extra["full_projection_exits_5"] in ("yes", "no")
    assert extra["full_projection_r0"] >= 0
    for k in range(3):
        assert extra[f"full_projection_arms_r{k}"] > 0
        assert extra[f"full_projection_usd_r{k}"] >= 0
    # rung 0 is the whole full arm space (rung-0 arms are always present)
    assert extra["full_projection_arms_r0"] == float(len(arm_space("rung0")))
    assert "smoke-calibrated" in str(extra["full_projection_note"])
    assert "full_projection_note" in (state / "report.md").read_text()
    assert sink.summaries[-1]["full_projection_r0"] == extra["full_projection_r0"]


def test_full_runs_do_not_log_a_full_ladder_projection(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    state = tmp_path / "sweep"
    _run(_env(tmp_path, SWEEP_R0_STEPS="10000000"), clock, pool_factory)

    extra = json.loads((state / "report.json").read_text())["extra"]
    assert not any(k.startswith("full_projection") for k in extra)


def test_size_rungs_charges_a_full_job_wall_when_the_pool_is_not_full() -> None:
    # One arm on 2 workers: the pod bills the whole job wall, r0 / (1000 / 2) + 4 s, not the
    # 1/n_workers share (r0 / 1000 + 2 s). 900 s budget -> r0 = floor((900 - 4) / 0.002).
    sizing = size_rungs(
        arms_per_rung=({A: 1.0},),
        plan=PLAN,
        remaining_usd=10.0,
        usd_per_s=0.01,
        min_r0_steps=1,
    )

    assert sizing.rungs[0] == 448_000
    assert sizing.cost_usd == pytest.approx(((448_000 * 0.002 + 4) * 0.01,))


def test_size_rungs_refuses_r0_below_the_minimum() -> None:
    with pytest.raises(BudgetTooSmallError) as info:
        size_rungs(
            arms_per_rung=COUNTS,
            plan=PLAN,
            remaining_usd=10.0,
            usd_per_s=0.01,
            min_r0_steps=3000,
        )
    message = str(info.value)
    for needle in (
        "2955",
        "3000",
        "SWEEP_MIN_R0_STEPS",
        "fewer tasks",
        "fewer arms",
        "fewer seeds",
    ):
        assert needle in message


def test_size_rungs_raises_when_overhead_alone_exceeds_the_budget() -> None:
    # 0.9 * 0.5 / 0.01 = 45 s < 96 s of overhead alone
    with pytest.raises(BudgetTooSmallError, match="r0"):
        size_rungs(
            arms_per_rung=COUNTS,
            plan=PLAN,
            remaining_usd=0.5,
            usd_per_s=0.01,
            min_r0_steps=1,
        )


def test_settings_min_r0_default_and_override(tmp_path: Path) -> None:
    env = _env(tmp_path)
    del env["SWEEP_MIN_R0_STEPS"]
    assert load_settings(env).min_r0_steps == 2_000_000
    assert load_settings(_env(tmp_path, SWEEP_MIN_R0_STEPS="512")).min_r0_steps == 512


def test_main_returns_5_when_r0_would_be_below_the_minimum(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    # Smoke-patch item 1: the exit-5 floor is for full runs only.
    env = _env(tmp_path, SWEEP_MIN_R0_STEPS="1000000000000")

    assert _run(env, clock, pool_factory) == 5
    assert not (tmp_path / "sweep" / "scheduler.json").exists()


def test_smoke_ignores_the_full_run_floor_and_clamps_r0_to_its_own(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    # Smoke uses SWEEP_SMOKE_MIN_R0_STEPS (not SWEEP_MIN_R0_STEPS) as a floor it raises r0 to,
    # never an exit-5 refusal. A cap of 2 USD leaves too little for a 300k-step smoke rung.
    env = _env(
        tmp_path,
        SWEEP_SMOKE="1",
        SWEEP_CAP_USD="2",
        SWEEP_MIN_R0_STEPS="1000000000000",
        SWEEP_SMOKE_MIN_R0_STEPS="300000",
    )

    assert _run(env, clock, pool_factory) == 0
    budgets = json.loads((tmp_path / "sweep" / "scheduler.json").read_text())["config"][
        "rung_budgets"
    ]
    assert budgets[0] == 300_000


def test_settings_smoke_sizing_defaults_and_overrides(tmp_path: Path) -> None:
    s = load_settings(_env(tmp_path))
    assert (s.smoke_min_r0_steps, s.smoke_r0_max_steps) == (100_000, 1_000_000)
    s = load_settings(
        _env(
            tmp_path,
            SWEEP_SMOKE_MIN_R0_STEPS="5000",
            SWEEP_SMOKE_R0_MAX_STEPS="70000",
        )
    )
    assert (s.smoke_min_r0_steps, s.smoke_r0_max_steps) == (5000, 70_000)
    with pytest.raises(SettingsError, match="SWEEP_SMOKE_R0_MAX_STEPS"):
        load_settings(
            _env(
                tmp_path,
                SWEEP_SMOKE_MIN_R0_STEPS="5000",
                SWEEP_SMOKE_R0_MAX_STEPS="4000",
            )
        )


# Smoke sizing (smoke-patch item 1). Counts are keyed (family, method). PLAN has a and b
# calibrated; "c" is not, so it runs on the fallback (500 steps/s, 4 s overhead).
SMOKE_PLAN = dataclasses.replace(PLAN, method_factors={("a", "walk"): 2.0})
SMOKE_COUNTS = {
    ("a", "uniform"): 1.0,  # calibrated family, uniform: x1
    ("a", "walk"): 1.0,  # measured probe: x2
    ("a", "visited"): 1.0,  # no probe: x3
    ("c", "uniform"): 1.0,  # family not calibrated: x3
}


def test_smoke_sizing_spends_half_of_remaining_with_unprobed_jobs_at_3x() -> None:
    # Pod-s per r0 step, full pool (2 workers, aggregate steps/s):
    #   (1 + 2 + 3) / 1000 + 3 / 500 = 0.012; overhead 4 jobs * 4 s / 2 workers = 8 s.
    # One job wall (the slowest): c/uniform 3 * 2 / 500 = 0.012 per step + 4 s.
    # Budget 0.5 * 10 USD / 0.01 USD/s = 500 s:
    #   full pool (500 - 8) / 0.012 = 41000; wall (500 - 4) / 0.012 = 41333.3 -> r0 = 41000.
    sizing = size_smoke_rung(
        arms=SMOKE_COUNTS,
        plan=SMOKE_PLAN,
        remaining_usd=10.0,
        usd_per_s=0.01,
        min_r0_steps=1000,
        max_r0_steps=1_000_000,
    )

    assert sizing.rungs == (41_000, 123_000, 369_000)
    # max(41000 * 0.012 + 8, 41000 * 0.012 + 4) * 0.01 = 5.00 = half of 10 USD
    assert sizing.cost_usd == pytest.approx((5.0,))


def test_smoke_sizing_caps_r0_at_the_maximum() -> None:
    sizing = size_smoke_rung(
        arms=SMOKE_COUNTS,
        plan=SMOKE_PLAN,
        remaining_usd=10.0,
        usd_per_s=0.01,
        min_r0_steps=1000,
        max_r0_steps=20_000,
    )

    assert sizing.rungs[0] == 20_000
    # max(20000 * 0.012 + 8, 20000 * 0.012 + 4) * 0.01 = 2.48
    assert sizing.cost_usd == pytest.approx((2.48,))


def test_smoke_sizing_raises_r0_to_the_floor_instead_of_refusing() -> None:
    # 0.5 * 0.5 / 0.01 = 25 s: wall (25 - 4) / 0.012 = 1750 < the 100k floor.
    sizing = size_smoke_rung(
        arms=SMOKE_COUNTS,
        plan=SMOKE_PLAN,
        remaining_usd=0.5,
        usd_per_s=0.01,
        min_r0_steps=100_000,
        max_r0_steps=1_000_000,
    )

    assert sizing.rungs[0] == 100_000
    assert sizing.cost_usd == pytest.approx(((100_000 * 0.012 + 8) * 0.01,))


# --- the main loop -----------------------------------------------------------------------------


def test_main_stops_when_budget_exhausted(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    # Each rung-0 job: 1e7 steps at 256 * 100 / sqrt(4) = 12,800 steps/s per job + 2 s = 783 s
    # of wall clock. The pod bills 7.83 USD until the last job ends: after about 1.5 USD of
    # calibration, 4 jobs start together (1.5 + 7.83 <= 15), and the next 4 would end at
    # 9.3 + 7.83 > 15, so the guard stops there.
    env = _env(tmp_path, SWEEP_R0_STEPS="10000000")
    state = tmp_path / "sweep"

    code = _run(env, clock, pool_factory)

    jobs = _sweep_jobs(world)
    assert code == 0
    assert 0 < len(jobs) < len(arm_space("rung0"))
    assert _ledger(state)[-1]["spent"] <= 20.0 - 5.0 + 1e-9
    assert (state / "report.md").exists() and (state / "report.json").exists()


def test_main_waits_for_in_flight_jobs_before_report(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    env = _env(tmp_path, SWEEP_R0_STEPS="10000000")
    state = tmp_path / "sweep"

    _run(env, clock, pool_factory)

    submitted = {(s.spec.arm_id(), s.spec.seed) for s in _sweep_jobs(world)}
    recorded = {(r["arm_id"], r["seed"]) for r in _results(state)}
    assert submitted == recorded


def test_main_resumes_from_state_dir(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    state = tmp_path / "sweep"
    _run(
        _env(tmp_path, SWEEP_CAP_USD="20", SWEEP_R0_STEPS="10000000"),
        clock,
        pool_factory,
    )
    first_done = {(r["arm_id"], r["seed"]) for r in _results(state)}
    first_spent = _ledger(state)[-1]["spent"]
    budgets = json.loads((state / "scheduler.json").read_text())["config"][
        "rung_budgets"
    ]

    world.session = 1
    # 9.3 spent + 0.6 restart allowance + 7.83 for the next 4 jobs <= 28 - 5
    _run(_env(tmp_path, SWEEP_CAP_USD="28", SWEEP_R0_STEPS="999"), clock, pool_factory)

    second = _sweep_jobs(world, session=1)
    assert [s for s in world.submissions if s.session == 1 and s.job.rung < 0] == []
    assert first_done and second
    assert first_done.isdisjoint({(s.spec.arm_id(), s.spec.seed) for s in second})
    # the saved rung budgets win over a changed override
    assert {s.job.step_budget for s in second} == {budgets[0]}
    assert _ledger(state)[-1]["spent"] > first_spent


def test_main_resume_ledger_counts_first_session(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    state = tmp_path / "sweep"
    _run(
        _env(tmp_path, SWEEP_CAP_USD="20", SWEEP_R0_STEPS="10000000"),
        clock,
        pool_factory,
    )
    n_lines = len(_ledger(state))
    first_spent = _ledger(state)[-1]["spent"]

    world.session = 1
    _run(
        _env(tmp_path, SWEEP_CAP_USD="20", SWEEP_R0_STEPS="10000000"),
        clock,
        pool_factory,
    )

    # session 2 starts from session 1's spend plus the restart allowance and, with the same
    # cap, cannot start any new job
    assert _ledger(state)[n_lines]["spent"] >= first_spent
    assert _sweep_jobs(world, session=1) == []


def test_smoke_mode_runs_uniform_and_visited_on_pendulum_and_uniform_on_maze_trap(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    """Final-review I8 adds the maze_trap uniform run and the arm-family probes.

    r0 is sized automatically (smoke-patch item 1): maze_trap's family is not calibrated in
    smoke, and the fallback plan (the slowest measured steps/s, at base N=128) is twice as fast
    as this fake world runs N=128, so sizing must leave room (50% of remaining, unprobed jobs
    at 3x) for every smoke job to finish. The cap is low enough that the sizing, not the
    SWEEP_SMOKE_R0_MAX_STEPS cap, sets r0."""
    env = _env(tmp_path, SWEEP_SMOKE="1", SWEEP_CAP_USD="5")
    state = tmp_path / "sweep"
    sink = RecordingSink()

    code = _run(env, clock, pool_factory, sink)

    r0, r1, r2 = json.loads((state / "scheduler.json").read_text())["config"][
        "rung_budgets"
    ]
    sweep = _sweep_jobs(world)
    calib_tasks = {s.spec.task for s in world.submissions if s.job.rung < 0}
    report = json.loads((state / "report.json").read_text())
    assert code == 0
    probes = {
        s.spec.method
        for s in world.submissions
        if s.job.rung < 0 and s.spec.task == "arm_sparse"
    }
    assert sorted(s.job.arm.label() for s in sweep) == sorted(
        [ARM_UNIFORM.label(), ARM_VISITED.label(), "maze_trap/uniform/-"]
    )
    assert {(s.job.rung, s.job.step_budget) for s in sweep} == {(0, r0)}
    assert (r1, r2) == (3 * r0, 9 * r0)
    assert calib_tasks <= {"pendulum_dense", "pendulum_sparse", "arm_sparse"}
    assert probes == {"visited", "buffer_scored"}
    assert sorted(len(g["arms"]) for g in report["groups"]) == [1, 2]
    assert "per_task" in sink.tables
    # every smoke job finished ok under the automatic r0, and the sizing (not a cap) set r0
    assert sorted(r["status"] for r in _results(state)) == ["ok", "ok", "ok"]
    assert 100_000 <= r0 < 1_000_000
    assert _ledger(state)[-1]["spent"] <= 5.0 - 0.5 + 1e-9


def test_main_logs_report_to_sweep_level_sink(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    sink = RecordingSink()

    _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory, sink)

    assert sorted(sink.tables) == ["arms", "overall", "per_task", "unranked"]
    assert sink.summaries[-1]["spent_usd"] > 0
    assert sink.closed is True


def test_main_writes_calibration_json(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory)

    record = json.loads((tmp_path / "sweep" / "calibration.json").read_text())
    assert record["version"] == 1
    assert record["spent_usd"] > 0


# --- SweepLoop with a stub scheduler -------------------------------------------------------------


def _spec(job: Job) -> RunSpec:
    return RunSpec(
        task=job.arm.task,
        method=job.arm.method,
        method_kw=job.arm.method_kwargs(),
        score=None,
        seed=job.seed,
        step_budget=job.step_budget,
        num_envs=64,
        rollout_len=8,
        ppo=dataclasses.replace(START_CONFIG, minibatch_size=128),
        eval_every_steps=job.step_budget,
        device="cpu",
    )


class StubScheduler:
    """Hands out `jobs` once; records reports; `live` decides `is_live`."""

    def __init__(
        self, jobs: list[Job], live: Callable[[Job], bool] = lambda j: True
    ) -> None:
        self._pending = list(jobs)
        self._running: set[Job] = set()
        self.live = live
        self.reports: list[RunResult] = []
        self.report_error: Exception | None = None

    def next_jobs(self, k: int, *, max_rung: int | None = None) -> list[Job]:
        pending = [j for j in self._pending if max_rung is None or j.rung <= max_rung]
        out = pending[:k]
        self._pending = [j for j in self._pending if j not in out]
        self._running |= set(out)
        return out

    def report(self, result: RunResult) -> None:
        self.reports.append(result)
        if self.report_error is not None:
            err, self.report_error = self.report_error, None
            self._running = {j for j in self._running if j.seed != result.seed}
            raise err
        self._running = {j for j in self._running if j.seed != result.seed}

    def is_live(self, job: Job) -> bool:
        return self.live(job)

    def done(self) -> bool:
        return not self._pending and not self._running

    def promotions(self) -> None:
        pass

    def states(self) -> tuple[()]:
        return ()


def _loop(
    tmp_path: Path,
    sched: StubScheduler,
    pool_factory: Any,
    clock: Any,
    sink: Any = None,
) -> SweepLoop:
    guard = BudgetGuard(
        cap_usd=1000.0,
        margin_usd=5.0,
        gpu_usd_per_h=36.0,
        volume_usd_per_h=0.0,
        ledger_path=tmp_path / "ledger.jsonl",
        clock=clock,
    )
    cfg = RunnerConfig(
        project="p", group="g", wandb_mode="disabled", runs_dir=tmp_path / "runs"
    )
    plan = Plan(n_workers=2, families={}, fallback=_family("fallback", 1000.0, 0.0))
    return SweepLoop(
        scheduler=sched,  # type: ignore[arg-type]
        pool=pool_factory(2, "cpu", cfg),
        guard=guard,
        plan=plan,
        make_spec=_spec,
        clock=clock,
        smoke=False,
        sink=sink or RecordingSink(),
    )


def _jobs(n: int, step_budget: int = 100_000) -> list[Job]:
    return [
        Job(arm=ARM_UNIFORM, seed=s, rung=0, step_budget=step_budget) for s in range(n)
    ]


def test_loop_kills_job_whose_arm_is_no_longer_live_and_reports_it_failed(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    sched = StubScheduler(_jobs(1), live=lambda job: False)

    reason = _loop(tmp_path, sched, pool_factory, clock).run()

    (result,) = sched.reports
    assert reason == "done"
    assert (result.status, result.error) == (
        "failed",
        "killed: job no longer live (arm pruned, failed or moved rung)",
    )
    assert math.isnan(result.auc)


def test_loop_keeps_going_after_conflicting_duplicate(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    sched = StubScheduler(_jobs(2))
    sched.report_error = ConflictingResultError("differs")

    reason = _loop(tmp_path, sched, pool_factory, clock).run()

    assert reason == "done"
    assert len(sched.reports) == 2


def test_loop_drops_result_without_running_job(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    sched = StubScheduler(_jobs(2))
    sched.report_error = ValueError("result ('x', 0) matches no running job")

    reason = _loop(tmp_path, sched, pool_factory, clock).run()

    assert reason == "done"
    assert len(sched.reports) == 2


def test_loop_turns_ok_result_with_nan_auc_into_failed(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    sched = StubScheduler(_jobs(1))
    real_finish = world.finish
    world.finish = lambda spec, n, cfg: dataclasses.replace(
        real_finish(spec, n, cfg), auc=math.nan
    )

    _loop(tmp_path, sched, pool_factory, clock).run()

    (result,) = sched.reports
    assert (result.status, result.error) == (
        "failed",
        "ok result with non-finite auc nan",
    )


def test_loop_counts_killed_jobs_in_spend(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    sched = StubScheduler(_jobs(1, step_budget=10**6), live=lambda job: False)
    loop = _loop(tmp_path, sched, pool_factory, clock)
    t0 = clock.t

    loop.run()

    # the killed job's pod time is in the ledger: spend is wall clock x price, flushed at the end
    assert clock.t > t0
    assert _ledger(tmp_path)[-1]["spent"] == pytest.approx((clock.t - t0) * 0.01)


def test_main_hard_stop_kills_jobs_slower_than_estimated_and_keeps_them_for_resume(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    # Sweep jobs (1e7 steps) take 3x the calibrated 783 s, so the 4 jobs started together are
    # still running when spend reaches cap - margin = 15 USD.
    world.slowdown = lambda spec: 3.0 if spec.step_budget >= 10**6 else 1.0
    state = tmp_path / "sweep"

    code = _run(_env(tmp_path, SWEEP_R0_STEPS="10000000"), clock, pool_factory)

    saved = json.loads((state / "scheduler.json").read_text())
    assert code == 0
    assert len(_sweep_jobs(world)) == 4
    assert not (state / "results.jsonl").exists() or _results(state) == []
    assert len(saved["running"]) == 4  # resubmitted by the next session
    # killed within one 5 s poll of the limit: at most 0.05 USD over
    assert 15.0 <= _ledger(state)[-1]["spent"] <= 15.0 + 0.05 + 1e-9


def test_main_returns_3_when_budget_already_spent(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    state = tmp_path / "sweep"
    state.mkdir()
    (state / "ledger.jsonl").write_text(json.dumps({"t": 0.0, "spent": 15.0}) + "\n")

    code = _run(_env(tmp_path), clock, pool_factory)

    assert code == 3
    assert world.submissions == []


# --- fix round 1: state checks, cap ceiling, live logging ----------------------------------------


def test_settings_reject_cap_above_the_default_cap(tmp_path: Path) -> None:
    with pytest.raises(
        SettingsError, match=r"^SWEEP_CAP_USD must be <= 100.0 \(CAP_USD_DEFAULT\)$"
    ):
        load_settings(_env(tmp_path, SWEEP_CAP_USD="150"))


def test_main_refuses_to_resume_a_smoke_state_dir_as_a_full_sweep(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory)
    n_before = len(world.submissions)

    code = _run(_env(tmp_path), clock, pool_factory)

    meta = json.loads((tmp_path / "sweep" / "sweep_meta.json").read_text())
    assert code == 4
    assert meta["mode"] == "smoke"
    assert len(world.submissions) == n_before


def test_main_refuses_state_dir_whose_arm_set_changed(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory)
    meta_path = tmp_path / "sweep" / "sweep_meta.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "arms_sha256": "0" * 64}))

    assert _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory) == 4


def test_main_refuses_scheduler_state_without_calibration_json(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory)
    (tmp_path / "sweep" / "calibration.json").unlink()
    n_before = len(world.submissions)

    code = _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory)

    assert code == 4
    assert len(world.submissions) == n_before


def test_main_logs_live_sweep_metrics(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    sink = RecordingSink()

    _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory, sink)

    live = [
        d for _, d in sink.logs if "jobs_done" in d
    ]  # the first log is calibration's
    assert len(live) >= 2
    assert "calibration_usd" in sink.logs[0][1]
    first, last = live[0], live[-1]
    keys = {"spent_usd", "projected_usd", "remaining_usd", "jobs_running", "jobs_done"}
    assert keys <= set(first)
    # all three smoke jobs finish (smoke-patch item 1: the maze_trap run no longer overruns)
    assert (last["jobs_done"], last["jobs_running"]) == (3, 0)
    # after rung 0 both pendulum arms are promoted to rung 1 (uniform outside the quota,
    # visited inside), and so is maze_trap's protected uniform
    assert (
        last["arms/promoted"],
        last["arms/pruned"],
        last["rung/pendulum_sparse"],
        last["rung/maze_trap"],
    ) == (3, 0, 1, 1)
    assert last["projected_usd"] >= last["spent_usd"]
    elapsed = [t for t, _ in sink.logs]
    assert elapsed == sorted(elapsed)


def test_loop_logs_projected_spend_including_jobs_in_flight(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    sched = StubScheduler(_jobs(1, step_budget=10**6))
    sink = RecordingSink()
    loop = _loop(tmp_path, sched, pool_factory, clock, sink=sink)

    loop.run()

    # one job of 10^6 / (1000 / 2) = 2000 s: while it runs, projected = spent + what is left
    running = [d for _, d in sink.logs if d["jobs_running"] == 1]
    assert running
    assert all(d["projected_usd"] > d["spent_usd"] for d in running)


# --- follow-up: live sink failures, corrupt meta ---------------------------------------------------


class BrokenSink(RecordingSink):
    """W&B is down: every call that would reach the network raises."""

    def log(self, elapsed_s: float, data: dict[str, float]) -> None:
        raise ConnectionError("wandb: network unreachable")

    def log_table(
        self, name: str, columns: list[str], rows: list[list[object]]
    ) -> None:
        raise ConnectionError("wandb: network unreachable")


def _sweep_metrics(state: Path) -> list[dict[str, Any]]:
    path = state / "sweep_metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_main_finishes_when_the_live_sink_raises(
    tmp_path: Path, clock: Any, pool_factory: Any, caplog: pytest.LogCaptureFixture
) -> None:
    state = tmp_path / "sweep"

    code = _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory, BrokenSink())

    rows = _sweep_metrics(state)
    live = [r for r in rows if r.get("type") == "log" and "jobs_done" in r]
    errors = [
        r
        for r in caplog.records
        if r.levelname == "ERROR" and "sweep sink" in r.getMessage()
    ]
    assert code == 0
    assert (state / "report.md").exists()
    assert (
        live and live[-1]["jobs_done"] == 3
    )  # all three smoke jobs (smoke-patch item 1)
    assert live[-1]["sink_errors"] >= 1
    assert errors and "network unreachable" in errors[0].getMessage()


def test_main_runs_when_the_sweep_sink_cannot_start(
    tmp_path: Path, clock: Any, pool_factory: Any, caplog: pytest.LogCaptureFixture
) -> None:
    def factory(project: str, group: str, mode: str) -> RecordingSink:
        raise ConnectionError("wandb.init: timed out")

    code = main(
        [],
        environ=_env(tmp_path, SWEEP_SMOKE="1"),
        clock=clock,
        pool_factory=pool_factory,
        report_sink_factory=factory,
        device="cpu",
    )

    rows = _sweep_metrics(tmp_path / "sweep")
    assert code == 0
    assert any("jobs_done" in r for r in rows)
    assert any(
        "timed out" in r.getMessage() for r in caplog.records if r.levelname == "ERROR"
    )


def test_main_exits_4_on_corrupt_sweep_meta(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    state = tmp_path / "sweep"
    state.mkdir()
    (state / "sweep_meta.json").write_text('{"mode": "smo')

    code = _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory)

    assert code == 4
    assert world.submissions == []


# --- no CPU fallback on a pod (final-review I1) --------------------------------------------------


def test_main_refuses_cpu_device_on_a_runpod_pod(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    env = _env(tmp_path, RUNPOD_POD_ID="pod-xyz")

    assert _run(env, clock, pool_factory) == 6
    assert not (tmp_path / "sweep" / "ledger.jsonl").exists()


def test_main_allows_cpu_device_off_pod(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    assert _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory) == 0


# --- deadline, SIGTERM and offline report (final-review I2) ---------------------------------------


def test_settings_read_deadline_and_report_reserve(tmp_path: Path) -> None:
    s = load_settings(_env(tmp_path))
    assert (s.deadline_epoch, s.report_reserve_s) == (None, 300.0)

    s = load_settings(
        _env(tmp_path, SWEEP_DEADLINE_EPOCH="1800000000", SWEEP_REPORT_RESERVE_S="60")
    )
    assert (s.deadline_epoch, s.report_reserve_s) == (1_800_000_000.0, 60.0)


def test_settings_reject_bad_deadline(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="^SWEEP_DEADLINE_EPOCH"):
        load_settings(_env(tmp_path, SWEEP_DEADLINE_EPOCH="soon"))


def _deadline_run(
    tmp_path: Path, clock: Any, pool_factory: Any, deadline: float
) -> int:
    # Cap 100: the deadline (about 17 USD of fake pod time away), not cap - margin, binds.
    env = _env(
        tmp_path,
        SWEEP_CAP_USD="100",
        SWEEP_R0_STEPS="10000000",
        SWEEP_DEADLINE_EPOCH=str(deadline),
    )
    return main(
        [],
        environ=env,
        clock=clock,
        epoch_clock=clock,
        pool_factory=pool_factory,
        report_sink_factory=lambda project, group, mode: RecordingSink(),
        device="cpu",
    )


def test_deadline_refuses_jobs_that_would_run_past_it_and_writes_the_report(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    # Fake epoch = fake clock. Calibration ends near t = 1150; the first four 783 s jobs end
    # near 1933; four more would end near 2716 > 3000 - 300, so the guard refuses them.
    deadline = clock.t + 2000.0

    code = _deadline_run(tmp_path, clock, pool_factory, deadline)

    report = json.loads((tmp_path / "sweep" / "report.json").read_text())
    assert code == 0
    assert report["extra"]["stop_reason"].startswith(
        "deadline: the next job would run past"
    )
    assert clock.t <= deadline - 300.0
    assert (tmp_path / "sweep" / "report.md").exists()


def test_deadline_kills_jobs_still_running_and_writes_the_report(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    # Jobs take 3x their estimate, so the four started near 1150 are still running at the
    # stop time (deadline - 300 = 2700): they are killed and stay in flight for a resume.
    world.slowdown = lambda spec: 3.0
    world.enforce_timeouts = False
    deadline = clock.t + 2000.0
    state = tmp_path / "sweep"

    code = _deadline_run(tmp_path, clock, pool_factory, deadline)

    report = json.loads((state / "report.json").read_text())
    assert code == 0
    assert report["extra"]["stop_reason"] == (
        "deadline: stopped SWEEP_REPORT_RESERVE_S before SWEEP_DEADLINE_EPOCH"
    )
    assert clock.t <= deadline - 300.0 + 5.0  # one 5 s poll of slack
    assert json.loads((state / "scheduler.json").read_text())["running"]
    assert (state / "report.md").exists()


def test_sweep_report_rebuilds_the_report_from_saved_state(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    from resetsweep.sweep.main import report_main

    state = tmp_path / "sweep"
    assert _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory) == 0
    original = json.loads((state / "report.json").read_text())
    (state / "report.json").unlink()
    (state / "report.md").unlink()

    assert report_main(["--state-dir", str(state)]) == 0

    rebuilt = json.loads((state / "report.json").read_text())
    assert {k: v for k, v in rebuilt.items() if k != "extra"} == {
        k: v for k, v in original.items() if k != "extra"
    }
    assert rebuilt["extra"]["stop_reason"] == "rebuilt offline by sweep-report"
    assert (state / "report.md").exists()


def test_sweep_report_refuses_a_dir_without_state(tmp_path: Path) -> None:
    from resetsweep.sweep.main import report_main

    assert report_main(["--state-dir", str(tmp_path)]) == 2


def test_sweep_report_is_a_console_script() -> None:
    import tomllib

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    )
    assert (
        pyproject["project"]["scripts"]["sweep-report"]
        == "resetsweep.sweep.main:report_main"
    )


CHILD = '''
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from conftest import FakeClock, FakePool, FakeWorld
from resetsweep.sweep import main as m

state = Path(sys.argv[2])
world = FakeWorld(clock=FakeClock(1000.0))

class BlockingPool(FakePool):
    """Calibration runs in fake time; sweep jobs never finish, so the loop blocks in poll."""
    def poll(self, timeout_s):
        if any(j.rung >= 0 for j in self.running()):
            (state / "READY").write_text("1")
            time.sleep(0.05)
            return []
        return super().poll(timeout_s)

sys.exit(m.main([], clock=world.clock, device="cpu",
    pool_factory=lambda n, d, c: BlockingPool(world, n, d, c),
    report_sink_factory=lambda p, g, mode: m.NullReportSink()))
'''


def test_sigterm_kills_workers_saves_state_and_writes_the_report(
    tmp_path: Path,
) -> None:
    import os
    import signal
    import subprocess
    import sys
    import time

    state = tmp_path / "sweep"
    state.mkdir()
    env = {**os.environ, **_env(tmp_path, SWEEP_SMOKE="1", SWEEP_R0_STEPS="10000000")}
    env.pop("RUNPOD_POD_ID", None)
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(Path(__file__).parent), str(state)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(1200):  # at most 60 s for imports and fake calibration
            if (state / "READY").exists() or proc.poll() is not None:
                break
            time.sleep(0.05)
        assert (state / "READY").exists(), proc.communicate(timeout=30)[0]
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert proc.returncode == 128 + signal.SIGTERM, out
    report = json.loads((state / "report.json").read_text())
    assert report["extra"]["stop_reason"] == "terminated by SIGTERM"
    assert (state / "report.md").exists()
    assert json.loads((state / "scheduler.json").read_text())["running"]


# --- per-method estimates and the hang guard (final-review I5) -----------------------------------


def test_sweep_jobs_get_per_method_ceilings_and_the_hang_guard(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    # Smoke plan: pendulum_dense at N=256 on 4 workers = 51,200 steps/s, overhead 2 s.
    # Raw estimate for r0 = 1e7: 1e7 / (51,200 / 4) + 2 = 783.25 s. uniform factor 1;
    # visited has no probe on pendulum -> its step term counts 3x for the ceiling:
    # 3 * 781.25 + 2 = 2345.75 s. Ceiling = 4 x estimate.
    _run(
        _env(tmp_path, SWEEP_SMOKE="1", SWEEP_R0_STEPS="10000000"), clock, pool_factory
    )

    sweep = {s.job.arm.method: s for s in _sweep_jobs(world)}
    runs = tmp_path / "sweep" / "runs"
    assert sweep["uniform"].timeout_s == pytest.approx(4 * 783.25)
    assert sweep["visited"].timeout_s == pytest.approx(4 * (3 * 781.25 + 2))
    for s in sweep.values():
        assert s.hang_s == 900.0
        assert s.heartbeat == runs / f"{s.spec.arm_id()}-s{s.spec.seed}.heartbeat"


def test_settings_hang_s_override(tmp_path: Path) -> None:
    assert load_settings(_env(tmp_path)).hang_s == 900.0
    assert load_settings(_env(tmp_path, SWEEP_HANG_S="60")).hang_s == 60.0


# --- smoke margin (final-review I6) ---------------------------------------------------------------


def test_smoke_run_uses_the_50_cent_margin(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    # Cap 5: a full run has 5 - 5 = 0 left and refuses; a smoke run has 4.50 to spend.
    assert _run(_env(tmp_path / "full", SWEEP_CAP_USD="5"), clock, pool_factory) == 3
    assert (
        _run(
            _env(tmp_path / "smoke", SWEEP_CAP_USD="5", SWEEP_SMOKE="1"),
            clock,
            pool_factory,
        )
        == 0
    )

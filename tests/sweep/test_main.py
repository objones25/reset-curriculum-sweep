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
    size_rungs,
)
from resetsweep.sweep.pool import RunnerConfig
from resetsweep.sweep.scheduler import ConflictingResultError, Job
from resetsweep.train import RunResult, RunSpec

ARM_UNIFORM = next(a for a in arm_space("rung0") if a.label() == "pendulum_sparse/uniform/-")
ARM_VISITED = next(
    a for a in arm_space("rung0") if a.label() == "pendulum_sparse/visited[p_task=0.5]/-"
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

    def log_table(self, name: str, columns: list[str], rows: list[list[object]]) -> None:
        self.tables[name] = (columns, rows)

    def summary(self, data: dict[str, Any]) -> None:
        self.summaries.append(data)

    def close(self) -> None:
        self.closed = True


def _run(env: dict[str, str], clock: Any, pool_factory: Any, sink: RecordingSink | None = None) -> int:
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
    return [json.loads(line) for line in (state_dir / "ledger.jsonl").read_text().splitlines()]


def _results(state_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (state_dir / "results.jsonl").read_text().splitlines()]


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

    with pytest.raises(SettingsError, match="^SWEEP_GPU_USD_PER_H must be a finite number > 0$"):
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


def test_main_returns_2_on_bad_settings(tmp_path: Path, clock: Any, pool_factory: Any) -> None:
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


def test_size_rungs_puts_rung0_at_35_percent_of_remaining() -> None:
    # budget 0.35 * 10 / 0.01 = 350 s; overhead (10 * 4 + 5 * 4) / 2 = 30 s;
    # pod-s per step: 10 / 1000 + 5 / 500 = 0.02 -> r0 = 320 / 0.02 = 16,000
    rungs = size_rungs(arms_per_family={"a": 10, "b": 5}, plan=PLAN, remaining_usd=10.0, usd_per_s=0.01)

    assert rungs == (16_000, 48_000, 144_000)


def test_size_rungs_raises_when_budget_cannot_cover_one_rollout() -> None:
    # 0.35 * 0.5 / 0.01 = 17.5 s < 30 s of overhead alone
    with pytest.raises(BudgetTooSmallError, match="rung 0"):
        size_rungs(arms_per_family={"a": 10, "b": 5}, plan=PLAN, remaining_usd=0.5, usd_per_s=0.01)


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
    _run(_env(tmp_path, SWEEP_CAP_USD="20", SWEEP_R0_STEPS="10000000"), clock, pool_factory)
    first_done = {(r["arm_id"], r["seed"]) for r in _results(state)}
    first_spent = _ledger(state)[-1]["spent"]
    budgets = json.loads((state / "scheduler.json").read_text())["config"]["rung_budgets"]

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
    _run(_env(tmp_path, SWEEP_CAP_USD="20", SWEEP_R0_STEPS="10000000"), clock, pool_factory)
    n_lines = len(_ledger(state))
    first_spent = _ledger(state)[-1]["spent"]

    world.session = 1
    _run(_env(tmp_path, SWEEP_CAP_USD="20", SWEEP_R0_STEPS="10000000"), clock, pool_factory)

    # session 2 starts from session 1's spend plus the restart allowance and, with the same
    # cap, cannot start any new job
    assert _ledger(state)[n_lines]["spent"] >= first_spent
    assert _sweep_jobs(world, session=1) == []


def test_smoke_mode_runs_only_uniform_and_visited_on_pendulum(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    env = _env(tmp_path, SWEEP_SMOKE="1")
    state = tmp_path / "sweep"
    sink = RecordingSink()

    code = _run(env, clock, pool_factory, sink)

    r0, r1, r2 = json.loads((state / "scheduler.json").read_text())["config"]["rung_budgets"]
    sweep = _sweep_jobs(world)
    calib_tasks = {s.spec.task for s in world.submissions if s.job.rung < 0}
    report = json.loads((state / "report.json").read_text())
    assert code == 0
    assert sorted(s.job.arm.label() for s in sweep) == sorted([ARM_UNIFORM.label(), ARM_VISITED.label()])
    assert {(s.job.rung, s.job.step_budget) for s in sweep} == {(0, r0)}
    assert (r1, r2) == (3 * r0, 9 * r0)
    assert calib_tasks <= {"pendulum_dense", "pendulum_sparse"}
    assert [len(g["arms"]) for g in report["groups"]] == [2]
    assert "per_task" in sink.tables


def test_main_logs_report_to_sweep_level_sink(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    sink = RecordingSink()

    _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory, sink)

    assert sorted(sink.tables) == ["arms", "overall", "per_task", "unranked"]
    assert sink.summaries[-1]["spent_usd"] > 0
    assert sink.closed is True


def test_main_writes_calibration_json(tmp_path: Path, clock: Any, pool_factory: Any) -> None:
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

    def __init__(self, jobs: list[Job], live: Callable[[Job], bool] = lambda j: True) -> None:
        self._pending = list(jobs)
        self._running: set[Job] = set()
        self.live = live
        self.reports: list[RunResult] = []
        self.report_error: Exception | None = None

    def next_jobs(self, k: int) -> list[Job]:
        out, self._pending = self._pending[:k], self._pending[k:]
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
    tmp_path: Path, sched: StubScheduler, pool_factory: Any, clock: Any, sink: Any = None
) -> SweepLoop:
    guard = BudgetGuard(
        cap_usd=1000.0,
        margin_usd=5.0,
        gpu_usd_per_h=36.0,
        volume_usd_per_h=0.0,
        ledger_path=tmp_path / "ledger.jsonl",
        clock=clock,
    )
    cfg = RunnerConfig(project="p", group="g", wandb_mode="disabled", runs_dir=tmp_path / "runs")
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
    return [Job(arm=ARM_UNIFORM, seed=s, rung=0, step_budget=step_budget) for s in range(n)]


def test_loop_kills_job_whose_arm_is_no_longer_live_and_reports_it_failed(
    tmp_path: Path, clock: Any, pool_factory: Any
) -> None:
    sched = StubScheduler(_jobs(1), live=lambda job: False)

    reason = _loop(tmp_path, sched, pool_factory, clock).run()

    (result,) = sched.reports
    assert reason == "done"
    assert (result.status, result.error) == ("failed", "killed: job no longer live (arm pruned, failed or moved rung)")
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
    world.finish = lambda spec, n, cfg: dataclasses.replace(real_finish(spec, n, cfg), auc=math.nan)

    _loop(tmp_path, sched, pool_factory, clock).run()

    (result,) = sched.reports
    assert (result.status, result.error) == ("failed", "ok result with non-finite auc nan")


def test_loop_counts_killed_jobs_in_spend(tmp_path: Path, clock: Any, pool_factory: Any) -> None:
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
    with pytest.raises(SettingsError, match=r"^SWEEP_CAP_USD must be <= 100.0 \(CAP_USD_DEFAULT\)$"):
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


def test_main_logs_live_sweep_metrics(tmp_path: Path, clock: Any, pool_factory: Any) -> None:
    sink = RecordingSink()

    _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory, sink)

    live = [d for _, d in sink.logs if "jobs_done" in d]  # the first log is calibration's
    assert len(live) >= 2
    assert "calibration_usd" in sink.logs[0][1]
    first, last = live[0], live[-1]
    keys = {"spent_usd", "projected_usd", "remaining_usd", "jobs_running", "jobs_done"}
    assert keys <= set(first)
    assert (last["jobs_done"], last["jobs_running"]) == (2, 0)
    # after rung 0 both arms are promoted to rung 1 (uniform outside the quota, visited inside)
    assert (last["arms/promoted"], last["arms/pruned"], last["rung/pendulum_sparse"]) == (2, 0, 1)
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

    def log_table(self, name: str, columns: list[str], rows: list[list[object]]) -> None:
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
    errors = [r for r in caplog.records if r.levelname == "ERROR" and "sweep sink" in r.getMessage()]
    assert code == 0
    assert (state / "report.md").exists()
    assert live and live[-1]["jobs_done"] == 2
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
    assert any("timed out" in r.getMessage() for r in caplog.records if r.levelname == "ERROR")


def test_main_exits_4_on_corrupt_sweep_meta(
    tmp_path: Path, clock: Any, pool_factory: Any, world: Any
) -> None:
    state = tmp_path / "sweep"
    state.mkdir()
    (state / "sweep_meta.json").write_text('{"mode": "smo')

    code = _run(_env(tmp_path, SWEEP_SMOKE="1"), clock, pool_factory)

    assert code == 4
    assert world.submissions == []

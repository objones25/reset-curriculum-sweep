"""Final report (spec section 6): per-task table, overall normalised IQM, unranked arms.

A real `Scheduler` is driven to the end with a fake arm id and hand-picked AUCs, the way
tests/sweep/test_scheduler.py does. Every seed of an arm gets the same AUC, so each IQM is
that AUC and the normalised values below are hand-computed.
"""

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from resetsweep.sweep.arms import Arm, arm_space
from resetsweep.sweep.report import (
    NullReportSink,
    WandbReportSink,
    build_report,
    write_report,
)
from resetsweep.sweep.scheduler import Job, Scheduler
from resetsweep.train import RunResult

TASK_A = "acrobot_swingup_sparse"
TASK_B = "arm_sparse"
BUDGETS = (10, 30, 90)
TAILS = (
    "uniform/-",
    "point/-",
    "visited[p_task=0.0]/-",
    "visited[p_task=0.2]/-",
    "visited[p_task=0.5]/-",
)


def _arm(task: str, tail: str) -> Arm:
    (match,) = [a for a in arm_space("rung0") if a.label() == f"{task}/{tail}"]
    return match


ARMS = [_arm(t, tail) for t in (TASK_A, TASK_B) for tail in TAILS]
# A: visited[0.0] wins; B: visited[0.2] wins. uniform below point below the winner.
AUC = {
    f"{TASK_A}/uniform/-": 0.1,
    f"{TASK_A}/point/-": 0.3,
    f"{TASK_A}/visited[p_task=0.0]/-": 0.9,
    f"{TASK_B}/uniform/-": 0.2,
    f"{TASK_B}/point/-": 0.4,
    f"{TASK_B}/visited[p_task=0.2]/-": 0.6,
}
DEFAULT_AUC = 0.15  # the visited arms that get pruned at rung 0


def _arm_id(arm: Arm, step_budget: int) -> str:
    return hashlib.sha256(f"{arm.label()}@{step_budget}".encode()).hexdigest()[:16]


def _result(job: Job, auc: float, status: str = "ok") -> RunResult:
    return RunResult(
        arm_id=_arm_id(job.arm, job.step_budget),
        seed=job.seed,
        status=status,  # type: ignore[arg-type]
        error=None if status == "ok" else "CUDA OOM",
        auc=auc if status == "ok" else math.nan,
        final_success=0.0 if auc == 0.0 else auc,
        reached_reward=auc > 0,
        steps_to_first_reward=None,
        ever_succeeded_share=math.nan,
        env_steps_train=job.step_budget,
        env_steps_extra={},
        wall_s=1.0,
    )


def _drive(
    s: Scheduler,
    fails: Callable[[Job], bool] = lambda job: False,
    auc: Mapping[str, float] = AUC,
) -> None:
    for _ in range(50):
        if s.done():
            return
        for job in s.next_jobs(1000):
            status = "failed" if fails(job) else "ok"
            s.report(_result(job, auc.get(job.arm.label(), DEFAULT_AUC), status))
    raise AssertionError("scheduler did not finish in 50 passes")


def _finished(tmp_path: Path, fails: Callable[[Job], bool] = lambda job: False) -> Scheduler:
    s = Scheduler(arms=ARMS, rung_budgets=BUDGETS, state_dir=tmp_path / "state", arm_id_of=_arm_id)
    _drive(s, fails)
    return s


def _write(tmp_path: Path, s: Scheduler, **kw: object) -> dict:
    return write_report(s, tmp_path / "report.md", tmp_path / "report.json", NullReportSink(), **kw)  # type: ignore[arg-type]


class RecordingSink:
    def __init__(self) -> None:
        self.tables: dict[str, tuple[list[str], list[list[object]]]] = {}
        self.summaries: list[dict[str, float | str]] = []
        self.closed = False

    def log(self, elapsed_s: float, data: dict[str, float]) -> None:
        del elapsed_s, data

    def log_table(self, name: str, columns: list[str], rows: list[list[object]]) -> None:
        self.tables[name] = (columns, rows)

    def summary(self, data: dict[str, float | str]) -> None:
        self.summaries.append(data)

    def close(self) -> None:
        self.closed = True


def test_report_ranks_only_final_rung_arms_and_lists_pruned(tmp_path: Path) -> None:
    s = _finished(tmp_path)

    report = _write(tmp_path, s)

    ranked = {
        g["task"]: [(a["label"], a["rung"], a["seeds_ok"]) for a in g["arms"]]
        for g in report["groups"]
        if g["task"] in (TASK_A, TASK_B)
    }
    assert ranked == {
        TASK_A: [
            (f"{TASK_A}/visited[p_task=0.0]/-", 2, 5),
            (f"{TASK_A}/point/-", 2, 5),
            (f"{TASK_A}/uniform/-", 2, 5),
        ],
        TASK_B: [
            (f"{TASK_B}/visited[p_task=0.2]/-", 2, 5),
            (f"{TASK_B}/point/-", 2, 5),
            (f"{TASK_B}/uniform/-", 2, 5),
        ],
    }
    unranked = sorted(
        (a["label"], a["status"], a["rung"])
        for a in report["unranked"]
        if a["label"].split("/")[0] in (TASK_A, TASK_B)
    )
    assert unranked == sorted(
        [
            (f"{TASK_A}/visited[p_task=0.2]/-", "pruned", 0),
            (f"{TASK_A}/visited[p_task=0.5]/-", "pruned", 0),
            (f"{TASK_B}/visited[p_task=0.0]/-", "pruned", 0),
            (f"{TASK_B}/visited[p_task=0.5]/-", "pruned", 0),
        ]
    )


def test_report_normalises_per_task_between_uniform_and_best(tmp_path: Path) -> None:
    report = _write(tmp_path, _finished(tmp_path))

    norm = {
        a["label"]: a["normalised"]
        for g in report["groups"]
        if g["task"] in (TASK_A, TASK_B)
        for a in g["arms"]
    }

    # A: (0.3 - 0.1) / (0.9 - 0.1) = 0.25; B: (0.4 - 0.2) / (0.6 - 0.2) = 0.5
    assert norm == {
        f"{TASK_A}/visited[p_task=0.0]/-": pytest.approx(1.0),
        f"{TASK_A}/point/-": pytest.approx(0.25),
        f"{TASK_A}/uniform/-": 0.0,
        f"{TASK_B}/visited[p_task=0.2]/-": pytest.approx(1.0),
        f"{TASK_B}/point/-": pytest.approx(0.5),
        f"{TASK_B}/uniform/-": 0.0,
    }


def test_report_overall_is_iqm_of_normalised_scores_across_tasks(tmp_path: Path) -> None:
    report = _write(tmp_path, _finished(tmp_path))

    overall = [(row["config"], row["iqm_normalised"], row["n_tasks"]) for row in report["overall"]]

    # The dense groups that rung 1 adds score DEFAULT_AUC for every arm, uniform included, so
    # best == uniform: they have no scale and stay out of the overall IQM.
    # point: IQM of [0.25, 0.5] keeps both (floor(0.25 * 2) = 0 trimmed) = 0.375
    assert overall == [
        ("visited[p_task=0.0]/-", pytest.approx(1.0), 1),
        ("visited[p_task=0.2]/-", pytest.approx(1.0), 1),
        ("point/-", pytest.approx(0.375), 2),
        ("uniform/-", 0.0, 2),
    ]


def test_report_counts_seeds_ending_at_zero(tmp_path: Path) -> None:
    AUC_ZERO = {**AUC, f"{TASK_A}/uniform/-": 0.0}
    s = Scheduler(arms=ARMS, rung_budgets=BUDGETS, state_dir=tmp_path / "state", arm_id_of=_arm_id)
    _drive(s, auc=AUC_ZERO)

    report = _write(tmp_path, s)

    zeros = {a["label"]: a["zero_seeds"] for g in report["groups"] for a in g["arms"]}
    assert (zeros[f"{TASK_A}/uniform/-"], zeros[f"{TASK_A}/point/-"]) == (5, 0)


def test_report_anchorless_group_ranked_by_raw_auc_and_left_out_of_overall(
    tmp_path: Path,
) -> None:
    uniform_a = _arm(TASK_A, "uniform/-")
    s = _finished(tmp_path, fails=lambda job: job.arm == uniform_a)

    report = _write(tmp_path, s)

    group_a = next(g for g in report["groups"] if g["task"] == TASK_A)
    assert group_a["anchored"] is False
    assert [(a["label"], a["normalised"]) for a in group_a["arms"]] == [
        (f"{TASK_A}/visited[p_task=0.0]/-", None),
        (f"{TASK_A}/point/-", None),
    ]
    assert report["anchorless_groups"] == [{"task": TASK_A, "task_kw": {}}]
    # point now only counts task B: 0.5
    point = next(row for row in report["overall"] if row["config"] == "point/-")
    assert (point["iqm_normalised"], point["n_tasks"]) == (pytest.approx(0.5), 1)
    assert (f"{TASK_A}/uniform/-", "failed", 0) in [
        (a["label"], a["status"], a["rung"]) for a in report["unranked"]
    ]


def test_report_rank_rung_zero_ranks_rung0_survivors(tmp_path: Path) -> None:
    s = Scheduler(arms=ARMS, rung_budgets=BUDGETS, state_dir=tmp_path / "state", arm_id_of=_arm_id)
    for job in s.next_jobs(1000):
        s.report(_result(job, AUC.get(job.arm.label(), DEFAULT_AUC)))
    s.promotions()

    report = _write(tmp_path, s, rank_rung=0)

    group_a = next(g for g in report["groups"] if g["task"] == TASK_A)
    assert [(a["label"], a["rung"], a["seeds_ok"]) for a in group_a["arms"]] == [
        (f"{TASK_A}/visited[p_task=0.0]/-", 0, 1),
        (f"{TASK_A}/point/-", 0, 1),
        (f"{TASK_A}/uniform/-", 0, 1),
    ]


def test_report_writes_markdown_and_json(tmp_path: Path) -> None:
    s = _finished(tmp_path)

    report = _write(tmp_path, s, extra={"spent_usd": 12.5})

    md = (tmp_path / "report.md").read_text()
    assert json.loads((tmp_path / "report.json").read_text()) == json.loads(json.dumps(report))
    assert report["extra"] == {"spent_usd": 12.5}
    assert f"| {TASK_A}/visited[p_task=0.0]/- |" in md
    assert "## Unranked arms" in md


def test_report_logs_wandb_tables_to_sink(tmp_path: Path) -> None:
    sink = RecordingSink()

    write_report(_finished(tmp_path), tmp_path / "r.md", tmp_path / "r.json", sink)

    assert sorted(sink.tables) == ["arms", "overall", "per_task", "unranked"]
    columns, rows = sink.tables["per_task"]
    assert columns[:3] == ["group", "arm", "iqm_auc"]
    assert {f"{TASK_A}/point/-", f"{TASK_B}/point/-"} <= {row[1] for row in rows}
    assert sink.closed is False  # the caller owns the sink


def test_report_group_where_uniform_is_best_has_no_scale(tmp_path: Path) -> None:
    report = _write(tmp_path, _finished(tmp_path))

    dense = next(g for g in report["groups"] if g["task"] == "pendulum_dense")

    # every arm on pendulum_dense scored DEFAULT_AUC: best == uniform, nothing to divide by
    assert (dense["anchored"], dense["scaled"]) == (True, False)
    assert {a["normalised"] for a in dense["arms"]} == {None}


def test_wandb_report_sink_logs_tables_in_disabled_mode(tmp_path: Path) -> None:
    sink = WandbReportSink(project="p", group="sweep-test", mode="disabled", config={"sweep_id": "x"})
    try:
        sink.log(1.0, {"spent_usd": 0.5, "jobs_running": 2.0})
        report = write_report(_finished(tmp_path), tmp_path / "r.md", tmp_path / "r.json", sink)
    finally:
        sink.close()

    assert report["rank_rung"] == 2


def test_wandb_report_sink_rejects_secret_like_config_keys() -> None:
    with pytest.raises(ValueError, match=r"secret-like keys \['api_key'\]"):
        WandbReportSink(project="p", group="g", mode="disabled", config={"api_key": "***"})


def _to_final_rung(tmp_path: Path) -> tuple[Scheduler, list[Job]]:
    """Drive every rung below the last; return the scheduler and its first final-rung jobs."""
    s = Scheduler(arms=ARMS, rung_budgets=BUDGETS, state_dir=tmp_path / "state", arm_id_of=_arm_id)
    for _ in range(50):
        jobs = s.next_jobs(1000)
        final = [j for j in jobs if j.rung == 2]
        if final:
            return s, final
        for job in jobs:
            s.report(_result(job, AUC.get(job.arm.label(), DEFAULT_AUC)))
    raise AssertionError("never reached the final rung")


def test_report_ranks_on_common_seed_count_after_budget_stop_mid_final_rung(
    tmp_path: Path,
) -> None:
    # The budget stops the sweep with uniform at 3 final seeds, point at 4, visited[0.0] at 5.
    s, final = _to_final_rung(tmp_path)
    v00 = _arm(TASK_A, "visited[p_task=0.0]/-")
    seeds = {_arm(TASK_A, "uniform/-"): 3, _arm(TASK_A, "point/-"): 4, v00: 5}
    v00_auc = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.1, 4: 0.1}  # seeds 3 and 4 must not count
    for job in final:
        if job.arm in seeds and job.seed < seeds[job.arm]:
            auc = v00_auc[job.seed] if job.arm == v00 else AUC[job.arm.label()]
            s.report(_result(job, auc))

    report = _write(tmp_path, s)

    group_a = next(g for g in report["groups"] if g["task"] == TASK_A)
    rows = {a["label"]: (a["seeds_ok"], a["seeds_used"], a["iqm_auc"]) for a in group_a["arms"]}
    assert (group_a["common_seeds"], group_a["insufficient_seeds"]) == (3, False)
    # With all 5 seeds the IQM would be mean(0.1, 0.9, 0.9) = 0.633; on seeds 0-2 it is 0.9.
    assert rows == {
        v00.label(): (5, 3, pytest.approx(0.9)),
        f"{TASK_A}/point/-": (4, 3, pytest.approx(0.3)),
        f"{TASK_A}/uniform/-": (3, 3, pytest.approx(0.1)),
    }


def test_report_flags_group_with_fewer_than_two_common_seeds(tmp_path: Path) -> None:
    s = Scheduler(arms=ARMS, rung_budgets=BUDGETS, state_dir=tmp_path / "state", arm_id_of=_arm_id)
    for job in s.next_jobs(1000):
        s.report(_result(job, AUC.get(job.arm.label(), DEFAULT_AUC)))

    report = _write(tmp_path, s, rank_rung=0)

    group_a = next(g for g in report["groups"] if g["task"] == TASK_A)
    md = (tmp_path / "report.md").read_text()
    assert (group_a["common_seeds"], group_a["insufficient_seeds"]) == (1, True)
    assert "insufficient seeds: ranked on 1 seed per arm" in md


def test_report_arm_table_has_mean_ci_bounds_rung_and_status(tmp_path: Path) -> None:
    sink = RecordingSink()

    report = write_report(_finished(tmp_path), tmp_path / "r.md", tmp_path / "r.json", sink)

    columns, rows = sink.tables["arms"]
    by_arm = {row[0]: dict(zip(columns, row, strict=True)) for row in rows}
    v00 = by_arm[f"{TASK_A}/visited[p_task=0.0]/-"]
    pruned = by_arm[f"{TASK_A}/visited[p_task=0.5]/-"]
    assert columns == ["arm", "group", "status", "rung", "mean_auc", "ci_low", "ci_high", "lcb", "ucb", "seeds_ok"]
    # 5 seeds at 0.9: sd 0 floored at 0.05, half = 2 * 0.05 / sqrt(5)
    assert (v00["status"], v00["rung"], v00["mean_auc"]) == ("final", 2, pytest.approx(0.9))
    assert v00["ucb"] == pytest.approx(0.9 + 0.1 / math.sqrt(5))
    assert (pruned["status"], pruned["rung"], pruned["lcb"], pruned["ucb"]) == ("pruned", 0, None, None)
    assert len(report["arms"]) == len(rows)


def test_build_report_calls_tick_for_each_group(tmp_path: Path) -> None:
    ticks: list[int] = []

    report = build_report(_finished(tmp_path), tick=lambda: ticks.append(1))

    assert len(ticks) >= len(report["groups"]) > 0

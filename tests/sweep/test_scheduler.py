"""UCB successive-rung scheduler (spec section 7, controller rulings for Task 22).

Hand-computed values are worked in the comments next to each assertion. Results are fed in
through the public `next_jobs` / `report` loop; arm ids come from a deterministic fake
(`_arm_id`), except in the cross-process test, which uses the real `spec_for(...).arm_id()`.
"""

import hashlib
import json
import math
import os
import random
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from resetsweep.ppo.update import START_CONFIG
from resetsweep.sweep.arms import Arm, arm_space
from resetsweep.sweep.scheduler import (
    RETRY_SEED_OFFSET,
    ConflictingResultError,
    Job,
    RunDefaults,
    Scheduler,
    group_of,
    spec_for,
    ucb_bounds,
)
from resetsweep.train import RunResult

TASK_A = "acrobot_swingup_sparse"
TASK_B = "arm_sparse"
BUDGETS = (10, 30, 90)
LIVE = {"active", "promoted", "final"}


def _arms(task: str) -> list[Arm]:
    """uniform, point and 6 others (phased + 5 visited) on `task`, in arm_space order."""
    keep = ("uniform", "point", "visited", "phased")
    return [a for a in arm_space("rung0") if a.task == task and a.method in keep]


def _arm(task: str, label_tail: str) -> Arm:
    (match,) = [a for a in arm_space("rung0") if a.label() == f"{task}/{label_tail}"]
    return match


UNIFORM_A = _arm(TASK_A, "uniform/-")
POINT_A = _arm(TASK_A, "point/-")
V00_A = _arm(TASK_A, "visited[p_task=0.0]/-")
V02_A = _arm(TASK_A, "visited[p_task=0.2]/-")
V05_A = _arm(TASK_A, "visited[p_task=0.5]/-")
SMALL_A = [UNIFORM_A, POINT_A, V00_A, V02_A, V05_A]


def _arm_id(arm: Arm, step_budget: int) -> str:
    return hashlib.sha256(f"{arm.label()}@{step_budget}".encode()).hexdigest()[:16]


def _new(tmp_path: Path, arms: list[Arm], **kw: object) -> Scheduler:
    return Scheduler(
        arms=arms,
        rung_budgets=BUDGETS,
        state_dir=tmp_path,
        arm_id_of=_arm_id,
        **kw,  # type: ignore[arg-type]
    )


def _result(job: Job, auc: float, *, status: str = "ok") -> RunResult:
    return RunResult(
        arm_id=_arm_id(job.arm, job.step_budget),
        seed=job.seed,
        status=status,  # type: ignore[arg-type]
        error=None if status == "ok" else "CUDA OOM",
        auc=auc,
        final_success=auc,
        reached_reward=False,
        steps_to_first_reward=None,
        ever_succeeded_share=math.nan,  # one object, so results compare equal (nan is nan)
        env_steps_train=job.step_budget,
        env_steps_extra={},
        wall_s=1.0,
    )


AucOf = Callable[[Job], float]


def _step(s: Scheduler, k: int, auc_of: AucOf) -> list[Job]:
    jobs = s.next_jobs(k)
    for job in jobs:
        s.report(_result(job, auc_of(job)))
    return jobs


def _by_label(aucs: dict[str, float], default: float = 0.0) -> AucOf:
    return lambda job: aucs.get(job.arm.label(), default)


def _rung_aucs(s: Scheduler, arm: Arm, rung: int) -> list[float]:
    rid = _arm_id(arm, BUDGETS[rung])
    return [r.auc for r in s.state(arm).results if r.arm_id == rid and r.status == "ok"]


def _finish_rung0_small(tmp_path: Path) -> Scheduler:
    """SMALL_A through rung 0 with V00_A on top: rung 1 on task A is uniform, point, V00_A."""
    s = _new(tmp_path, SMALL_A)
    _step(s, 100, _by_label({V00_A.label(): 0.9, V02_A.label(): 0.2}))
    s.promotions()
    return s


# --- rung 0 ------------------------------------------------------------------------------------


def test_rung0_schedules_every_arm_once_before_repeats(tmp_path: Path) -> None:
    arms = _arms(TASK_A) + _arms(TASK_B)
    s = _new(tmp_path, arms)

    first = s.next_jobs(100)
    second = s.next_jobs(100)

    assert [j.arm for j in first] == arms
    assert {(j.seed, j.rung, j.step_budget) for j in first} == {(0, 0, 10)}
    assert second == []


def test_rung0_jobs_respect_k(tmp_path: Path) -> None:
    arms = _arms(TASK_A)
    s = _new(tmp_path, arms)

    jobs = s.next_jobs(3)

    assert [j.arm for j in jobs] == arms[:3]


def test_promote_top_third_per_task(tmp_path: Path) -> None:
    arms = _arms(TASK_A) + _arms(TASK_B)
    s = _new(tmp_path, arms)
    # 6 non-protected arms per task -> ceil(6/3) = 2 promoted per task, plus uniform and point.
    aucs = {
        f"{TASK_A}/visited[p_task=0.8]/-": 0.7,
        f"{TASK_A}/phased[p_task=0.5]/-": 0.6,
        f"{TASK_A}/visited[p_task=0.2]/-": 0.5,
        f"{TASK_B}/visited[p_task=0.0]/-": 0.4,
        f"{TASK_B}/visited[p_task=1.0]/-": 0.3,
        f"{TASK_B}/visited[p_task=0.5]/-": 0.1,
    }
    _step(s, 100, _by_label(aucs))

    s.promotions()

    at_rung1 = {a.label() for a in arms if s.state(a).rung == 1}
    assert at_rung1 == {
        f"{TASK_A}/visited[p_task=0.8]/-",
        f"{TASK_A}/phased[p_task=0.5]/-",
        f"{TASK_A}/uniform/-",
        f"{TASK_A}/point/-",
        f"{TASK_B}/visited[p_task=0.0]/-",
        f"{TASK_B}/visited[p_task=1.0]/-",
        f"{TASK_B}/uniform/-",
        f"{TASK_B}/point/-",
    }


def test_promotion_sets_statuses(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)

    statuses = {a.label(): (s.state(a).rung, s.state(a).status) for a in SMALL_A}

    assert statuses == {
        UNIFORM_A.label(): (1, "promoted"),
        POINT_A.label(): (1, "promoted"),
        V00_A.label(): (1, "promoted"),
        V02_A.label(): (0, "pruned"),
        V05_A.label(): (0, "pruned"),
    }


def test_rung0_promotion_adds_rung1_expansion_arms(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    expected = arm_space("rung1_expand", survivors=[V00_A])

    added = [st for st in s.states() if st.arm not in SMALL_A]

    assert [st.arm for st in added] == expected
    assert {(st.rung, st.status, len(st.results)) for st in added} == {(1, "active", 0)}


def test_expansion_arms_get_rung1_seeds(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    expanded = set(arm_space("rung1_expand", survivors=[V00_A]))

    jobs = s.next_jobs(1000)

    seeds = {(j.arm, j.seed) for j in jobs if j.arm in expanded}
    assert seeds == {(a, i) for a in expanded for i in range(3)}
    assert {j.rung for j in jobs} == {1}


# --- UCB ---------------------------------------------------------------------------------------


def test_ucb_bounds_hand_computed_for_two_seeds() -> None:
    # iqm([0.50, 0.52]) = 0.51; sample sd = 0.0141421; half-width = 2 * 0.0141421 / sqrt(2) = 0.02
    lcb, ucb = ucb_bounds([0.50, 0.52], spread_sd=99.0, beta=2.0)

    assert (lcb, ucb) == pytest.approx((0.49, 0.53))


def test_ucb_bounds_use_spread_for_one_seed() -> None:
    lcb, ucb = ucb_bounds([0.48], spread_sd=0.25, beta=2.0)

    assert (lcb, ucb) == pytest.approx((-0.02, 0.98))


def test_ucb_bounds_unbounded_without_results() -> None:
    assert ucb_bounds([], spread_sd=0.25, beta=2.0) == (-math.inf, math.inf)


def test_ucb_prefers_uncertain_arm_with_close_mean(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    n_rung1 = sum(1 for st in s.states() if st.rung == 1)
    rung1 = {UNIFORM_A.label(): 0.50, POINT_A.label(): 0.48, V00_A.label(): 0.10}
    _step(s, n_rung1, _by_label(rung1))  # one seed each (dense arms score 0.0)
    # Task A means [0.50, 0.48, 0.10]: uniform has the highest UCB (same spread for all).
    (pick,) = _step(s, 1, lambda job: 0.52)
    assert (pick.arm, pick.seed) == (UNIFORM_A, 1)

    # uniform: [0.50, 0.52], sd 0.0141 floored to 0.05 -> UCB 0.51 + 2 * 0.05 / sqrt(2) = 0.581.
    # point: n=1, sd = sample std of task means [0.51, 0.48, 0.10] = 0.22855,
    #        UCB = 0.48 + 2 * 0.22855 = 0.937. V00_A: 0.10 + 0.457 = 0.557. Dense arms: 0.0.
    (job,) = s.next_jobs(1)

    assert job == Job(arm=POINT_A, seed=1, rung=1, step_budget=30)


def test_lone_result_uses_unit_spread(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    (dense,) = [
        a
        for a in arm_space("rung1_expand", survivors=[V00_A])
        if a.label() == "pendulum_dense/visited[p_task=0.0]/-"
    ]
    jobs = {(j.arm, j.seed): j for j in s.next_jobs(1000)}
    # Failures leave one pending retry each for point and `dense`.
    s.report(_result(jobs[(POINT_A, 0)], 0.0, status="failed"))
    s.report(_result(jobs[(POINT_A, 1)], 0.3))
    s.report(_result(jobs[(UNIFORM_A, 0)], 0.3))
    s.report(_result(jobs[(dense, 0)], 0.0, status="failed"))
    s.report(_result(jobs[(dense, 1)], 0.2))
    # point: task A means [0.3, 0.3] -> spread 0 -> UCB 0.3.
    # dense: the only arm with a result on pendulum_dense -> sd 1.0 -> UCB 0.2 + 2.0 = 2.2.

    assert s.next_jobs(1) == [Job(arm=dense, seed=1000, rung=1, step_budget=30)]


# --- pruning -----------------------------------------------------------------------------------


def test_prune_when_ucb_below_leader_lcb(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    jobs = {(j.arm, j.seed): j for j in s.next_jobs(1000)}
    # uniform [0.80, 0.82, 0.81]: sd 0.01 floored to 0.05, LCB = 0.81 - 2 * 0.05 / sqrt(3) = 0.7523.
    for seed, auc in enumerate((0.80, 0.82, 0.81)):
        s.report(_result(jobs[(UNIFORM_A, seed)], auc))
    # V00_A [0.10, 0.11]: UCB = 0.105 + 2 * 0.05 / sqrt(2) = 0.176 < 0.7523.
    for seed, auc in enumerate((0.10, 0.11)):
        s.report(_result(jobs[(V00_A, seed)], auc))

    assert (s.state(V00_A).rung, s.state(V00_A).status) == (1, "pruned")


def test_no_prune_while_ucb_reaches_leader_lcb(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    jobs = {(j.arm, j.seed): j for j in s.next_jobs(1000)}
    for seed, auc in enumerate((0.80, 0.82, 0.81)):
        s.report(_result(jobs[(UNIFORM_A, seed)], auc))
    # V00_A [0.78, 0.80]: UCB = 0.79 + 2 * 0.05 / sqrt(2) = 0.861 >= 0.7523.
    for seed, auc in enumerate((0.78, 0.80)):
        s.report(_result(jobs[(V00_A, seed)], auc))

    assert s.state(V00_A).status == "promoted"


def test_uniform_and_point_never_pruned(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    jobs = {(j.arm, j.seed): j for j in s.next_jobs(1000)}
    # V00_A [0.9] x3 has LCB 0.9; uniform and point at 0.0 x3 have UCB 0.0.
    for arm, auc in ((V00_A, 0.9), (UNIFORM_A, 0.0), (POINT_A, 0.0)):
        for seed in range(3):
            s.report(_result(jobs[(arm, seed)], auc))

    assert {s.state(a).status for a in (UNIFORM_A, POINT_A)} == {"promoted"}


def test_uniform_and_point_reach_final_rung_with_equal_seeds(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    good = _by_label({V00_A.label(): 0.9})
    for _ in range(10):
        _step(s, 1000, good)

    finals = {a: (s.state(a).status, len(_rung_aucs(s, a, 2))) for a in SMALL_A[:3]}

    assert s.done()
    assert finals == {
        UNIFORM_A: ("final", 5),
        POINT_A: ("final", 5),
        V00_A: ("final", 5),
    }


def test_protected_arms_do_not_use_promotion_quota(tmp_path: Path) -> None:
    # uniform and point score best at rung 0, yet 3 non-protected arms still promote 1.
    s = _new(tmp_path, SMALL_A)
    _step(
        s,
        100,
        _by_label({UNIFORM_A.label(): 0.9, POINT_A.label(): 0.8, V05_A.label(): 0.5}),
    )

    s.promotions()

    assert [s.state(a).rung for a in SMALL_A] == [1, 1, 0, 0, 1]


def test_rung1_promotion_waits_for_every_rung0_task(tmp_path: Path) -> None:
    small_b = [a for a in _arms(TASK_B) if a.method != "phased"][:5]
    s = _new(tmp_path, SMALL_A + small_b)
    for job in [j for j in s.next_jobs(100) if j.arm.task == TASK_A]:
        s.report(_result(job, 0.5))  # task B's rung-0 jobs stay running
    _step(s, 1000, lambda job: 0.5)  # all of rung 1 that exists so far

    s.promotions()

    assert {
        st.rung for st in s.states() if st.status in LIVE and st.arm.task != TASK_B
    } == {1}


# --- failures ----------------------------------------------------------------------------------


def test_failed_job_retried_with_seed_plus_offset(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    jobs = s.next_jobs(100)
    s.report(_result(jobs[2], 0.0, status="failed"))

    retry = s.next_jobs(100)

    assert retry == [Job(arm=V00_A, seed=RETRY_SEED_OFFSET, rung=0, step_budget=10)]
    assert RETRY_SEED_OFFSET == 1000


def test_failed_job_retried_once_then_excluded(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    jobs = s.next_jobs(100)
    for job in jobs:
        s.report(_result(job, 0.9 if job.arm == V00_A else 0.1, status="failed"))
    for job in s.next_jobs(100):  # retries: V00_A fails again, everyone else succeeds
        s.report(
            _result(job, 0.9 if job.arm == V00_A else 0.1, status=_fail_if(job, V00_A))
        )

    s.promotions()
    later = s.next_jobs(1000)

    assert (s.state(V00_A).rung, s.state(V00_A).status) == (0, "failed")
    assert V00_A not in {j.arm for j in later}


def _fail_if(job: Job, arm: Arm) -> str:
    return "failed" if job.arm == arm else "ok"


def test_retry_success_counts_for_ranking(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    jobs = s.next_jobs(100)
    for job in jobs:
        s.report(_result(job, 0.1, status=_fail_if(job, V05_A)))
    (retry,) = s.next_jobs(100)
    s.report(_result(retry, 0.9))

    s.promotions()

    assert (s.state(V05_A).rung, s.state(V05_A).status) == (1, "promoted")


# --- stopping ----------------------------------------------------------------------------------


def _final_live(s: Scheduler) -> int:
    return sum(1 for st in s.states() if st.status == "final")


def _run_to_final_rung(s: Scheduler, auc_of: AucOf) -> None:
    for _ in range(10):
        _step(s, 1000, auc_of)
        s.promotions()
        if _final_live(s):
            return
    raise AssertionError("no arm reached the final rung in 10 rounds")


def test_done_waits_for_all_final_seeds_even_when_separated(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    # The visited arms score 0.9 on every group and the controls 0.0, so every group's
    # leader separates as soon as it has 2 seeds (sd 0).
    good = lambda job: 0.9 if job.arm.method == "visited" else 0.0
    _run_to_final_rung(s, good)
    final = s.next_jobs(1000)
    (last,) = [
        j for j in final if j.arm.label() == "pendulum_dense/uniform/-" and j.seed == 4
    ]
    for job in [j for j in final if j != last]:
        s.report(_result(job, good(job)))
    s.report(_result(last, 0.0, status="failed"))  # its retry is pending, not running
    not_yet = s.done()

    retry = _step(s, 1000, good)

    assert not not_yet
    assert retry == [Job(arm=last.arm, seed=1004, rung=2, step_budget=90)]
    assert s.done()


def test_not_done_before_any_result(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)

    assert not s.done()


def test_done_when_no_jobs_remain(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    for _ in range(10):
        _step(s, 1000, lambda job: 0.5)

    assert s.done()
    assert s.next_jobs(1000) == []


def _maze_arms() -> list[Arm]:
    return [
        _arm("maze", tail)
        for tail in (
            "uniform/-",
            "point/-",
            "visited[p_task=0.0]/-",
            "visited[p_task=0.2]/-",
            "visited[p_task=0.5]/-",
        )
    ]


LANES5 = (("lanes", 5),)


def _variant(arm: Arm) -> Arm:
    return replace(arm, task_kw=LANES5)


def _finish_rung0_maze(tmp_path: Path) -> tuple[Scheduler, list[Arm]]:
    """maze through rung 0 with visited[p_task=0.0] on top: rung 1 adds maze[lanes=5]."""
    arms = _maze_arms()
    s = _new(tmp_path, arms)
    _step(s, 100, _by_label({arms[2].label(): 0.9}))
    s.promotions()
    return s, arms


def test_task_variant_is_pruned_only_within_its_group(tmp_path: Path) -> None:
    s, arms = _finish_rung0_maze(tmp_path)
    uniform, v00 = arms[0], arms[2]
    jobs = {(j.arm, j.seed): j for j in s.next_jobs(1000)}
    for seed, auc in enumerate((0.80, 0.82, 0.81)):  # maze leader LCB 0.7523
        s.report(_result(jobs[(uniform, seed)], auc))
    # maze[lanes=5] visited: UCB 0.176 < 0.7523, but it is in its own group, where it
    # is the only arm with results.
    for seed, auc in enumerate((0.10, 0.11)):
        s.report(_result(jobs[(_variant(v00), seed)], auc))

    assert s.state(_variant(v00)).status == "active"


def test_task_variant_gets_its_own_anchors_to_the_final_rung(tmp_path: Path) -> None:
    s, arms = _finish_rung0_maze(tmp_path)
    _run_to_final_rung(s, lambda job: 0.5)

    statuses = {s.state(_variant(a)).status for a in arms[:2]}

    assert statuses == {"final"}


def test_variant_anchors_enter_at_rung1(tmp_path: Path) -> None:
    s, arms = _finish_rung0_maze(tmp_path)

    entry = {(s.state(_variant(a)).rung, s.state(_variant(a)).status) for a in arms[:2]}

    assert entry == {(1, "active")}


def test_protected_arm_gets_three_retries(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    jobs = s.next_jobs(100)
    for job in jobs:
        s.report(_result(job, 0.1, status=_fail_if(job, UNIFORM_A)))
    seeds = []
    for _ in range(3):
        (retry,) = s.next_jobs(100)
        seeds.append(retry.seed)
        s.report(_result(retry, 0.0, status="failed"))

    assert seeds == [1000, 2000, 3000]
    assert s.state(UNIFORM_A).status == "failed"
    assert UNIFORM_A not in {j.arm for j in s.next_jobs(100)}


def test_protected_arm_survives_two_failures(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    for job in s.next_jobs(100):
        s.report(_result(job, 0.1, status=_fail_if(job, UNIFORM_A)))
    (retry1,) = s.next_jobs(100)
    s.report(_result(retry1, 0.0, status="failed"))
    (retry2,) = s.next_jobs(100)
    s.report(_result(retry2, 0.4))

    s.promotions()

    assert (s.state(UNIFORM_A).rung, s.state(UNIFORM_A).status) == (1, "promoted")


def _kill_uniform_a(s: Scheduler) -> None:
    for job in s.next_jobs(100):
        s.report(_result(job, 0.1, status=_fail_if(job, UNIFORM_A)))
    for _ in range(3):
        (retry,) = s.next_jobs(100)
        s.report(_result(retry, 0.0, status="failed"))


def test_anchorless_group_flagged_at_final_rung(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    _kill_uniform_a(s)
    for _ in range(10):
        _step(s, 1000, lambda job: 0.5)

    assert s.done()
    assert s.anchorless_groups() == [(TASK_A, ())]


def test_anchorless_not_flagged_before_final_rung(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    _kill_uniform_a(s)

    s.promotions()

    assert s.anchorless_groups() == []


def test_no_anchorless_groups_when_anchors_succeed(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    for _ in range(10):
        _step(s, 1000, lambda job: 0.5)

    assert s.done()
    assert s.anchorless_groups() == []


def test_batch_counts_in_flight_jobs_as_observations(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    n_rung1 = sum(1 for st in s.states() if st.rung == 1)
    rung1 = {UNIFORM_A.label(): 0.50, POINT_A.label(): 0.48, V00_A.label(): 0.10}
    _step(s, n_rung1, _by_label(rung1))  # one seed each (dense arms score 0.0)
    # Task A spread = sample std of [0.50, 0.48, 0.10] = 0.22554; bonus 2 * 0.22554 / sqrt(n).
    # Pick 1: uniform 0.951 > point 0.931 > V00_A 0.551 (dense arms 0.0).
    # Pick 2: uniform now has 1 in flight, n = 2: 0.5 + 0.319 = 0.819 < point 0.931.
    # Pick 3: point n = 2: 0.48 + 0.319 = 0.799 < uniform 0.819 > V00_A 0.551.

    jobs = s.next_jobs(3)

    assert [(j.arm, j.seed) for j in jobs] == [
        (UNIFORM_A, 1),
        (POINT_A, 1),
        (UNIFORM_A, 2),
    ]


# --- hardening: sd floor, liveness, torn log, directory fsync -------------------------------


def test_identical_seeds_do_not_prune_close_competitor(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    jobs = {(j.arm, j.seed): j for j in s.next_jobs(1000)}
    # uniform [0.5] x3: sd 0 floored to 0.05 -> LCB 0.5 - 2 * 0.05 / sqrt(3) = 0.4423.
    for seed in range(3):
        s.report(_result(jobs[(UNIFORM_A, seed)], 0.5))
    # V00_A [0.46, 0.47]: UCB 0.465 + 2 * 0.05 / sqrt(2) = 0.536 >= 0.4423 (unfloored 0.475 < 0.5).
    for seed, auc in enumerate((0.46, 0.47)):
        s.report(_result(jobs[(V00_A, seed)], auc))

    assert s.state(V00_A).status == "promoted"


def test_identical_seeds_arm_not_pruned_by_close_competitor(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    jobs = {(j.arm, j.seed): j for j in s.next_jobs(1000)}
    # uniform [0.52, 0.53, 0.54]: sd 0.01 floored -> LCB 0.53 - 0.0577 = 0.4723.
    for seed, auc in enumerate((0.52, 0.53, 0.54)):
        s.report(_result(jobs[(UNIFORM_A, seed)], auc))
    # V00_A [0.5, 0.5]: UCB 0.5 + 0.0707 = 0.571 >= 0.4723 (unfloored: 0.5 < 0.5185).
    for seed in range(2):
        s.report(_result(jobs[(V00_A, seed)], 0.5))

    assert s.state(V00_A).status == "promoted"


def test_sd_floor_zero_restores_zero_width_intervals(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A, sd_floor=0.0)
    _step(s, 100, _by_label({V00_A.label(): 0.9, V02_A.label(): 0.2}))
    s.promotions()
    jobs = {(j.arm, j.seed): j for j in s.next_jobs(1000)}
    for seed in range(3):
        s.report(_result(jobs[(UNIFORM_A, seed)], 0.5))
    for seed, auc in enumerate((0.46, 0.47)):
        s.report(_result(jobs[(V00_A, seed)], auc))

    assert s.state(V00_A).status == "pruned"


def test_ucb_bounds_apply_sd_floor() -> None:
    lcb, ucb = ucb_bounds([0.5, 0.5], spread_sd=0.0, beta=2.0, sd_floor=0.05)

    assert (lcb, ucb) == pytest.approx(
        (0.5 - 0.1 / math.sqrt(2), 0.5 + 0.1 / math.sqrt(2))
    )


def test_constructor_rejects_negative_sd_floor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sd_floor"):
        _new(tmp_path, SMALL_A, sd_floor=-0.1)


def test_is_live_false_once_arm_pruned_while_running(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    jobs = {(j.arm, j.seed): j for j in s.next_jobs(1000)}
    for seed, auc in enumerate((0.80, 0.82, 0.81)):
        s.report(_result(jobs[(UNIFORM_A, seed)], auc))
    for seed, auc in enumerate((0.10, 0.11)):
        s.report(_result(jobs[(V00_A, seed)], auc))

    assert s.is_live(jobs[(V00_A, 2)]) is False
    assert s.is_live(jobs[(POINT_A, 0)]) is True


def test_is_live_false_once_arm_failed_while_running(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    jobs = {(j.arm, j.seed): j for j in s.next_jobs(1000)}
    s.report(_result(jobs[(V00_A, 0)], 0.0, status="failed"))
    (retry,) = [j for j in s.next_jobs(1000) if j.arm == V00_A]
    s.report(_result(retry, 0.0, status="failed"))

    assert s.is_live(jobs[(V00_A, 1)]) is False


def test_is_live_false_for_job_from_an_earlier_rung(tmp_path: Path) -> None:
    # A stale job held by an old worker: its arm has since been promoted.
    s = _finish_rung0_small(tmp_path)
    old = Job(arm=V00_A, seed=0, rung=0, step_budget=10)

    assert s.is_live(old) is False
    assert s.is_live(Job(arm=V00_A, seed=0, rung=1, step_budget=30)) is True


def test_is_live_rejects_unknown_arm(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    stranger = _arm(TASK_B, "uniform/-")

    with pytest.raises(ValueError, match="unknown arm"):
        s.is_live(Job(arm=stranger, seed=0, rung=0, step_budget=10))


def test_load_truncates_torn_last_result_and_resubmits_its_job(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    jobs = s.next_jobs(2)
    s.report(_result(jobs[0], 0.4))
    log = tmp_path / "results.jsonl"
    good = log.read_text()
    whole = json.dumps({"arm_id": _arm_id(jobs[1].arm, 10), "seed": 0, "status": "ok"})
    log.write_text(good + whole[: len(whole) // 2])  # crash mid-append

    loaded = Scheduler.load(tmp_path, arm_id_of=_arm_id)

    assert log.read_text() == good
    assert loaded.next_jobs(1) == [jobs[1]]


def test_load_rejects_bad_line_before_the_last(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    jobs = s.next_jobs(2)
    s.report(_result(jobs[0], 0.4))
    s.report(_result(jobs[1], 0.4))
    log = tmp_path / "results.jsonl"
    first, second = log.read_text().splitlines()
    log.write_text(first[:10] + "\n" + second + "\n")

    with pytest.raises(ValueError, match=r"results\.jsonl line 1:"):
        Scheduler.load(tmp_path, arm_id_of=_arm_id)


def test_load_rejects_torn_line_the_state_already_applied(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    job = s.next_jobs(1)[0]
    s.report(_result(job, 0.4))
    log = tmp_path / "results.jsonl"
    log.write_text(log.read_text()[:20])

    with pytest.raises(ValueError, match=r"results\.jsonl line 1:"):
        Scheduler.load(tmp_path, arm_id_of=_arm_id)


def test_save_fsyncs_the_state_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _new(tmp_path, SMALL_A)
    synced_dirs: list[bool] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        synced_dirs.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)

    s.save()

    assert synced_dirs == [False, True]


# --- reporting ---------------------------------------------------------------------------------


def test_report_rejects_result_without_running_job(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    job = Job(arm=V00_A, seed=0, rung=0, step_budget=10)

    with pytest.raises(ValueError, match="no running job"):
        s.report(_result(job, 0.5))


def test_report_ignores_exact_duplicate_result(tmp_path: Path) -> None:
    # A late result from a worker that outlived a restart.
    s = _new(tmp_path, SMALL_A)
    job = s.next_jobs(1)[0]
    s.report(_result(job, 0.5))

    s.report(_result(job, 0.5))

    assert s.state(job.arm).results == [_result(job, 0.5)]
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 1


def test_report_ignores_exact_duplicate_after_load(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    job = s.next_jobs(1)[0]
    s.report(_result(job, 0.5))
    loaded = Scheduler.load(tmp_path, arm_id_of=_arm_id)

    loaded.report(_result(job, 0.5))

    assert loaded.state(job.arm).results == [_result(job, 0.5)]


def test_report_rejects_conflicting_duplicate_result(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    job = s.next_jobs(1)[0]
    s.report(_result(job, 0.5))

    with pytest.raises(ConflictingResultError, match="differs from the recorded"):
        s.report(_result(job, 0.6))


def test_results_jsonl_appends_one_line_per_report(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    jobs = s.next_jobs(3)
    for job in jobs:
        s.report(_result(job, 0.25))

    lines = (tmp_path / "results.jsonl").read_text().splitlines()

    assert [(json.loads(x)["seed"], json.loads(x)["auc"]) for x in lines] == [
        (0, 0.25)
    ] * 3


# --- state and resume --------------------------------------------------------------------------


def test_save_load_round_trip_resubmits_running_jobs(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    jobs = s.next_jobs(3)
    s.report(_result(jobs[1], 0.4))

    loaded = Scheduler.load(tmp_path, arm_id_of=_arm_id)
    again = loaded.next_jobs(2)

    assert again == [jobs[0], jobs[2]]
    assert loaded.state(jobs[1].arm).results == [_result(jobs[1], 0.4)]


def test_load_resubmits_in_flight_jobs_before_new_ones(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    rung1 = s.next_jobs(1000)
    dense = [j for j in rung1 if j.arm.task != TASK_A]
    for job in [j for j in rung1 if j != dense[0]]:
        s.report(_result(job, 0.5))
    s.promotions()  # task A moves to rung 2; the dense tasks wait for dense[0]
    (in_flight,) = s.next_jobs(1)
    s.report(_result(dense[0], 0.0, status="failed"))  # a rung-1 retry is now pending

    loaded = Scheduler.load(tmp_path, arm_id_of=_arm_id)

    assert in_flight.rung == 2
    assert loaded.next_jobs(1) == [in_flight]


def test_resubmits_beyond_k_keep_priority(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    rung1 = s.next_jobs(1000)
    dense = [j for j in rung1 if j.arm.task != TASK_A]
    for job in [j for j in rung1 if j != dense[0]]:
        s.report(_result(job, 0.5))
    s.promotions()  # task A moves to rung 2; the dense tasks wait for dense[0]
    in_flight = s.next_jobs(2)
    s.report(_result(dense[0], 0.0, status="failed"))  # a rung-1 retry is now pending
    loaded = Scheduler.load(tmp_path, arm_id_of=_arm_id)

    first, second = loaded.next_jobs(1), loaded.next_jobs(1)

    assert first + second == in_flight


def test_load_does_not_resubmit_finished_jobs(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    jobs = s.next_jobs(3)
    s.report(_result(jobs[1], 0.4))

    loaded = Scheduler.load(tmp_path, arm_id_of=_arm_id)
    issued = loaded.next_jobs(100)

    assert [j.arm for j in issued] == [SMALL_A[0], SMALL_A[2], SMALL_A[3], SMALL_A[4]]


def test_load_restores_states(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    _step(s, 4, lambda job: 0.3)

    loaded = Scheduler.load(tmp_path, arm_id_of=_arm_id)

    assert loaded.states() == s.states()


def test_load_applies_result_logged_before_state_save(tmp_path: Path) -> None:
    # A crash between the results.jsonl append and the scheduler.json replace.
    s = _new(tmp_path, SMALL_A)
    jobs = s.next_jobs(2)
    before = (tmp_path / "scheduler.json").read_bytes()
    s.report(_result(jobs[0], 0.7))
    (tmp_path / "scheduler.json").write_bytes(before)

    loaded = Scheduler.load(tmp_path, arm_id_of=_arm_id)
    again = loaded.next_jobs(100)

    assert loaded.state(jobs[0].arm).results == [_result(jobs[0], 0.7)]
    assert jobs[0] not in again
    assert again[0] == jobs[1]


def test_load_rejects_changed_arm_ids(tmp_path: Path) -> None:
    _new(tmp_path, SMALL_A).next_jobs(1)

    with pytest.raises(ValueError, match="arm_id"):
        Scheduler.load(tmp_path, arm_id_of=lambda arm, budget: _arm_id(arm, budget + 1))


def test_new_scheduler_refuses_existing_state(tmp_path: Path) -> None:
    _new(tmp_path, SMALL_A).next_jobs(1)

    with pytest.raises(FileExistsError, match="scheduler.json"):
        _new(tmp_path, SMALL_A)


def test_state_file_written_atomically(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)
    s.next_jobs(2)

    names = sorted(p.name for p in tmp_path.iterdir())

    assert names == ["scheduler.json"]


# --- construction ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"rung_budgets": (10, 10, 90)}, "strictly increasing"),
        ({"rung_budgets": (0, 30, 90)}, "positive"),
        ({"seeds_per_rung": (1, 3)}, "one seed count per rung"),
        ({"seeds_per_rung": (1, 0, 5)}, "at least 1"),
        ({"eta": 1}, "eta"),
        ({"beta": -1.0}, "beta"),
        ({"beta": float("nan")}, "beta"),
    ],
)
def test_constructor_rejects_bad_config(
    tmp_path: Path, kw: dict[str, object], match: str
) -> None:
    base: dict[str, object] = {
        "arms": SMALL_A,
        "rung_budgets": BUDGETS,
        "state_dir": tmp_path,
        "arm_id_of": _arm_id,
    }

    with pytest.raises(ValueError, match=match):
        Scheduler(**{**base, **kw})  # type: ignore[arg-type]


def test_constructor_rejects_duplicate_arms(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _new(tmp_path, [V00_A, V00_A])


def test_constructor_rejects_non_rung0_arm(tmp_path: Path) -> None:
    dense = arm_space("rung1_expand", survivors=[V00_A])[0]

    with pytest.raises(ValueError, match="rung-0 arm"):
        _new(tmp_path, [dense])


def test_constructor_rejects_colliding_arm_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="arm_id"):
        Scheduler(
            arms=SMALL_A,
            rung_budgets=BUDGETS,
            state_dir=tmp_path,
            arm_id_of=lambda arm, budget: "same",
        )


def test_next_jobs_rejects_negative_k(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="k must be"):
        _new(tmp_path, SMALL_A).next_jobs(-1)


# --- arm id mapping ----------------------------------------------------------------------------

_DEFAULTS = RunDefaults(
    num_envs=4096, rollout_len=24, ppo=START_CONFIG, eval_every_steps=1000, device="cpu"
)
_ID_SCRIPT = """
from resetsweep.ppo.update import START_CONFIG
from resetsweep.sweep.arms import arm_space
from resetsweep.sweep.scheduler import RunDefaults, spec_for
d = RunDefaults(num_envs=4096, rollout_len=24, ppo=START_CONFIG, eval_every_steps=1000,
                device="cuda:0")
arm = next(a for a in arm_space("rung0") if a.label() == "maze/ball[p_task=0.5]/rnd")
print(spec_for(arm, seed=7, step_budget=30, defaults=d).arm_id())
"""


def test_spec_for_arm_id_is_stable_across_processes() -> None:
    arm = _arm("maze", "ball[p_task=0.5]/rnd")
    here = spec_for(arm, seed=0, step_budget=30, defaults=_DEFAULTS).arm_id()
    env = {**os.environ, "PYTHONHASHSEED": "12345"}

    out = subprocess.run(
        [sys.executable, "-c", _ID_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    assert out.stdout.strip() == here


def test_spec_for_copies_arm_fields() -> None:
    arm = _arm("maze", "ball[p_task=0.5]/rnd")

    spec = spec_for(arm, seed=3, step_budget=30, defaults=_DEFAULTS)

    assert (
        spec.task,
        dict(spec.method_kw),
        spec.score,
        spec.seed,
        spec.step_budget,
    ) == (
        "maze",
        {"p_task": 0.5},
        "rnd",
        3,
        30,
    )


def test_spec_for_arm_id_depends_on_step_budget() -> None:
    arm = _arm("maze", "ball[p_task=0.5]/rnd")

    ids = {
        spec_for(arm, seed=0, step_budget=b, defaults=_DEFAULTS).arm_id()
        for b in BUDGETS
    }

    assert len(ids) == 3


# --- property ----------------------------------------------------------------------------------


@settings(max_examples=20, deadline=None)
@given(data=st.data())
def test_next_jobs_never_returns_pruned_or_failed_arm(data: st.DataObject) -> None:
    with tempfile.TemporaryDirectory() as d:
        s = _new(Path(d), _arms(TASK_A))
        running: list[Job] = []
        issued: set[Job] = set()
        for _ in range(40):
            for job in s.next_jobs(data.draw(st.integers(0, 12), label="k")):
                assert s.state(job.arm).status in LIVE
                assert job not in issued
                issued.add(job)
                running.append(job)
            n_report = data.draw(st.integers(0, len(running)), label="n_report")
            for job in running[:n_report]:
                auc = data.draw(st.floats(0.0, 1.0), label="auc")
                failed = data.draw(st.booleans(), label="failed")
                s.report(_result(job, auc, status="failed" if failed else "ok"))
            running = running[n_report:]
        shutil.rmtree(d, ignore_errors=True)


# --- simulation: many groups, random failures, random restarts ---------------------------------

SIM_TASKS = ("maze", TASK_A, TASK_B)  # maze adds a task-variant group at rung 1
SIM_MAX_STEPS = 5000


def _sim_arms() -> list[Arm]:
    out: list[Arm] = []
    for task in SIM_TASKS:
        arms = [a for a in arm_space("rung0") if a.task == task]
        out += [a for a in arms if a.method in ("uniform", "point")]
        out += [a for a in arms if a.method not in ("uniform", "point")][:4]
    return out


def _sim_result(seed: int, job: Job, fail_p: float) -> RunResult:
    # Deterministic per (run, arm, rung, seed): a restart that reissues a job sees the same draw.
    noise = random.Random(f"{seed}|{job.arm.label()}|{job.rung}|{job.seed}")
    mean = random.Random(f"{seed}|{job.arm.label()}").random()
    auc = min(1.0, max(0.0, mean + noise.gauss(0.0, 0.1)))
    return _result(job, auc, status="failed" if noise.random() < fail_p else "ok")


def _simulate(
    tmp_path: Path, seed: int, *, reload_p: float, fail_p: float
) -> tuple[Scheduler, Counter[Job]]:
    rng = random.Random(seed)
    s = _new(tmp_path, _sim_arms())
    running: list[Job] = []
    issued: Counter[Job] = Counter()
    for _ in range(SIM_MAX_STEPS):
        if s.done():
            return s, issued
        jobs = s.next_jobs(7)
        issued.update(jobs)
        running += jobs
        n = rng.randint(1, len(running)) if running else 0
        for job in running[:n]:
            s.report(_sim_result(seed, job, fail_p))
        running = running[n:]
        if running and rng.random() < reload_p:
            s = Scheduler.load(tmp_path, arm_id_of=_arm_id)
            issued.subtract(
                running
            )  # lost with the old workers; resubmits reissue them
            running = []
    raise AssertionError(f"not done after {SIM_MAX_STEPS} steps (seed {seed})")


@pytest.mark.parametrize("seed", range(3))
def test_simulated_sweep_terminates_without_duplicates_and_with_full_final_seeds(
    tmp_path: Path, seed: int
) -> None:
    s, issued = _simulate(tmp_path, seed, reload_p=0.15, fail_p=0.1)

    final_seed_counts = {
        len(_rung_aucs(s, st.arm, 2)) for st in s.states() if st.status == "final"
    }
    stuck = [st.arm.label() for st in s.states() if st.status in ("active", "promoted")]

    assert max(issued.values()) == 1
    assert final_seed_counts == {5}
    assert stuck == []
    groups = {group_of(st.arm) for st in s.states() if st.status == "final"}
    assert len(groups) == 7
    assert ("maze", (("lanes", 5),)) in groups


def test_rung_results_returns_only_that_rungs_results(tmp_path: Path) -> None:
    s = _finish_rung0_small(tmp_path)
    jobs = [j for j in s.next_jobs(1000) if j.arm == V00_A]
    for job in jobs:
        s.report(_result(job, 0.7))

    rung0 = s.rung_results(V00_A, 0)
    rung1 = s.rung_results(V00_A, 1)

    assert [(r.auc, r.seed) for r in rung0] == [(0.9, 0)]
    assert sorted((r.auc, r.seed) for r in rung1) == [(0.7, 0), (0.7, 1), (0.7, 2)]
    assert s.rung_results(V00_A, 2) == []
    assert s.rung_budgets == BUDGETS


def test_rung_results_rejects_unknown_rung(tmp_path: Path) -> None:
    s = _new(tmp_path, SMALL_A)

    with pytest.raises(ValueError, match="rung 3 out of range"):
        s.rung_results(UNIFORM_A, 3)

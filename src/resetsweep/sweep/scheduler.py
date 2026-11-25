"""UCB successive-rung scheduler with resume (spec section 7, Task 22 controller rulings).

The scheduler is a state machine over arms. Each arm has a current rung and a status:

    active    live; entered its rung without a promotion (rung-0 arms, rung-1 expansions)
    promoted  live; promoted into its current rung (below the final rung)
    final     live; on the final rung
    pruned    stopped: UCB fell below the task leader's LCB, or not promoted out of its rung
    failed    stopped: one seed failed twice (seed, then seed + RETRY_SEED_OFFSET)

Decisions (all binding rulings unless marked "ours"):

- Arm ids. `RunResult.arm_id` is `RunSpec.arm_id()`, which hashes `step_budget`, so one arm
  has a different id on every rung. The scheduler takes `arm_id_of(arm, step_budget) -> str`
  and keeps the map arm_id -> (arm, rung). Task 23 passes
  ``lambda arm, b: spec_for(arm, seed=0, step_budget=b, defaults=d).arm_id()`` (seed is not
  hashed). The ids are stored in `scheduler.json`; `load` recomputes them and refuses to
  resume if any differ (a changed RunSpec default would orphan every result).
- Seeds. Rung r runs seeds ``0 .. seeds_per_rung[r] - 1``. A failed seed is retried once as
  ``seed + RETRY_SEED_OFFSET``; a second failure marks the arm `failed`, excluded from ranking,
  promotion and job selection. Results at a rung count only toward that rung.
- Order. Pending rung-0 jobs are always issued first, in arm order, so rung 0 runs every arm
  once before any UCB choice. After that, live arms on rung >= 1 are sorted by
  (rung ascending, UCB descending, arm order) and given one job per pass, round-robin
  (ours: lower rungs first, because they gate promotions; UCB is not comparable across rungs).
- UCB at rung >= 1: mean = IQM of ok AUCs at the arm's current rung; sd = sample std
  (ddof 1) for n >= 2, else the task's spread at the rung (sample std of the per-arm means of
  that task's non-failed arms at that rung; 1.0, the AUC range, if fewer than 2 have results).
  An arm with no result yet (n = 0, e.g. a rung-1 expansion) has UCB +inf and LCB -inf: it is
  explored first and can be neither pruned nor the leader until it has a result.
- Rung 0: the "mean" is the `rung0_key` order rank-transformed to [0, 1] within the task (ties
  share the average rank). Every arm has one seed there, so the spread term is the same for
  all arms and UCB order equals rank order; rung 0 only uses it to rank promotions, and
  pruning starts at rung 1.
- Promotion out of rung r for a task happens once the rung is finished for that task (no job
  running at that rung, no live arm with a pending seed). Out of rung >= 1 it also waits until
  every task with rung-0 arms has promoted out of rung 0, because rung-1 expansions keep adding
  dense-task arms until then. The top ``ceil(n / eta)`` non-protected arms are promoted, where
  n counts the task's non-protected, non-failed arms on the rung (pruned ones included); rank
  at rung 0, UCB above. The rest are pruned.
- Protected arms (`uniform`, `point`) are never pruned and are promoted to every rung outside
  the quota, so the normalisation anchor reaches the final rung with the same seed count.
  They can still fail.
- rung1_expand: promotion out of rung 0 calls ``arm_space("rung1_expand", survivors=winners)``
  (the quota winners; protected arms are not survivors) and adds every arm the scheduler does
  not already have at rung 1 with status `active`.
- Tasks are grouped by `arm.task` (the name): task variants (`task_kw`) compete in their base
  task's group (ours).
- Done: nothing running, and either every task has a final-rung arm whose LCB beats the UCB of
  every other live final-rung arm, or no job and no promotion remains.
- State: `results.jsonl` is append-only, one RunResult per line (NaN written as null), and is
  the only copy of the results. `scheduler.json` holds everything else (config, arms, rungs,
  statuses, arm ids, jobs in flight, promotions done, and how many results.jsonl lines it
  reflects); it is rewritten atomically (temp file, fsync, `os.replace`) after every change,
  so its size does not grow with the results. A result is appended before the state is
  saved; `load` applies any logged result the snapshot missed (a crash between the two
  writes), then hands the in-flight jobs back first from `next_jobs`.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from resetsweep.ppo.update import PPOConfig
from resetsweep.sweep.arms import DENSE_TASKS, SPARSE_TASKS, Arm, Kw, arm_space
from resetsweep.sweep.stats import iqm, rung0_key
from resetsweep.train import RunResult, RunSpec

Status = Literal["active", "promoted", "pruned", "final", "failed"]
ArmIdOf = Callable[[Arm, int], str]

LIVE_STATUSES: Final = frozenset({"active", "promoted", "final"})
PROTECTED_METHODS: Final = frozenset({"uniform", "point"})
RETRY_SEED_OFFSET: Final = 1000
STATE_FILE: Final = "scheduler.json"
RESULTS_FILE: Final = "results.jsonl"
_STATE_VERSION: Final = 1
_NO_RESULTS_SD: Final = 1.0  # the AUC range
_FLOAT_RESULT_FIELDS: Final = ("auc", "final_success", "ever_succeeded_share", "wall_s")


@dataclass(frozen=True)
class Job:
    arm: Arm
    seed: int
    rung: int
    step_budget: int


@dataclass
class ArmState:
    arm: Arm
    rung: int
    results: list[RunResult]
    status: Status


@dataclass(frozen=True, kw_only=True)
class RunDefaults:
    """The RunSpec fields that are the same for every job of a sweep."""

    num_envs: int
    rollout_len: int
    ppo: PPOConfig
    eval_every_steps: int
    eval_n: int = 512
    eval_seed: int = 12345
    device: str


def spec_for(
    arm: Arm, *, seed: int, step_budget: int, defaults: RunDefaults
) -> RunSpec:
    """The RunSpec of one job. Its `arm_id()` hashes everything except seed and device."""
    return RunSpec(
        task=arm.task,
        task_kw=arm.task_kwargs(),
        method=arm.method,
        method_kw=arm.method_kwargs(),
        score=arm.score,
        score_kw=arm.score_kwargs(),
        seed=seed,
        step_budget=step_budget,
        num_envs=defaults.num_envs,
        rollout_len=defaults.rollout_len,
        ppo=defaults.ppo,
        eval_every_steps=defaults.eval_every_steps,
        eval_n=defaults.eval_n,
        eval_seed=defaults.eval_seed,
        device=defaults.device,
    )


def ucb_bounds(
    values: Sequence[float], *, spread_sd: float, beta: float
) -> tuple[float, float]:
    """(LCB, UCB) = IQM -/+ beta * sd / sqrt(n); sd is the sample std for n >= 2, else
    `spread_sd`. No values gives (-inf, inf)."""
    if not (math.isfinite(spread_sd) and spread_sd >= 0.0):
        raise ValueError(f"spread_sd must be finite and >= 0, got {spread_sd}")
    if not (math.isfinite(beta) and beta >= 0.0):
        raise ValueError(f"beta must be finite and >= 0, got {beta}")
    n = len(values)
    if n == 0:
        return -math.inf, math.inf
    arr = np.asarray(values, dtype=float)
    mean = iqm(arr)  # raises on non-finite values
    sd = float(np.std(arr, ddof=1)) if n >= 2 else spread_sd
    half = beta * sd / math.sqrt(n)
    return mean - half, mean + half


def _check_config(
    rung_budgets: Sequence[int], seeds_per_rung: Sequence[int], eta: int, beta: float
) -> None:
    if len(rung_budgets) < 2:
        raise ValueError(f"need at least 2 rungs, got rung_budgets={rung_budgets}")
    if len(seeds_per_rung) != len(rung_budgets):
        raise ValueError(
            f"need one seed count per rung: seeds_per_rung={seeds_per_rung}, "
            f"rung_budgets={rung_budgets}"
        )
    if any(b <= 0 for b in rung_budgets):
        raise ValueError(f"rung budgets must be positive, got {rung_budgets}")
    if any(a >= b for a, b in itertools.pairwise(rung_budgets)):
        raise ValueError(
            f"rung budgets must be strictly increasing, got {rung_budgets}"
        )
    if any(not 1 <= s < RETRY_SEED_OFFSET for s in seeds_per_rung):
        raise ValueError(
            f"seeds per rung must be at least 1 and below {RETRY_SEED_OFFSET}, "
            f"got {seeds_per_rung}"
        )
    if eta < 2:
        raise ValueError(f"eta must be >= 2, got {eta}")
    if not (math.isfinite(beta) and beta >= 0.0):
        raise ValueError(f"beta must be finite and >= 0, got {beta}")


class Scheduler:
    """See the module docstring for the rules. Writes `state_dir/scheduler.json` and
    `state_dir/results.jsonl`; refuses a `state_dir` that already holds a sweep (use `load`)."""

    def __init__(
        self,
        *,
        arms: list[Arm],
        rung_budgets: tuple[int, int, int],
        seeds_per_rung: tuple[int, int, int] = (1, 3, 5),
        eta: int = 3,
        beta: float = 2.0,
        state_dir: Path,
        arm_id_of: ArmIdOf,
    ) -> None:
        self._configure(rung_budgets, seeds_per_rung, eta, beta, state_dir, arm_id_of)
        if not arms:
            raise ValueError("arms is empty")
        if len(set(arms)) != len(arms):
            raise ValueError("duplicate arms")
        for arm in arms:
            if arm.task not in SPARSE_TASKS or arm.task_kw:
                raise ValueError(
                    f"{arm.label()} is not a rung-0 arm (sparse task, no task_kw)"
                )
        for name in (STATE_FILE, RESULTS_FILE):
            if (state_dir / name).exists():
                raise FileExistsError(
                    f"{state_dir / name} exists; resume with Scheduler.load({state_dir!s})"
                )
        state_dir.mkdir(parents=True, exist_ok=True)
        self._rung0_tasks = tuple(dict.fromkeys(a.task for a in arms))
        for arm in arms:
            self._add_arm(arm, rung=0, status="active")
        self.save()

    def _configure(
        self,
        rung_budgets: Sequence[int],
        seeds_per_rung: Sequence[int],
        eta: int,
        beta: float,
        state_dir: Path,
        arm_id_of: ArmIdOf,
    ) -> None:
        _check_config(rung_budgets, seeds_per_rung, eta, beta)
        self._budgets = tuple(rung_budgets)
        self._seeds = tuple(seeds_per_rung)
        self._eta = eta
        self._beta = beta
        self._dir = state_dir
        self._arm_id_of = arm_id_of
        self._last_rung = len(self._budgets) - 1
        self._rung0_tasks: tuple[str, ...] = ()
        self._states: list[ArmState] = []
        self._index: dict[Arm, int] = {}
        self._ids: dict[str, tuple[int, int]] = {}  # arm_id -> (arm index, rung)
        self._id_at: dict[tuple[int, int], str] = {}
        self._running: dict[tuple[str, int], Job] = {}  # (arm_id, seed) -> job
        self._resubmit: list[Job] = []
        self._promoted_out: set[tuple[str, int]] = set()
        self._n_results = 0  # lines of results.jsonl reflected in this state

    # --- public interface ----------------------------------------------------------------------

    def next_jobs(self, k: int) -> list[Job]:
        """Up to `k` new jobs: resubmitted in-flight jobs first, then rung 0 in arm order, then
        UCB round-robin. Never a job already running or done, past the final rung's seed count,
        or for an arm that is not live. Runs `promotions()` first."""
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.promotions()
        jobs: list[Job] = []
        resubmit, self._resubmit = self._resubmit, []
        for job in resubmit:
            i = self._index[job.arm]
            ok = self._is_live(i) and self._states[i].rung == job.rung
            if not (ok and job.seed in self._pending_seeds(i)):
                continue  # pruned, failed or promoted since: drop it
            if len(jobs) < k:
                jobs.append(self._issue(i, job.seed))
            else:
                self._resubmit.append(job)  # keeps its priority for the next call
        rung0 = [i for i in self._live() if self._states[i].rung == 0]
        for i in rung0:
            for seed in self._pending_seeds(i)[: k - len(jobs)]:
                jobs.append(self._issue(i, seed))
        if len(jobs) < k:
            jobs += self._ucb_round_robin(k - len(jobs))
        if jobs:
            self.save()
        assert len(jobs) <= k, (len(jobs), k)
        return jobs

    def report(self, result: RunResult) -> None:
        """Record the result of a running job, append it to results.jsonl, save state."""
        key = (result.arm_id, result.seed)
        job = self._running.pop(key, None)
        if job is None:
            raise ValueError(f"result {key} matches no running job")
        if result.status == "ok" and not math.isfinite(result.auc):
            self._running[key] = job
            raise ValueError(f"ok result {key} has non-finite auc {result.auc}")
        self._append_result(result)
        self._apply(job, result)
        self.save()

    def promotions(self) -> None:
        """Promote every task whose current rung has finished; idempotent."""
        max_passes = (len(SPARSE_TASKS) + len(DENSE_TASKS)) * len(self._budgets) + 1
        changed = False
        for _ in range(max_passes):
            ready = self._ready_promotions()
            if not ready:
                break
            for task, rung in ready:
                self._promote(task, rung)
            changed = True
        else:
            raise RuntimeError(f"promotions did not settle in {max_passes} passes")
        if changed:
            self.save()

    def done(self) -> bool:
        if self._running:
            return False
        if self._all_separated():
            return True
        return not self._ready_promotions() and not any(
            self._pending_seeds(i) for i in self._live()
        )

    def state(self, arm: Arm) -> ArmState:
        st = self._states[self._index[arm]]
        return replace(st, results=list(st.results))

    def states(self) -> tuple[ArmState, ...]:
        return tuple(replace(st, results=list(st.results)) for st in self._states)

    # --- arms, ids, slots ----------------------------------------------------------------------

    def _add_arm(self, arm: Arm, *, rung: int, status: Status) -> None:
        assert arm not in self._index, arm.label()
        self._index[arm] = len(self._states)
        self._states.append(ArmState(arm=arm, rung=rung, results=[], status=status))
        self._register_id(self._index[arm], rung)

    def _register_id(self, i: int, rung: int) -> None:
        arm_id = self._arm_id_of(self._states[i].arm, self._budgets[rung])
        other = self._ids.get(arm_id)
        if other is not None and other != (i, rung):
            a, b = self._states[other[0]].arm.label(), self._states[i].arm.label()
            raise ValueError(
                f"arm_id {arm_id} maps both {a}@rung{other[1]} and {b}@rung{rung}"
            )
        self._ids[arm_id] = (i, rung)
        self._id_at[(i, rung)] = arm_id

    def _is_live(self, i: int) -> bool:
        return self._states[i].status in LIVE_STATUSES

    def _live(self) -> list[int]:
        return [i for i in range(len(self._states)) if self._is_live(i)]

    def _rung_results(self, i: int, rung: int) -> list[RunResult]:
        arm_id = self._id_at.get((i, rung))
        return [r for r in self._states[i].results if r.arm_id == arm_id]

    def _ok_aucs(self, i: int) -> list[float]:
        st = self._states[i]
        return [r.auc for r in self._rung_results(i, st.rung) if r.status == "ok"]

    def _pending_seeds(self, i: int) -> list[int]:
        """Seeds of arm `i` at its current rung that may be issued now (retries included)."""
        st = self._states[i]
        arm_id = self._id_at[(i, st.rung)]
        got = {r.seed: r.status for r in self._rung_results(i, st.rung)}
        running = {seed for (aid, seed) in self._running if aid == arm_id}
        out: list[int] = []
        for seed in range(self._seeds[st.rung]):
            retry = seed + RETRY_SEED_OFFSET
            if seed in running or retry in running:
                continue
            first, second = got.get(seed), got.get(retry)
            if first is None:
                out.append(seed)
            elif first == "failed" and second is None:
                out.append(retry)
        return out

    def _issue(self, i: int, seed: int) -> Job:
        st = self._states[i]
        assert self._is_live(i), st.arm.label()
        job = Job(
            arm=st.arm, seed=seed, rung=st.rung, step_budget=self._budgets[st.rung]
        )
        key = (self._id_at[(i, st.rung)], seed)
        assert key not in self._running, key
        self._running[key] = job
        return job

    # --- UCB -----------------------------------------------------------------------------------

    def _spread(self, task: str, rung: int) -> float:
        means = [
            iqm(np.asarray(aucs))
            for i, st in enumerate(self._states)
            if st.arm.task == task and st.rung == rung and st.status != "failed"
            if (aucs := self._ok_aucs(i))
        ]
        return float(np.std(means, ddof=1)) if len(means) >= 2 else _NO_RESULTS_SD

    def _bounds(self, indices: Iterable[int]) -> dict[int, tuple[float, float]]:
        spreads: dict[tuple[str, int], float] = {}
        out: dict[int, tuple[float, float]] = {}
        for i in indices:
            st = self._states[i]
            assert st.rung >= 1, "UCB is not used at rung 0"
            key = (st.arm.task, st.rung)
            if key not in spreads:
                spreads[key] = self._spread(*key)
            out[i] = ucb_bounds(
                self._ok_aucs(i), spread_sd=spreads[key], beta=self._beta
            )
        return out

    def _ucb_round_robin(self, k: int) -> list[Job]:
        pending = {
            i: s
            for i in self._live()
            if self._states[i].rung >= 1
            if (s := self._pending_seeds(i))
        }
        bounds = self._bounds(pending)
        order = sorted(pending, key=lambda i: (self._states[i].rung, -bounds[i][1], i))
        jobs: list[Job] = []
        for _ in range(2 * max(self._seeds)):  # each pass issues one seed per arm
            issued = False
            for i in order:
                if len(jobs) < k and pending[i]:
                    jobs.append(self._issue(i, pending[i].pop(0)))
                    issued = True
            if not issued or len(jobs) >= k:
                break
        return jobs

    def _rung0_ranks(self, members: Sequence[int]) -> dict[int, float]:
        keys = {i: rung0_key(self._rung_results(i, 0)) for i in members}
        m = len(keys)
        ranks: dict[int, float] = {}
        for i, key in keys.items():
            below = sum(1 for other in keys.values() if other < key)
            equal = sum(1 for other in keys.values() if other == key)
            ranks[i] = (below + (equal - 1) / 2) / (m - 1) if m > 1 else 1.0
        assert all(0.0 <= r <= 1.0 for r in ranks.values()), ranks
        return ranks

    # --- results, pruning, promotion -----------------------------------------------------------

    def _apply(self, job: Job, result: RunResult) -> None:
        i = self._index[job.arm]
        st = self._states[i]
        st.results.append(result)
        if (
            result.status == "failed"
            and job.seed >= RETRY_SEED_OFFSET
            and self._is_live(i)
        ):
            st.status = "failed"
        if job.rung >= 1 and st.rung == job.rung:
            self._prune(st.arm.task, job.rung)

    def _prune(self, task: str, rung: int) -> None:
        live = [
            i
            for i in self._live()
            if self._states[i].arm.task == task and self._states[i].rung == rung
        ]
        bounds = self._bounds(live)
        leader_lcb = max((lcb for lcb, _ in bounds.values()), default=-math.inf)
        if not math.isfinite(leader_lcb):
            return
        for i in live:
            if (
                self._states[i].arm.method not in PROTECTED_METHODS
                and bounds[i][1] < leader_lcb
            ):
                self._states[i].status = "pruned"

    def _tasks(self) -> list[str]:
        return list(dict.fromkeys(st.arm.task for st in self._states))

    def _rung_finished(self, task: str, rung: int) -> bool:
        if any(j.arm.task == task and j.rung == rung for j in self._running.values()):
            return False
        return not any(
            self._pending_seeds(i)
            for i in self._live()
            if self._states[i].arm.task == task and self._states[i].rung == rung
        )

    def _ready_promotions(self) -> list[tuple[str, int]]:
        rung0_done = all((t, 0) in self._promoted_out for t in self._rung0_tasks)
        ready: list[tuple[str, int]] = []
        for task in self._tasks():
            for rung in range(self._last_rung):
                if (task, rung) in self._promoted_out:
                    continue
                if rung == 0 and task not in self._rung0_tasks:
                    continue
                if (rung == 0 or rung0_done) and self._rung_finished(task, rung):
                    ready.append((task, rung))
                break
        return ready

    def _promote(self, task: str, rung: int) -> None:
        members = [
            i
            for i, st in enumerate(self._states)
            if st.arm.task == task and st.rung == rung
        ]
        protected = [
            i
            for i in members
            if self._is_live(i) and self._states[i].arm.method in PROTECTED_METHODS
        ]
        contenders = [
            i
            for i in members
            if self._states[i].arm.method not in PROTECTED_METHODS
            and self._states[i].status != "failed"
        ]
        candidates = [i for i in contenders if self._is_live(i)]
        if rung == 0:
            score = self._rung0_ranks(candidates)
        else:
            score = {i: ucb for i, (_, ucb) in self._bounds(candidates).items()}
        quota = math.ceil(len(contenders) / self._eta)
        winners = sorted(candidates, key=lambda i: (-score[i], i))[:quota]
        for i in candidates:
            if i not in winners:
                self._states[i].status = "pruned"
        new_rung = rung + 1
        for i in sorted(protected + winners):
            st = self._states[i]
            st.rung = new_rung
            st.status = "final" if new_rung == self._last_rung else "promoted"
            self._register_id(i, new_rung)
        self._promoted_out.add((task, rung))
        if rung == 0:
            survivors = [self._states[i].arm for i in winners]
            entry: Status = "final" if self._last_rung == 1 else "active"
            for arm in arm_space("rung1_expand", survivors=survivors):
                if arm not in self._index:
                    self._add_arm(arm, rung=1, status=entry)

    def _all_separated(self) -> bool:
        tasks = self._tasks()
        for task in tasks:
            finals = [
                i
                for i in self._live()
                if self._states[i].arm.task == task
                and self._states[i].status == "final"
            ]
            if not finals:
                return False
            bounds = self._bounds(finals)
            leader = max(finals, key=lambda i: (bounds[i][0], -i))
            leader_lcb = bounds[leader][0]
            if not math.isfinite(leader_lcb):
                return False
            if any(bounds[i][1] >= leader_lcb for i in finals if i != leader):
                return False
        return bool(tasks)

    # --- persistence ---------------------------------------------------------------------------

    def _append_result(self, result: RunResult) -> None:
        with (self._dir / RESULTS_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(_result_to_json(result), allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._n_results += 1

    def _job_json(self, job: Job) -> dict[str, int]:
        return {"arm": self._index[job.arm], "seed": job.seed, "rung": job.rung}

    def save(self) -> None:
        """Write scheduler.json atomically (temp file, fsync, os.replace)."""
        in_flight = list(self._running.values()) + self._resubmit
        state = {
            "version": _STATE_VERSION,
            "config": {
                "rung_budgets": list(self._budgets),
                "seeds_per_rung": list(self._seeds),
                "eta": self._eta,
                "beta": self._beta,
            },
            "rung0_tasks": list(self._rung0_tasks),
            "arms": [
                {
                    "arm": _arm_to_json(st.arm),
                    "rung": st.rung,
                    "status": st.status,
                }
                for st in self._states
            ],
            "arm_ids": [[i, rung, arm_id] for (i, rung), arm_id in self._id_at.items()],
            "running": [self._job_json(j) for j in in_flight],
            "promoted_out": sorted([t, r] for t, r in self._promoted_out),
            "n_results": self._n_results,
        }
        tmp = self._dir / f"{STATE_FILE}.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            f.write(json.dumps(state, allow_nan=False))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._dir / STATE_FILE)

    @classmethod
    def load(cls, state_dir: Path, *, arm_id_of: ArmIdOf) -> Scheduler:
        """Resume from `state_dir`. Raises ValueError if `arm_id_of` gives different ids than
        the saved ones, or if results.jsonl holds a result the state cannot account for."""
        state = json.loads((state_dir / STATE_FILE).read_text(encoding="utf-8"))
        if state.get("version") != _STATE_VERSION:
            raise ValueError(
                f"unsupported scheduler state version {state.get('version')!r}"
            )
        cfg = state["config"]
        self = cls.__new__(cls)
        self._configure(
            tuple(cfg["rung_budgets"]),
            tuple(cfg["seeds_per_rung"]),
            cfg["eta"],
            cfg["beta"],
            state_dir,
            arm_id_of,
        )
        self._rung0_tasks = tuple(state["rung0_tasks"])
        for i, row in enumerate(state["arms"]):
            arm = _arm_from_json(row["arm"])
            self._index[arm] = i
            self._states.append(ArmState(arm, row["rung"], [], row["status"]))
        for i, rung, saved_id in state["arm_ids"]:
            self._register_id(i, rung)
            if self._id_at[(i, rung)] != saved_id:
                raise ValueError(
                    f"arm_id of {self._states[i].arm.label()}@rung{rung} changed from {saved_id} "
                    f"to {self._id_at[(i, rung)]}; the RunSpec defaults differ from the saved sweep"
                )
        self._promoted_out = {(t, r) for t, r in state["promoted_out"]}
        for row in state["running"]:
            i, rung = row["arm"], row["rung"]
            arm = self._states[i].arm
            job = Job(
                arm=arm, seed=row["seed"], rung=rung, step_budget=self._budgets[rung]
            )
            self._running[(self._id_at[(i, rung)], job.seed)] = job
        self._replay_results(state["n_results"])
        self._resubmit = list(self._running.values())
        self._running = {}
        self.save()
        return self

    def _replay_results(self, n_applied: int) -> None:
        """Attach the first `n_applied` logged results (already reflected in the saved
        statuses); apply later ones, which a crash kept out of the snapshot, as reports."""
        path = self._dir / RESULTS_FILE
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        if len(lines) < n_applied:
            raise ValueError(
                f"{path} has {len(lines)} results, the state expects {n_applied}"
            )
        for n, line in enumerate(lines, start=1):
            result = _result_from_json(json.loads(line))
            key = (result.arm_id, result.seed)
            if n <= n_applied:
                i, _ = self._ids[result.arm_id]
                self._states[i].results.append(result)
                continue
            job = self._running.pop(key, None)
            if job is None:
                raise ValueError(
                    f"{path} line {n}: result {key} matches no running job"
                )
            self._apply(job, result)
        self._n_results = len(lines)


# --- JSON codecs -------------------------------------------------------------------------------


def _kw_to_json(kw: Kw) -> list[list[Any]]:
    return [[k, _kw_to_json(v) if isinstance(v, tuple) else v] for k, v in kw]  # pyright: ignore[reportUnknownArgumentType] # nested kw pairs


def _kw_from_json(rows: list[list[Any]]) -> Kw:
    return tuple((k, _kw_from_json(v) if isinstance(v, list) else v) for k, v in rows)  # pyright: ignore[reportUnknownArgumentType] # nested kw pairs


def _arm_to_json(arm: Arm) -> dict[str, Any]:
    return {
        "task": arm.task,
        "task_kw": _kw_to_json(arm.task_kw),
        "method": arm.method,
        "method_kw": _kw_to_json(arm.method_kw),
        "score": arm.score,
        "score_kw": _kw_to_json(arm.score_kw),
    }


def _arm_from_json(row: dict[str, Any]) -> Arm:
    return Arm(
        task=row["task"],
        task_kw=_kw_from_json(row["task_kw"]),
        method=row["method"],
        method_kw=_kw_from_json(row["method_kw"]),
        score=row["score"],
        score_kw=_kw_from_json(row["score_kw"]),
    )


def _result_to_json(result: RunResult) -> dict[str, Any]:
    # JSON has no NaN: write null, read back as math.nan (ever_succeeded_share is NaN for
    # methods without per-IC outcomes).
    out = dataclasses.asdict(result)
    for name in _FLOAT_RESULT_FIELDS:
        if not math.isfinite(out[name]):
            out[name] = None
    return out


def _result_from_json(row: dict[str, Any]) -> RunResult:
    fields = dict(row)
    for name in _FLOAT_RESULT_FIELDS:
        if fields[name] is None:
            fields[name] = math.nan
    return RunResult(**fields)

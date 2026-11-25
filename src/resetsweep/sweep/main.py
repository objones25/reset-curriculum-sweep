"""`sweep-run`: the in-pod sweep entry point (spec sections 6 to 8).

Steps:
1. Settings from the environment (`load_settings`). `SWEEP_CAP_USD` (at most
   `CAP_USD_DEFAULT`, 100) and `SWEEP_GPU_USD_PER_H` are required (the launcher sets them); a
   missing or bad value raises `SettingsError` naming the variable, never echoing a value. The W&B key is read by wandb itself and never
   touched here.
2. Calibrate (`calibration.calibrate`), or reuse `state_dir/calibration.json`. Once a scheduler
   exists in the state dir, calibration is never re-run: the plan, and so every arm id, must
   stay what the saved sweep used.
3. Size the rungs (`size_rungs`): rung 0 runs every rung-0 arm once on about 35% of the
   remaining budget; r1 = 3 r0, r2 = 9 r0. On resume the saved budgets are used.
4. Loop (`SweepLoop`): poll results -> report -> kill jobs whose arm is no longer live ->
   heartbeat -> stop checks -> `next_jobs(1)` -> `can_start` -> submit, until the scheduler is
   done or the guard refuses a job; then wait for the jobs in flight.
5. Write the report (`report.write_report`) and log it to a sweep-level W&B run.
6. Return 0. Bad settings return 2; a new sweep whose budget is already spent, a budget too
   small to size rung 0, or calibration with no usable measurement returns 3; a state dir made
   by another mode or arm set (`sweep_meta.json`), or a scheduler state without its
   calibration.json, returns 4.

Live per-sweep metrics (spec 5.6) go to the sweep-level W&B run every `HEARTBEAT_S` and when
the loop stops: spend, projected spend (spend + the pod time owed to jobs in flight), jobs
running and done, arms by status, and the current rung per group. The report adds the arm
table (mean, CI, LCB/UCB, rung, status) and the final ranking to the same run.

`SWEEP_SMOKE=1`: calibrate the pendulum family only, then run only `uniform` and `visited`
(p_task 0.5) on `pendulum_sparse` for one rung of r0, and rank that rung in the report.

Budget handling (spec section 8):
- A job's wall-clock estimate is ``step_budget / (family steps/s / n_workers) + overhead``,
  scaled by a correction learned from finished jobs (median of actual / estimated over the last
  50, clipped to [0.5, 4]). The pod bills at full rate until its last job ends, however many
  workers are busy, so starting a job and then stopping costs ``max(remaining wall of the jobs
  in flight, the new job's wall)`` more pod-seconds. `can_start` gets exactly that: in_flight_s
  = the longest remaining in-flight wall, est_job_s = how far the new job reaches past it.
  (Rung sizing uses 1/n_workers of the pod per job instead: there the pool is full.)
- Hard stop: once spend reaches cap - margin, every running job is killed without a result
  (the scheduler keeps it as in flight, so a resumed sweep reruns it) and the report is
  written. The margin pays for the report and the pod's self-termination (Task 24).
- A job whose arm is no longer live (`Scheduler.is_live`) is killed and reported as failed, so
  the scheduler stops waiting for it. Killed jobs used pod time, and the guard counts wall
  clock, so their cost is in the ledger.
- Result handling: an ok result with a non-finite AUC is recorded as failed; a conflicting
  duplicate (`ConflictingResultError`) keeps the first result; a result with no running job
  (a stray worker after a restart) is dropped. Each case is logged.

Tunable sizes for tests and CPU smoke runs (not needed on the pod): `SWEEP_NUM_ENVS_GRID`,
`SWEEP_WORKERS_GRID`, `SWEEP_ROLLOUT_LEN`, `SWEEP_CALIB_STEPS`, `SWEEP_THROUGHPUT_ITERS`,
`SWEEP_EVAL_N`, and `SWEEP_R0_STEPS` (overrides the rung-0 sizing).
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol, get_args

from resetsweep.metrics import WandbMode
from resetsweep.sweep.arms import Arm, arm_space
from resetsweep.sweep.budget import (
    CALIBRATION_CAP_USD,
    CAP_USD_DEFAULT,
    HEARTBEAT_S,
    MARGIN_USD,
    NETWORK_VOLUME_GB,
    BudgetGuard,
    volume_usd_per_h,
)
from resetsweep.sweep.calibration import (
    CONFIG_STEPS,
    NUM_ENVS_GRID,
    ROLLOUT_LEN,
    THROUGHPUT_ITERS,
    WORKERS_GRID,
    CalibrationError,
    CalibrationRecord,
    CalibrationSettings,
    Plan,
    calibrate,
    choose_plan,
    family_of,
    sparse_families,
)
from resetsweep.sweep.pool import (
    Pool,
    PoolFactory,
    RunnerConfig,
    default_pool_factory,
    failed_result,
)
from resetsweep.sweep.report import (
    ReportSink,
    WandbReportSink,
    group_label,
    write_report,
)
from resetsweep.sweep.scheduler import (
    LIVE_STATUSES,
    STATE_FILE,
    ArmState,
    ConflictingResultError,
    Job,
    RunDefaults,
    Scheduler,
    Status,
    spec_for,
)
from resetsweep.train import RunResult, RunSpec

_log = logging.getLogger(__name__)

STATE_DIR_DEFAULT: Final = "/workspace/sweep"
PROJECT_DEFAULT: Final = "reset-curriculum-sweep"
RUNG0_SHARE: Final = 0.35
ETA: Final = 3
EVALS_PER_RUN: Final = 20  # eval points per run: the AUC's resolution
POLL_S: Final = 5.0
STATUS_EVERY_S: Final = 300.0
JOB_TIMEOUT_FACTOR: Final = 3.0
JOB_TIMEOUT_FLOOR_S: Final = 900.0
CALIBRATION_JOB_TIMEOUT_S: Final = 1800.0
CORRECTION_WINDOW: Final = 50
CORRECTION_BOUNDS: Final = (0.5, 4.0)
SMOKE_FAMILY: Final = "pendulum_dense"
SMOKE_LABELS: Final = ("pendulum_sparse/uniform/-", "pendulum_sparse/visited[p_task=0.5]/-")
_MAX_EVENTS: Final = 1_000_000  # loop passes that handle a result or a submission


class SettingsError(ValueError):
    """A sweep setting in the environment is missing or invalid."""


class BudgetTooSmallError(RuntimeError):
    """The remaining budget cannot pay for rung 0."""


# --- settings ----------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Settings:
    cap_usd: float
    gpu_usd_per_h: float
    state_dir: Path
    wandb_project: str
    wandb_mode: WandbMode
    smoke: bool
    num_envs_grid: tuple[int, ...]
    workers_grid: tuple[int, ...]
    rollout_len: int
    config_steps: int
    throughput_iters: int
    eval_n: int
    r0_steps: int | None


def _positive_float(env: Mapping[str, str], name: str) -> float:
    raw = env.get(name)
    if raw is None:
        raise SettingsError(f"{name} is not set")
    try:
        value = float(raw)
    except ValueError:
        raise SettingsError(f"{name} must be a finite number > 0") from None  # no value echoed
    if not (math.isfinite(value) and value > 0):
        raise SettingsError(f"{name} must be a finite number > 0")
    return value


def _positive_int(env: Mapping[str, str], name: str, default: int | None) -> int | None:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    if not raw.strip().isdigit() or int(raw) < 1:
        raise SettingsError(f"{name} must be a positive integer")
    return int(raw)


def _int_list(env: Mapping[str, str], name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    parts = [p.strip() for p in raw.split(",")]
    if not all(p.isdigit() and int(p) >= 1 for p in parts) or len(set(parts)) != len(parts):
        raise SettingsError(f"{name} must be distinct comma-separated positive integers")
    return tuple(int(p) for p in parts)


def load_settings(environ: Mapping[str, str]) -> Settings:
    """Parse the sweep settings; raises `SettingsError` naming the bad variable."""
    mode = environ.get("WANDB_MODE", "online")
    modes = get_args(WandbMode)
    if mode not in modes:
        raise SettingsError(f"WANDB_MODE must be one of {modes}")
    smoke_raw = environ.get("SWEEP_SMOKE", "0")
    if smoke_raw not in ("", "0", "1"):
        raise SettingsError("SWEEP_SMOKE must be 0 or 1")

    def required_int(name: str, default: int) -> int:
        value = _positive_int(environ, name, default)
        assert value is not None
        return value

    cap_usd = _positive_float(environ, "SWEEP_CAP_USD")
    if cap_usd > CAP_USD_DEFAULT:
        raise SettingsError(f"SWEEP_CAP_USD must be <= {CAP_USD_DEFAULT} (CAP_USD_DEFAULT)")
    return Settings(
        cap_usd=cap_usd,
        gpu_usd_per_h=_positive_float(environ, "SWEEP_GPU_USD_PER_H"),
        state_dir=Path(environ.get("SWEEP_STATE_DIR") or STATE_DIR_DEFAULT),
        wandb_project=environ.get("WANDB_PROJECT") or PROJECT_DEFAULT,
        wandb_mode=mode,  # pyright: ignore[reportArgumentType] # checked against WandbMode above
        smoke=smoke_raw == "1",
        num_envs_grid=_int_list(environ, "SWEEP_NUM_ENVS_GRID", NUM_ENVS_GRID),
        workers_grid=_int_list(environ, "SWEEP_WORKERS_GRID", WORKERS_GRID),
        rollout_len=required_int("SWEEP_ROLLOUT_LEN", ROLLOUT_LEN),
        config_steps=required_int("SWEEP_CALIB_STEPS", CONFIG_STEPS),
        throughput_iters=required_int("SWEEP_THROUGHPUT_ITERS", THROUGHPUT_ITERS),
        eval_n=required_int("SWEEP_EVAL_N", 512),
        r0_steps=_positive_int(environ, "SWEEP_R0_STEPS", None),
    )


# --- rung sizing -------------------------------------------------------------------------------


def size_rungs(
    *,
    arms_per_family: Mapping[str, int],
    plan: Plan,
    remaining_usd: float,
    usd_per_s: float,
    share: float = RUNG0_SHARE,
) -> tuple[int, int, int]:
    """(r0, 3 r0, 9 r0) with rung 0 (every arm once) costing `share` of `remaining_usd`.

    Pod time of rung 0 = sum over arms of ``r0 / steps_per_s_f + overhead_f / n_workers``
    (steps_per_s_f is the aggregate over n_workers, so each job holds 1/n of the pod).
    """
    if not (0 < share <= 1) or not (math.isfinite(usd_per_s) and usd_per_s > 0):
        raise ValueError(f"bad share {share} or usd_per_s {usd_per_s}")
    if not arms_per_family or any(n < 1 for n in arms_per_family.values()):
        raise ValueError(f"arms_per_family must be non-empty and positive: {arms_per_family}")
    budget_s = share * remaining_usd / usd_per_s
    fams = {f: plan.for_family(f) for f in arms_per_family}
    overhead_s = sum(n * fams[f].overhead_s for f, n in arms_per_family.items()) / plan.n_workers
    s_per_step = sum(n / fams[f].steps_per_s for f, n in arms_per_family.items())
    r0 = math.floor((budget_s - overhead_s) / s_per_step) if budget_s > overhead_s else 0
    min_r0 = max(fp.num_envs * fp.rollout_len for fp in fams.values())
    if r0 < min_r0:
        raise BudgetTooSmallError(
            f"rung 0 would get {r0} steps per run, below one rollout ({min_r0}); "
            f"{remaining_usd:.2f} USD remain, rung 0 may use {share:.0%} of it"
        )
    return r0, ETA * r0, ETA * ETA * r0


# --- the loop ----------------------------------------------------------------------------------


class SchedulerLike(Protocol):
    def next_jobs(self, k: int) -> list[Job]: ...

    def report(self, result: RunResult) -> None: ...

    def is_live(self, job: Job) -> bool: ...

    def done(self) -> bool: ...

    def promotions(self) -> None: ...

    def states(self) -> tuple[ArmState, ...]: ...


@dataclass(frozen=True)
class _InFlight:
    job: Job
    spec: RunSpec
    start_s: float
    est_wall_s: float  # uncorrected estimate, for learning the correction


class SweepLoop:
    """The main loop; see the module docstring for the rules."""

    def __init__(
        self,
        *,
        scheduler: SchedulerLike,
        pool: Pool,
        guard: BudgetGuard,
        plan: Plan,
        make_spec: Callable[[Job], RunSpec],
        clock: Callable[[], float],
        smoke: bool,
        sink: ReportSink,
        poll_s: float = POLL_S,
    ) -> None:
        self.guard = guard
        self._sink = sink
        self._last_live = -math.inf
        self._sched = scheduler
        self._pool = pool
        self._plan = plan
        self._make_spec = make_spec
        self._clock = clock
        self._smoke = smoke
        self._poll_s = poll_s
        self._inflight: dict[tuple[str, int], _InFlight] = {}
        self._ratios: list[float] = []
        self._n_results = 0
        self._last_status = -math.inf
        budget_polls = guard.cap_usd / guard.usd_per_s / poll_s
        self._max_iters = math.ceil(budget_polls) + _MAX_EVENTS

    # --- estimates ---

    def _raw_wall_s(self, job: Job) -> float:
        fp = self._plan.for_task(job.arm.task)
        return job.step_budget / (fp.steps_per_s / self._plan.n_workers) + fp.overhead_s

    def _correction(self) -> float:
        if not self._ratios:
            return 1.0
        lo, hi = CORRECTION_BOUNDS
        return min(hi, max(lo, statistics.median(self._ratios[-CORRECTION_WINDOW:])))

    def _in_flight_left_s(self) -> float:
        """Longest remaining wall clock among the jobs in flight: the pod time already owed."""
        now, c = self._clock(), self._correction()
        left = [max(0.0, f.est_wall_s * c - (now - f.start_s)) for f in self._inflight.values()]
        return max(left, default=0.0)

    # --- results ---

    def _handle(self, result: RunResult) -> None:
        key = (result.arm_id, result.seed)
        entry = self._inflight.pop(key, None)
        if result.status == "ok" and not math.isfinite(result.auc):
            _log.error("run %s: ok result with non-finite auc %s; recorded as failed", key, result.auc)
            result = dataclasses.replace(
                result, status="failed", error=f"ok result with non-finite auc {result.auc}"
            )
        if entry is not None and result.status == "ok" and entry.est_wall_s > 0:
            self._ratios.append((self._clock() - entry.start_s) / entry.est_wall_s)
        self._n_results += 1
        self.guard.heartbeat()  # a slow scheduler save must not starve the ledger
        log = _log.info if result.status == "ok" else _log.warning
        log(
            "result %s seed %d: %s auc=%.3f wall=%.0fs%s",
            result.arm_id,
            result.seed,
            result.status,
            result.auc,
            result.wall_s,
            f" error={result.error}" if result.error else "",
        )
        try:
            self._sched.report(result)
        except ConflictingResultError:
            _log.warning("conflicting duplicate for run %s; keeping the first result", key, exc_info=True)
        except ValueError:
            _log.warning("dropping result for run %s with no running job", key, exc_info=True)

    def _kill_dead(self) -> None:
        for job in self._pool.running():
            if self._sched.is_live(job):
                continue
            self._pool.kill(job)
            entry = next(f for f in self._inflight.values() if f.job == job)
            wall = self._clock() - entry.start_s
            _log.info("killed %s seed %d: arm no longer live", job.arm.label(), job.seed)
            self._handle(
                failed_result(
                    entry.spec,
                    "killed: job no longer live (arm pruned, failed or moved rung)",
                    wall_s=wall,
                )
            )

    def _kill_all(self) -> None:
        for job in self._pool.running():
            self._pool.kill(job)
            _log.warning("budget: killed %s seed %d; it reruns on resume", job.arm.label(), job.seed)
        self._inflight.clear()

    # --- submission ---

    def _smoke_rung0_over(self) -> bool:
        self._sched.promotions()
        return not any(st.status in LIVE_STATUSES and st.rung == 0 for st in self._sched.states())

    def _submit_free(self) -> bool:
        """Fill free workers; True if the guard refused a job."""
        while self._pool.free_slots() > 0:
            if self._smoke and self._smoke_rung0_over():
                return False
            jobs = self._sched.next_jobs(1)
            if not jobs:
                return False
            (job,) = jobs
            spec = self._make_spec(job)
            raw = self._raw_wall_s(job)
            est = raw * self._correction()
            owed = self._in_flight_left_s()
            # A refused job was already issued by `next_jobs`: it stays in flight in the saved
            # scheduler state, so a resumed sweep (with more budget) runs it first.
            if not self.guard.can_start(max(0.0, est - owed), owed):
                _log.warning(
                    "budget guard: not starting %s seed %d (about %.0f s); spent %.2f of %.2f USD "
                    "allowed; waiting for %d jobs in flight",
                    job.arm.label(),
                    job.seed,
                    est,
                    self.guard.spent(),
                    self.guard.limit_usd(),
                    len(self._inflight),
                )
                return True
            timeout_s = max(JOB_TIMEOUT_FLOOR_S, JOB_TIMEOUT_FACTOR * est)
            self._pool.submit(job, spec, timeout_s=timeout_s)
            self._inflight[(spec.arm_id(), spec.seed)] = _InFlight(job, spec, self._clock(), raw)
        return False

    def _live_metrics(self) -> dict[str, float]:
        """Per-sweep metrics (spec 5.6): spend, projection, jobs, arms by status, rung per group."""
        spent = self.guard.spent()
        owed_usd = self._in_flight_left_s() * self.guard.usd_per_s
        states = self._sched.states()
        counts = Counter(st.status for st in states)
        data: dict[str, float] = {
            "spent_usd": spent,
            "projected_usd": spent + owed_usd,
            "remaining_usd": self.guard.remaining(),
            "jobs_running": float(len(self._inflight)),
            "jobs_done": float(self._n_results),
        }
        data.update({f"arms/{status}": float(counts[status]) for status in get_args(Status)})
        rungs: dict[str, int] = {}
        for st in states:
            if st.status in LIVE_STATUSES:
                label = group_label((st.arm.task, st.arm.task_kw))
                rungs[label] = max(rungs.get(label, 0), st.rung)
        data.update({f"rung/{label}": float(r) for label, r in rungs.items()})
        return data

    def _log_live(self, *, force: bool = False) -> None:
        if not force and self.guard.elapsed_s() - self._last_live < HEARTBEAT_S:
            return
        self._last_live = self.guard.elapsed_s()
        self._sink.log(self._last_live, self._live_metrics())

    def _status(self) -> None:
        self._log_live()
        now = self._clock()
        if now - self._last_status < STATUS_EVERY_S:
            return
        self._last_status = now
        _log.info(
            "status: %d results, %d running, spent %.2f USD, %.2f left before cap - margin",
            self._n_results,
            len(self._inflight),
            self.guard.spent(),
            self.guard.remaining(),
        )

    def run(self) -> str:
        """Run until done, smoke rung 0 done, or the budget; returns why it stopped."""
        stopping: str | None = None
        for _ in range(self._max_iters):
            for result in self._pool.poll(timeout_s=self._poll_s):
                self._handle(result)
            self._kill_dead()
            self.guard.heartbeat()
            self._status()
            if self.guard.remaining() <= 0:
                self._kill_all()
                reason = "budget exhausted: spend reached cap - margin"
                break
            running = bool(self._pool.running())
            if stopping is not None:
                if not running:
                    reason = stopping
                    break
                continue
            if not running and self._smoke and self._smoke_rung0_over():
                reason = "smoke: rung 0 finished"
                break
            if not running and self._sched.done():
                reason = "done"
                break
            if self._submit_free():
                stopping = "budget: the next job would pass cap - margin"
            elif not self._pool.running() and not self._sched.done():
                raise RuntimeError("no job running, none to start, and the scheduler is not done")
        else:
            raise RuntimeError(f"sweep loop passed its bound of {self._max_iters} iterations")
        self.guard.flush()
        self._log_live(force=True)
        _log.info("sweep loop stopped: %s", reason)
        return reason


# --- composition root --------------------------------------------------------------------------


ReportSinkFactory = Callable[[str, str, WandbMode], ReportSink]


def _wandb_sweep_sink(project: str, group: str, mode: WandbMode) -> ReportSink:
    return WandbReportSink(project=project, group=group, mode=mode, config={"sweep_id": group})


def _sweep_id(state_dir: Path) -> str:
    path = state_dir / "sweep_id.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    sweep_id = datetime.now(UTC).strftime("sweep-%Y%m%d-%H%M%S")
    path.write_text(sweep_id + "\n", encoding="utf-8")
    return sweep_id


def _smoke_arms() -> list[Arm]:
    arms = [a for a in arm_space("rung0") if a.label() in SMOKE_LABELS]
    if sorted(a.label() for a in arms) != sorted(SMOKE_LABELS):
        raise RuntimeError(f"smoke arms {SMOKE_LABELS} not all in arm_space('rung0')")
    return arms


def _defaults(plan: Plan, arm: Arm, step_budget: int, settings: Settings, device: str) -> RunDefaults:
    fp = plan.for_task(arm.task)
    return RunDefaults(
        num_envs=fp.num_envs,
        rollout_len=fp.rollout_len,
        ppo=fp.ppo,
        eval_every_steps=max(1, step_budget // EVALS_PER_RUN),
        eval_n=settings.eval_n,
        device=device,
    )


def _calibration_settings(settings: Settings, device: str, cap_usd: float) -> CalibrationSettings:
    return CalibrationSettings(
        num_envs_grid=settings.num_envs_grid,
        workers_grid=settings.workers_grid,
        rollout_len=settings.rollout_len,
        config_steps=settings.config_steps,
        throughput_iters=settings.throughput_iters,
        eval_n=settings.eval_n,
        device=device,
        cap_usd=cap_usd,
        job_timeout_s=CALIBRATION_JOB_TIMEOUT_S,
        poll_s=POLL_S,
    )


META_FILE: Final = "sweep_meta.json"
CALIBRATION_FILE: Final = "calibration.json"


def _arms_sha256(arms: list[Arm]) -> str:
    return hashlib.sha256("\n".join(sorted(a.label() for a in arms)).encode()).hexdigest()


def _check_state(state: Path, *, smoke: bool, arms: list[Arm]) -> str | None:
    """Why `state` cannot be used for this sweep, or None (and the sweep's meta is saved).

    `sweep_meta.json` records the mode and a hash of the initial arm set on first start; a later
    start with another mode or arm set would resume a different sweep's state. A scheduler
    state without its calibration.json cannot rebuild the arm ids."""
    meta = {"version": 1, "mode": "smoke" if smoke else "full", "arms_sha256": _arms_sha256(arms)}
    path = state / META_FILE
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != meta:
            return (
                f"{path} is for a {saved.get('mode')} sweep with arm set {saved.get('arms_sha256')}, "
                f"this start is {meta['mode']} with arm set {meta['arms_sha256']}; use another "
                "SWEEP_STATE_DIR or the settings the state was made with"
            )
    elif any((state / name).exists() for name in (STATE_FILE, CALIBRATION_FILE, "results.jsonl")):
        return f"{state} holds sweep state but no {META_FILE}; refusing to guess what it is"
    if (state / STATE_FILE).exists() and not (state / CALIBRATION_FILE).exists():
        return f"{state / STATE_FILE} exists but {CALIBRATION_FILE} is missing: arm ids cannot be rebuilt"
    if not path.exists():
        path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    return None


def _sweep(
    settings: Settings,
    *,
    device: str,
    clock: Callable[[], float],
    pool_factory: PoolFactory,
    report_sink_factory: ReportSinkFactory,
) -> int:
    state = settings.state_dir
    state.mkdir(parents=True, exist_ok=True)
    arms = _smoke_arms() if settings.smoke else arm_space("rung0")
    problem = _check_state(state, smoke=settings.smoke, arms=arms)
    if problem is not None:
        _log.error("state dir mismatch: %s", problem)
        return 4
    sweep_id = _sweep_id(state)
    guard = BudgetGuard(
        cap_usd=settings.cap_usd,
        margin_usd=MARGIN_USD,
        gpu_usd_per_h=settings.gpu_usd_per_h,
        volume_usd_per_h=volume_usd_per_h(NETWORK_VOLUME_GB),
        ledger_path=state / "ledger.jsonl",
        clock=clock,
    )
    _log.info(
        "sweep %s on %s: cap %.2f USD, %.2f spent before this session, smoke=%s",
        sweep_id,
        device,
        settings.cap_usd,
        guard.spent(),
        settings.smoke,
    )
    if not (state / STATE_FILE).exists() and guard.remaining() <= 0:
        _log.error(
            "budget already spent: %.2f USD of the %.2f allowed (cap - margin); nothing started",
            guard.spent(),
            guard.limit_usd(),
        )
        return 3
    sink = report_sink_factory(settings.wandb_project, sweep_id, settings.wandb_mode)
    try:
        return _run_sweep(
            settings,
            arms=arms,
            sweep_id=sweep_id,
            guard=guard,
            sink=sink,
            device=device,
            clock=clock,
            pool_factory=pool_factory,
        )
    finally:
        sink.close()
        guard.flush()


def _run_sweep(
    settings: Settings,
    *,
    arms: list[Arm],
    sweep_id: str,
    guard: BudgetGuard,
    sink: ReportSink,
    device: str,
    clock: Callable[[], float],
    pool_factory: PoolFactory,
) -> int:
    state = settings.state_dir
    runner_cfg = RunnerConfig(
        project=settings.wandb_project,
        group=sweep_id,
        wandb_mode=settings.wandb_mode,
        runs_dir=state / "runs",
    )
    families = (SMOKE_FAMILY,) if settings.smoke else sparse_families()
    arms_per_family = Counter(family_of(a.task) for a in arms)
    calib_path = state / CALIBRATION_FILE
    resuming = (state / STATE_FILE).exists()
    prior_calib = CalibrationRecord.load(calib_path).spent_usd if calib_path.exists() else 0.0
    calib_cap = min(CALIBRATION_CAP_USD, prior_calib + max(guard.remaining(), 0.0))
    calib_settings = _calibration_settings(settings, device, max(calib_cap, 1e-9))
    try:
        if resuming:
            record = CalibrationRecord.load(calib_path)
        else:
            record = calibrate(
                families=families,
                settings=calib_settings,
                guard=guard,
                pool_factory=pool_factory,
                runner_cfg=runner_cfg,
                path=calib_path,
                clock=clock,
            )
        plan = choose_plan(record, arms_per_family=arms_per_family, settings=calib_settings)
    except CalibrationError:
        _log.exception("calibration gave nothing to size the sweep from")
        return 3
    sink.log(guard.elapsed_s(), {"calibration_usd": record.spent_usd, "spent_usd": guard.spent()})
    _log.info(
        "plan: %d workers; %s",
        plan.n_workers,
        "; ".join(
            f"{f}: N={p.num_envs} lr={p.ppo.lr:g} {p.steps_per_s:.0f} steps/s overhead {p.overhead_s:.0f}s"
            for f, p in plan.families.items()
        ),
    )

    def arm_id_of(arm: Arm, step_budget: int) -> str:
        defaults = _defaults(plan, arm, step_budget, settings, device)
        return spec_for(arm, seed=0, step_budget=step_budget, defaults=defaults).arm_id()

    def make_spec(job: Job) -> RunSpec:
        defaults = _defaults(plan, job.arm, job.step_budget, settings, device)
        return spec_for(job.arm, seed=job.seed, step_budget=job.step_budget, defaults=defaults)

    if resuming:
        scheduler = Scheduler.load(state, arm_id_of=arm_id_of)
        _log.info("resumed scheduler with rung budgets %s", scheduler.rung_budgets)
    else:
        try:
            if settings.r0_steps is not None:
                r0 = settings.r0_steps
                rungs = (r0, ETA * r0, ETA * ETA * r0)
            else:
                rungs = size_rungs(
                    arms_per_family=arms_per_family,
                    plan=plan,
                    remaining_usd=guard.remaining(),
                    usd_per_s=guard.usd_per_s,
                )
        except BudgetTooSmallError:
            _log.exception("budget too small for rung 0")
            return 3
        scheduler = Scheduler(arms=arms, rung_budgets=rungs, state_dir=state, arm_id_of=arm_id_of)
        _log.info("new scheduler: %d arms, rung budgets %s", len(arms), rungs)

    pool = pool_factory(plan.n_workers, device, runner_cfg)
    try:
        reason = SweepLoop(
            scheduler=scheduler,
            pool=pool,
            guard=guard,
            plan=plan,
            make_spec=make_spec,
            clock=clock,
            smoke=settings.smoke,
            sink=sink,
        ).run()
    finally:
        pool.close()
        guard.flush()

    extra: dict[str, float | str] = {
        "sweep_id": sweep_id,
        "stop_reason": reason,
        "spent_usd": guard.spent(),
        "cap_usd": settings.cap_usd,
        "calibration_usd": record.spent_usd,
        "n_workers": float(plan.n_workers),
    }
    write_report(
        scheduler,
        state / "report.md",
        state / "report.json",
        sink,
        rank_rung=0 if settings.smoke else None,
        extra=extra,
        tick=guard.heartbeat,
    )
    guard.flush()
    _log.info(
        "report written to %s; stopped: %s; spent %.2f of %.2f USD",
        state / "report.md",
        reason,
        guard.spent(),
        settings.cap_usd,
    )
    return 0


def _configure_logging() -> None:
    root = logging.getLogger()  # obs: allow LOG003 (the entry point configures the root)
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    clock: Callable[[], float] = time.monotonic,
    pool_factory: PoolFactory = default_pool_factory,
    report_sink_factory: ReportSinkFactory = _wandb_sweep_sink,
    device: str | None = None,
) -> int:
    """`sweep-run`. Settings come from the environment (see the module docstring)."""
    parser = argparse.ArgumentParser(
        prog="sweep-run",
        description="Run the reset-curriculum sweep in the pod. Configured by environment "
        "variables: SWEEP_CAP_USD and SWEEP_GPU_USD_PER_H (required), SWEEP_STATE_DIR, "
        "WANDB_PROJECT, WANDB_MODE, SWEEP_SMOKE.",
    )
    parser.parse_args(argv)
    _configure_logging()
    try:
        settings = load_settings(os.environ if environ is None else environ)
    except SettingsError as e:
        _log.error("bad settings: %s", e)  # obs: allow LOG006 (a clear message; no traceback)
        return 2
    if device is None:
        from resetsweep.device import pick_device

        device = pick_device()
    return _sweep(
        settings,
        device=device,
        clock=clock,
        pool_factory=pool_factory,
        report_sink_factory=report_sink_factory,
    )

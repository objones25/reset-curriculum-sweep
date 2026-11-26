"""`sweep-run`: the in-pod sweep entry point (spec sections 6 to 8).

Steps:
1. Settings from the environment (`load_settings`). `SWEEP_CAP_USD` (at most
   `CAP_USD_DEFAULT`, 100) and `SWEEP_GPU_USD_PER_H` are required (the launcher sets them); a
   missing or bad value raises `SettingsError` naming the variable, never echoing a value. The W&B key is read by wandb itself and never
   touched here.
2. Calibrate (`calibration.calibrate`), or reuse `state_dir/calibration.json`. Once a scheduler
   exists in the state dir, calibration is never re-run: the plan, and so every arm id, must
   stay what the saved sweep used.
3. Size the rungs (`size_rungs`, final-review ruling C2b): `ladder.project_ladder` gives the
   expected arm count per family at each rung from the real arm space (quota ceil(n / eta) per
   group plus the expected rung-1 and final-rung expansions); a rung-k arm costs
   seeds_k x budget_k = 1, 9, 45 rung-0 runs, times the job's measured probe slowdown
   (`Plan.method_factors`; unprobed methods x1). r0 is chosen so the whole expected ladder costs
   `LADDER_SHARE` (90%) of what calibration left, at the calibrated steps/s and worker count;
   r1 = 3 r0, r2 = 9 r0. The projected USD per rung and r0 are logged. If r0 would fall below
   `SWEEP_MIN_R0_STEPS` (default 2,000,000) the sweep refuses to start (exit 5) and logs the
   numbers and the knobs: a paid run must not produce noise. Smoke mode (`size_smoke_rung`)
   sizes rung 0 only, to cost `SMOKE_SHARE` (50%) of what calibration left, with every job
   whose (family, method) has no calibration probe (including uniform on a family smoke did
   not calibrate, e.g. maze_trap) at `UNPROBED_FACTOR` (3x), then clamps r0 into
   [`SWEEP_SMOKE_MIN_R0_STEPS` (default 100,000), `SWEEP_SMOKE_R0_MAX_STEPS` (default
   1,000,000)]; below the floor it raises r0 rather than exiting 5.
   On resume the saved budgets are used.
4. Loop (`SweepLoop`): poll results -> report -> kill jobs whose arm is no longer live ->
   heartbeat -> stop checks -> `next_jobs(1)` -> `can_start` -> submit, until the scheduler is
   done or the guard refuses a job; then wait for the jobs in flight.
5. Write the report (`report.write_report`) and log it to a sweep-level W&B run.
6. Return 0. Bad settings return 2; a new sweep whose budget is already spent, or calibration
   with no usable measurement, returns 3; r0 below `SWEEP_MIN_R0_STEPS` returns 5; device "cpu" while `RUNPOD_POD_ID` is set (a
   pod whose GPU is not visible) returns 6; a state dir made
   by another mode or arm set or on another commit (`sweep_meta.json` records `REPO_REF`;
   skipped when it is unset), or a scheduler state without its
   calibration.json, returns 4.

Stopping before the pod's deadline (final-review I2): the bootstrap exports
`SWEEP_DEADLINE_EPOCH`, the epoch second at which it TERMs this process. The guard's limit
becomes min(cap - margin, the spend at deadline - `SWEEP_REPORT_RESERVE_S` (default 300 s)): no
job starts that would run past that time, and at that time every running job is killed (it
reruns on resume) and the report is written. SIGTERM or SIGINT at any point after calibration
kills the workers, saves the scheduler state, writes the report (stop reason "terminated by
SIGTERM") and returns 128 + the signal number; scheduler writes run with both signals blocked,
so a signal never splits a result from its state save. `sweep-report --state-dir DIR`
(`report_main`) rebuilds report.md and report.json offline from the saved state.

Live per-sweep metrics (spec 5.6) go to `state_dir/sweep_metrics.jsonl` and the sweep-level W&B
run every `HEARTBEAT_S` and when the loop stops. A W&B failure, at start or mid-run, is logged,
counted in `sink_errors` (in the JSONL) and never stops the sweep (`GuardedSweepSink`). Metrics: spend, projected spend (spend + the pod time owed to jobs in flight), jobs
running and done, arms by status, and the current rung per group. The report adds the arm
table (mean, CI, LCB/UCB, rung, status) and the final ranking to the same run.

`SWEEP_SMOKE=1`: calibrate the pendulum family only (plus the arm-family probes:
`visited` and `buffer_scored` on arm_sparse), then run only `uniform` and `visited` (p_task 0.5)
on `pendulum_sparse` and `uniform` on `maze_trap` for one rung of r0, and rank that rung in the
report (final-review I8). After calibration, smoke also projects the FULL ladder
(`project_full_ladder`, smoke-patch item 2): what `size_rungs` would give the full arm set at
`SWEEP_PROJECT_CAP_USD` (default 100; the launcher passes the account cap) minus the $5 margin
and the whole $10 calibration cap, using smoke-calibrated throughput (measured only for the
families smoke calibrated, the fallback for the rest) and the measured method factors. It
prints one "full-ladder projection" line to stdout, logs it, and adds `full_projection_*`
keys (arms and USD per rung, r0, whether a full run would exit 5) to report.md, report.json
and the W&B summary.

Budget handling (spec section 8):
- A job's wall-clock estimate is
  ``job_factor(family, method) * step_budget / (family steps/s / n_workers) + overhead``
  (`Plan.job_factor`, final-review I5: the calibration probe's measured slowdown against
  uniform; for the timeout ceiling an unprobed method counts 3x uniform, for the budget 1x),
  scaled by a correction learned from finished jobs (median of actual / estimated over the last
  50, clipped to [0.5, 4]). The pod bills at full rate until its last job ends, however many
  workers are busy, so starting a job and then stopping costs ``max(remaining wall of the jobs
  in flight, the new job's wall)`` more pod-seconds. `can_start` gets exactly that: in_flight_s
  = the longest remaining in-flight wall, est_job_s = how far the new job reaches past it.
  (Rung sizing uses 1/n_workers of the pod per job instead: there the pool is full.)
- Timeouts are a hang guard, not a duration limit (final-review I5): a job is killed only if
  its heartbeat file (touched by the worker on every metrics log) has not moved for
  `SWEEP_HANG_S` (default 900 s), or past ``max(JOB_TIMEOUT_FLOOR_S, 4 x estimate)``.
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
`SWEEP_EVAL_N`, `SWEEP_R0_STEPS` (overrides the rung sizing; the projected cost is still
logged), `SWEEP_MIN_R0_STEPS` (the refusal floor above), `SWEEP_SMOKE_MIN_R0_STEPS` and
`SWEEP_SMOKE_R0_MAX_STEPS` (smoke's r0 clamp).
"""

import argparse
import dataclasses
import hashlib
import itertools
import json
import logging
import math
import os
import signal
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Final, NoReturn, Protocol, get_args

from resetsweep.metrics import JsonlSink, WandbMode
from resetsweep.sweep.arms import Arm, arm_space
from resetsweep.sweep.budget import (
    CALIBRATION_CAP_USD,
    CAP_USD_DEFAULT,
    HEARTBEAT_S,
    MARGIN_USD,
    NETWORK_VOLUME_GB,
    BudgetGuard,
    margin_usd,
    volume_usd_per_h,
)
from resetsweep.sweep.calibration import (
    CONFIG_STEPS,
    NUM_ENVS_GRID,
    ROLLOUT_LEN,
    THROUGHPUT_ITERS,
    UNPROBED_FACTOR,
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
from resetsweep.sweep.ladder import project_ladder
from resetsweep.sweep.pool import (
    Pool,
    PoolFactory,
    RunnerConfig,
    default_pool_factory,
    failed_result,
    heartbeat_path,
)
from resetsweep.sweep.report import (
    GuardedSweepSink,
    NullReportSink,
    ReportSink,
    ReportSource,
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
LADDER_SHARE: Final = 0.9  # the expected ladder may use 90% of what calibration left
MIN_R0_STEPS_DEFAULT: Final = 2_000_000
# Smoke-patch item 1: smoke sizes its one rung to cost half of what calibration left, with every
# job whose (family, method) has no calibration probe at UNPROBED_FACTOR, then clamps r0 into
# [SWEEP_SMOKE_MIN_R0_STEPS, SWEEP_SMOKE_R0_MAX_STEPS]. The floor raises r0; it never refuses.
SMOKE_SHARE: Final = 0.5
SMOKE_MIN_R0_STEPS_DEFAULT: Final = 100_000
SMOKE_R0_MAX_STEPS_DEFAULT: Final = 1_000_000
# Smoke-patch item 2: smoke projects the full ladder at SWEEP_PROJECT_CAP_USD (the launcher
# passes the account cap).
PROJECT_CAP_USD_DEFAULT: Final = CAP_USD_DEFAULT
REPORT_RESERVE_S_DEFAULT: Final = 300.0  # stop this long before SWEEP_DEADLINE_EPOCH
SEEDS_PER_RUNG: Final = (1, 3, 5)
ETA: Final = 3
EVALS_PER_RUN: Final = 20  # eval points per run: the AUC's resolution
POLL_S: Final = 5.0
STATUS_EVERY_S: Final = 300.0
JOB_TIMEOUT_FACTOR: Final = (
    4.0  # absolute ceiling: 4 x the job's estimate (final-review I5)
)
HANG_S_DEFAULT: Final = 900.0  # kill a job whose heartbeat has not moved for this long
JOB_TIMEOUT_FLOOR_S: Final = 900.0
CALIBRATION_JOB_TIMEOUT_S: Final = 1800.0
CORRECTION_WINDOW: Final = 50
CORRECTION_BOUNDS: Final = (0.5, 4.0)
SMOKE_FAMILY: Final = "pendulum_dense"
# Final-review I8: smoke also runs the arm-family probes (visited / buffer_scored on arm_sparse:
# the CPU contact check) and one maze_trap uniform run (substep graph + applied force).
SMOKE_PROBE_FAMILIES: Final = ("pendulum_dense", "arm_dense")
SMOKE_LABELS: Final = (
    "pendulum_sparse/uniform/-",
    "pendulum_sparse/visited[p_task=0.5]/-",
    "maze_trap/uniform/-",
)
_MAX_EVENTS: Final = 1_000_000  # loop passes that handle a result or a submission


class SettingsError(ValueError):
    """A sweep setting in the environment is missing or invalid."""


class SweepTerminated(BaseException):  # like KeyboardInterrupt: not an Exception
    """SIGTERM or SIGINT arrived; raised by the signal handler `main` installs."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum
        self.name = signal.Signals(signum).name


_STOP_SIGNALS: Final = (signal.SIGTERM, signal.SIGINT)


def _on_stop_signal(signum: int, frame: FrameType | None) -> NoReturn:
    del frame
    for sig in (
        _STOP_SIGNALS
    ):  # one stop is enough: ignore repeats while the report is written
        signal.signal(sig, signal.SIG_IGN)
    raise SweepTerminated(signum)


class _SignalSafeScheduler:
    """Blocks SIGTERM/SIGINT around the scheduler calls that write its state, so a stop
    signal never lands between appending a result and saving the state that reflects it
    (`signal.pthread_sigmask`, Unix; a pending signal is delivered when the mask is restored)."""

    def __init__(self, inner: Scheduler) -> None:
        self.inner = inner

    def _masked[T](self, fn: Callable[[], T]) -> T:
        old = signal.pthread_sigmask(signal.SIG_BLOCK, set(_STOP_SIGNALS))
        try:
            return fn()
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)

    def next_jobs(self, k: int, *, max_rung: int | None = None) -> list[Job]:
        return self._masked(lambda: self.inner.next_jobs(k, max_rung=max_rung))

    def report(self, result: RunResult) -> None:
        self._masked(lambda: self.inner.report(result))

    def promotions(self) -> None:
        self._masked(self.inner.promotions)

    def save(self) -> None:
        self._masked(self.inner.save)

    def is_live(self, job: Job) -> bool:
        return self.inner.is_live(job)

    def done(self) -> bool:
        return self._masked(self.inner.done)

    def states(self) -> tuple[ArmState, ...]:
        return self.inner.states()


class BudgetTooSmallError(RuntimeError):
    """The remaining budget cannot pay for the ladder at the minimum r0."""


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
    min_r0_steps: int
    smoke_min_r0_steps: int
    smoke_r0_max_steps: int
    project_cap_usd: float
    repo_ref: str | None  # REPO_REF: the commit the pod checked out (None off the pod)
    deadline_epoch: float | None
    report_reserve_s: float
    hang_s: float


def _positive_float(env: Mapping[str, str], name: str) -> float:
    raw = env.get(name)
    if raw is None:
        raise SettingsError(f"{name} is not set")
    try:
        value = float(raw)
    except ValueError:
        raise SettingsError(
            f"{name} must be a finite number > 0"
        ) from None  # no value echoed
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


def _int_list(
    env: Mapping[str, str], name: str, default: tuple[int, ...]
) -> tuple[int, ...]:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    parts = [p.strip() for p in raw.split(",")]
    if not all(p.isdigit() and int(p) >= 1 for p in parts) or len(set(parts)) != len(
        parts
    ):
        raise SettingsError(
            f"{name} must be distinct comma-separated positive integers"
        )
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
        raise SettingsError(
            f"SWEEP_CAP_USD must be <= {CAP_USD_DEFAULT} (CAP_USD_DEFAULT)"
        )
    smoke_min = required_int("SWEEP_SMOKE_MIN_R0_STEPS", SMOKE_MIN_R0_STEPS_DEFAULT)
    smoke_max = required_int("SWEEP_SMOKE_R0_MAX_STEPS", SMOKE_R0_MAX_STEPS_DEFAULT)
    if smoke_max < smoke_min:
        raise SettingsError(
            "SWEEP_SMOKE_R0_MAX_STEPS must be >= SWEEP_SMOKE_MIN_R0_STEPS"
        )
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
        min_r0_steps=required_int("SWEEP_MIN_R0_STEPS", MIN_R0_STEPS_DEFAULT),
        smoke_min_r0_steps=smoke_min,
        smoke_r0_max_steps=smoke_max,
        repo_ref=environ.get("REPO_REF") or None,
        project_cap_usd=(
            _positive_float(environ, "SWEEP_PROJECT_CAP_USD")
            if environ.get("SWEEP_PROJECT_CAP_USD")
            else PROJECT_CAP_USD_DEFAULT
        ),
        deadline_epoch=(
            _positive_float(environ, "SWEEP_DEADLINE_EPOCH")
            if environ.get("SWEEP_DEADLINE_EPOCH")
            else None
        ),
        hang_s=(
            _positive_float(environ, "SWEEP_HANG_S")
            if environ.get("SWEEP_HANG_S")
            else HANG_S_DEFAULT
        ),
        report_reserve_s=(
            _positive_float(environ, "SWEEP_REPORT_RESERVE_S")
            if environ.get("SWEEP_REPORT_RESERVE_S")
            else REPORT_RESERVE_S_DEFAULT
        ),
    )


# --- rung sizing -------------------------------------------------------------------------------


@dataclass(frozen=True)
class RungSizing:
    """The rung step budgets and the projected USD each rung costs (expected arm counts)."""

    rungs: tuple[int, int, int]
    cost_usd: tuple[float, ...]


_Line = tuple[float, float]  # pod-seconds = slope * r0 + intercept
ArmKey = tuple[str, str]  # (family, method): the unit rung sizing counts arms in
JobFactor = Callable[[str, str], float]  # (family, method) -> job wall vs uniform's


def full_job_factor(plan: Plan) -> JobFactor:
    """Full-run sizing's per-job factor (smoke-patch item 2): the probe's measured slowdown
    when (family, method) has one, else 1 (the loop's learned correction covers the rest)."""

    def factor(family: str, method: str) -> float:
        return plan.method_factors.get((family, method), 1.0)

    return factor


def smoke_job_factor(plan: Plan) -> JobFactor:
    """Smoke sizing's per-job factor: the probe's measured slowdown when (family, method) has
    one, 1 for uniform on a calibrated family, else `UNPROBED_FACTOR` (3). Uniform on a family
    smoke never calibrated (maze_trap) counts as unprobed: its fallback speed is a guess."""

    def factor(family: str, method: str) -> float:
        measured = plan.method_factors.get((family, method))
        if measured is not None:
            return measured
        if method == "uniform" and family in plan.families:
            return 1.0
        return UNPROBED_FACTOR

    return factor


def _rung_lines(
    counts: Mapping[ArmKey, float],
    plan: Plan,
    seeds: int,
    mult: int,
    factor: JobFactor,
) -> tuple[_Line, _Line]:
    """Two lower bounds on one rung's pod time, as lines in r0. Full pool: a job holds
    1/n_workers of the pod, so its pod time is ``x * mult * r0 / steps_per_s_f + overhead_f /
    n_workers`` (steps_per_s_f is the aggregate over n_workers, x = factor(family, method)).
    Not-full pool (fewer jobs than workers, e.g. smoke): the pod bills at least one whole job
    wall, ``x * mult * r0 / (steps_per_s_f / n_workers) + overhead_f``, for the slowest job."""
    keys = [k for k, n in counts.items() if n > 0]
    fps = {k: plan.for_family(k[0]) for k in keys}
    xs = {k: factor(*k) for k in keys}
    share = (
        sum(counts[k] * seeds * mult * xs[k] / fps[k].steps_per_s for k in keys),
        sum(counts[k] * seeds * fps[k].overhead_s for k in keys) / plan.n_workers,
    )
    wall = max(
        (
            (mult * xs[k] * plan.n_workers / fps[k].steps_per_s, fps[k].overhead_s)
            for k in keys
        ),
        default=(0.0, 0.0),
    )
    return share, wall


def _ladder_lines(
    arms_per_rung: Sequence[Mapping[ArmKey, float]],
    plan: Plan,
    seeds: Sequence[int],
    factor: JobFactor,
) -> list[tuple[_Line, _Line]]:
    return [
        _rung_lines(counts, plan, seeds[k], ETA**k, factor)
        for k, counts in enumerate(arms_per_rung)
    ]


def _rollout_steps(counts: Mapping[ArmKey, float], plan: Plan) -> int:
    """The largest one-rollout batch among the families in `counts`: the smallest usable r0."""
    return max(
        plan.for_family(f).num_envs * plan.for_family(f).rollout_len for f, _ in counts
    )


def _rung_cost_usd(
    lines: list[tuple[_Line, _Line]], r0: int, usd_per_s: float
) -> tuple[float, ...]:
    return tuple(max(a * r0 + b for a, b in pair) * usd_per_s for pair in lines)


def _max_r0(lines: list[tuple[_Line, _Line]], budget_s: float) -> int:
    """Largest r0 with sum over rungs of max(line_1, line_2) <= budget_s. The sum of maxima is
    the maximum over every choice of one line per rung, so it is <= budget_s exactly when each
    choice is: r0 = min over choices of (budget_s - sum intercepts) / sum slopes."""
    best = math.inf
    for choice in itertools.product(*lines):
        slope = sum(a for a, _ in choice)
        free = budget_s - sum(b for _, b in choice)
        if free <= 0:
            return 0
        best = min(best, free / slope)
    return math.floor(best)


def size_rungs(
    *,
    arms_per_rung: Sequence[Mapping[ArmKey, float]],
    plan: Plan,
    remaining_usd: float,
    usd_per_s: float,
    min_r0_steps: int,
    seeds_per_rung: Sequence[int] = SEEDS_PER_RUNG,
    share: float = LADDER_SHARE,
) -> RungSizing:
    """(r0, 3 r0, 9 r0) with the whole expected ladder costing `share` of `remaining_usd`.

    `arms_per_rung[k]` maps (family, method) -> expected arms at rung k
    (`ladder.project_ladder`); a
    rung-k arm costs ``seeds_per_rung[k] * ETA**k`` r0-runs (1, 9, 45). A rung's pod time is
    the larger of its full-pool share and one whole job wall (`_rung_lines`). Raises
    `BudgetTooSmallError` when r0 would be below `min_r0_steps` or one rollout: a paid run must
    not produce noise.
    """
    if not (0 < share <= 1) or not (math.isfinite(usd_per_s) and usd_per_s > 0):
        raise ValueError(f"bad share {share} or usd_per_s {usd_per_s}")
    if not arms_per_rung or len(arms_per_rung) > len(seeds_per_rung):
        raise ValueError(
            f"need 1..{len(seeds_per_rung)} rungs of arm counts, got {len(arms_per_rung)}"
        )
    if not arms_per_rung[0] or any(n < 0 for c in arms_per_rung for n in c.values()):
        raise ValueError(
            f"arm counts must be non-negative with a non-empty rung 0: {arms_per_rung}"
        )
    budget_s = share * remaining_usd / usd_per_s
    lines = _ladder_lines(arms_per_rung, plan, seeds_per_rung, full_job_factor(plan))
    r0 = _max_r0(lines, budget_s)
    rollout = _rollout_steps(arms_per_rung[0], plan)
    floor_steps = max(min_r0_steps, rollout)
    if r0 < floor_steps:
        totals = ", ".join(f"{sum(c.values()):.0f}" for c in arms_per_rung)
        overhead_s = sum(max(b for _, b in pair) for pair in lines)
        raise BudgetTooSmallError(
            f"r0 would be {r0} env steps per rung-0 run, below the minimum {floor_steps} "
            f"(SWEEP_MIN_R0_STEPS={min_r0_steps}, one rollout={rollout}): {remaining_usd:.2f} USD "
            f"remain, the ladder may use {share:.0%} of it = {budget_s:.0f} pod-s, of which "
            f"{overhead_s:.0f} s is job overhead; expected arms per rung {totals} at seeds "
            f"{tuple(seeds_per_rung[: len(arms_per_rung)])}. Knobs: fewer tasks, fewer arms, fewer "
            "seeds, a higher cap, or a lower SWEEP_MIN_R0_STEPS"
        )
    return RungSizing(
        rungs=(r0, ETA * r0, ETA * ETA * r0),
        cost_usd=_rung_cost_usd(lines, r0, usd_per_s),
    )


def size_smoke_rung(
    *,
    arms: Mapping[ArmKey, float],
    plan: Plan,
    remaining_usd: float,
    usd_per_s: float,
    min_r0_steps: int,
    max_r0_steps: int,
) -> RungSizing:
    """Smoke's one rung (smoke-patch item 1): the largest r0 whose projected cost is
    `SMOKE_SHARE` (50%) of `remaining_usd`, every job weighted by `smoke_job_factor`, then
    clamped into [max(min_r0_steps, one rollout), max_r0_steps]. Below the floor r0 is raised
    to it (and may then cost more than half); smoke never refuses with exit 5."""
    if not arms or any(n < 0 for n in arms.values()):
        raise ValueError(f"smoke arm counts must be non-negative and non-empty: {arms}")
    if not (math.isfinite(usd_per_s) and usd_per_s > 0):
        raise ValueError(f"bad usd_per_s {usd_per_s}")
    lines = [_rung_lines(arms, plan, 1, 1, smoke_job_factor(plan))]
    sized = _max_r0(lines, SMOKE_SHARE * remaining_usd / usd_per_s)
    floor_steps = max(min_r0_steps, _rollout_steps(arms, plan))
    r0 = min(max_r0_steps, max(sized, floor_steps))
    if r0 > sized:
        _log.warning(
            "smoke r0 raised from %d to the floor %d (SWEEP_SMOKE_MIN_R0_STEPS=%d, one "
            "rollout=%d): the smoke rung may cost more than %.0f%% of the %.2f USD remaining",
            sized,
            r0,
            min_r0_steps,
            _rollout_steps(arms, plan),
            SMOKE_SHARE * 100,
            remaining_usd,
        )
    return RungSizing(
        rungs=(r0, ETA * r0, ETA * ETA * r0),
        cost_usd=_rung_cost_usd(lines, r0, usd_per_s),
    )


def ladder_cost_usd(
    arms_per_rung: Sequence[Mapping[ArmKey, float]],
    plan: Plan,
    r0: int,
    usd_per_s: float,
    *,
    factor: JobFactor,
) -> tuple[float, ...]:
    """Projected USD per rung for a given r0 (used to log an `SWEEP_R0_STEPS` override)."""
    return _rung_cost_usd(
        _ladder_lines(arms_per_rung, plan, SEEDS_PER_RUNG, factor), r0, usd_per_s
    )


@dataclass(frozen=True, kw_only=True)
class FullLadderProjection:
    """What `size_rungs` would give a full sweep at `cap_usd` (smoke-patch item 2)."""

    cap_usd: float
    remaining_usd: float  # cap - MARGIN_USD - CALIBRATION_CAP_USD
    arms_per_rung: tuple[float, ...]
    cost_usd: tuple[float, ...]
    r0: int
    min_r0_steps: int
    would_exit_5: bool


def project_full_ladder(
    *,
    arms_per_rung: Sequence[Mapping[ArmKey, float]],
    plan: Plan,
    cap_usd: float,
    usd_per_s: float,
    min_r0_steps: int,
) -> FullLadderProjection:
    """The full ladder at a hypothetical `SWEEP_CAP_USD = cap_usd`: the budget is what a full run
    would have after its `MARGIN_USD` and, at worst, the whole `CALIBRATION_CAP_USD`; then the
    same lines, share and floor as `size_rungs` (measured method factors included), without
    raising, so the caller can say whether the full run would exit 5."""
    remaining = cap_usd - MARGIN_USD - CALIBRATION_CAP_USD
    lines = _ladder_lines(arms_per_rung, plan, SEEDS_PER_RUNG, full_job_factor(plan))
    r0 = _max_r0(lines, LADDER_SHARE * remaining / usd_per_s)
    floor_steps = max(min_r0_steps, _rollout_steps(arms_per_rung[0], plan))
    return FullLadderProjection(
        cap_usd=cap_usd,
        remaining_usd=remaining,
        arms_per_rung=tuple(sum(c.values()) for c in arms_per_rung),
        cost_usd=_rung_cost_usd(lines, r0, usd_per_s),
        r0=r0,
        min_r0_steps=min_r0_steps,
        would_exit_5=r0 < floor_steps,
    )


def _projection_note(plan: Plan) -> str:
    measured = ", ".join(f"{f} {p.steps_per_s:.0f}" for f, p in plan.families.items())
    factors = (
        ", ".join(f"{f}/{m} x{x:.2f}" for (f, m), x in plan.method_factors.items())
        or "none"
    )
    return (
        "smoke-calibrated: steps/s measured only for the families smoke calibrated "
        f"({measured}); every other family uses the fallback "
        f"{plan.fallback.steps_per_s:.0f} steps/s; measured method factors: {factors}; "
        "unprobed methods x1"
    )


def _projection_extra(proj: FullLadderProjection, plan: Plan) -> dict[str, float | str]:
    extra: dict[str, float | str] = {
        "full_projection_note": _projection_note(plan),
        "full_projection_cap_usd": proj.cap_usd,
        "full_projection_remaining_usd": proj.remaining_usd,
        "full_projection_r0": float(proj.r0),
        "full_projection_min_r0_steps": float(proj.min_r0_steps),
        "full_projection_exits_5": "yes" if proj.would_exit_5 else "no",
    }
    for k, (n, usd) in enumerate(zip(proj.arms_per_rung, proj.cost_usd, strict=True)):
        extra[f"full_projection_arms_r{k}"] = n
        extra[f"full_projection_usd_r{k}"] = usd
    return extra


def _projection_line(proj: FullLadderProjection, plan: Plan) -> str:
    return (
        f"full-ladder projection ({_projection_note(plan)}): cap {proj.cap_usd:.2f} USD -> "
        f"{proj.remaining_usd:.2f} USD after the {MARGIN_USD:.2f} margin and the "
        f"{CALIBRATION_CAP_USD:.2f} calibration cap; expected arms per rung "
        + ", ".join(f"{n:.1f}" for n in proj.arms_per_rung)
        + "; projected USD per rung "
        + ", ".join(f"{c:.2f}" for c in proj.cost_usd)
        + f" (total {sum(proj.cost_usd):.2f}); r0={proj.r0}; SWEEP_MIN_R0_STEPS="
        f"{proj.min_r0_steps}: a full run would "
        + (
            "EXIT 5 (r0 below the minimum)"
            if proj.would_exit_5
            else "start (r0 >= the minimum)"
        )
    )


# --- the loop ----------------------------------------------------------------------------------


class SchedulerLike(Protocol):
    def next_jobs(self, k: int, *, max_rung: int | None = None) -> list[Job]: ...

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
        heartbeat_of: Callable[[RunSpec], Path] | None = None,
        hang_s: float = HANG_S_DEFAULT,
    ) -> None:
        self._heartbeat_of = heartbeat_of
        self._hang_s = hang_s
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

    def _raw_wall_s(self, job: Job, *, unprobed: float = 1.0) -> float:
        """Uncorrected wall-clock estimate. Budget: measured probe slowdowns only (unprobed
        methods count as uniform; the learned correction covers the rest). Timeouts pass
        `UNPROBED_FACTOR` for methods without a probe."""
        fp = self._plan.for_task(job.arm.task)
        factor = self._plan.job_factor(fp.family, job.arm.method, unprobed=unprobed)
        return (
            factor * job.step_budget / (fp.steps_per_s / self._plan.n_workers)
            + fp.overhead_s
        )

    def _correction(self) -> float:
        if not self._ratios:
            return 1.0
        lo, hi = CORRECTION_BOUNDS
        return min(hi, max(lo, statistics.median(self._ratios[-CORRECTION_WINDOW:])))

    def _in_flight_left_s(self) -> float:
        """Longest remaining wall clock among the jobs in flight: the pod time already owed."""
        now, c = self._clock(), self._correction()
        left = [
            max(0.0, f.est_wall_s * c - (now - f.start_s))
            for f in self._inflight.values()
        ]
        return max(left, default=0.0)

    # --- results ---

    def _handle(self, result: RunResult) -> None:
        key = (result.arm_id, result.seed)
        entry = self._inflight.pop(key, None)
        if result.status == "ok" and not math.isfinite(result.auc):
            _log.error(
                "run %s: ok result with non-finite auc %s; recorded as failed",
                key,
                result.auc,
            )
            result = dataclasses.replace(
                result,
                status="failed",
                error=f"ok result with non-finite auc {result.auc}",
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
            _log.warning(
                "conflicting duplicate for run %s; keeping the first result",
                key,
                exc_info=True,
            )
        except ValueError:
            _log.warning(
                "dropping result for run %s with no running job", key, exc_info=True
            )

    def _kill_dead(self) -> None:
        for job in self._pool.running():
            if self._sched.is_live(job):
                continue
            self._pool.kill(job)
            entry = next(f for f in self._inflight.values() if f.job == job)
            wall = self._clock() - entry.start_s
            _log.info(
                "killed %s seed %d: arm no longer live", job.arm.label(), job.seed
            )
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
            _log.warning(
                "budget: killed %s seed %d; it reruns on resume",
                job.arm.label(),
                job.seed,
            )
        self._inflight.clear()

    # --- submission ---

    def _smoke_rung0_over(self) -> bool:
        self._sched.promotions()
        return not any(
            st.status in LIVE_STATUSES and st.rung == 0 for st in self._sched.states()
        )

    def _submit_free(self) -> bool:
        """Fill free workers; True if the guard refused a job."""
        while self._pool.free_slots() > 0:
            if self._smoke and self._smoke_rung0_over():
                return False
            # Smoke runs rung 0 only: a group that finished it must not start rung 1 while
            # another group is still on rung 0.
            jobs = self._sched.next_jobs(1, max_rung=0 if self._smoke else None)
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
            ceiling_est = (
                self._raw_wall_s(job, unprobed=UNPROBED_FACTOR) * self._correction()
            )
            timeout_s = max(
                JOB_TIMEOUT_FLOOR_S, JOB_TIMEOUT_FACTOR * max(est, ceiling_est)
            )
            if self._heartbeat_of is None:
                self._pool.submit(job, spec, timeout_s=timeout_s)
            else:
                self._pool.submit(
                    job,
                    spec,
                    timeout_s=timeout_s,
                    heartbeat=self._heartbeat_of(spec),
                    hang_s=self._hang_s,
                )
            self._inflight[(spec.arm_id(), spec.seed)] = _InFlight(
                job, spec, self._clock(), raw
            )
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
        data.update(
            {f"arms/{status}": float(counts[status]) for status in get_args(Status)}
        )
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
                reason = (
                    "deadline: stopped SWEEP_REPORT_RESERVE_S before SWEEP_DEADLINE_EPOCH"
                    if self.guard.deadline_reached()
                    else "budget exhausted: spend reached cap - margin"
                )
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
                stopping = (
                    "deadline: the next job would run past SWEEP_DEADLINE_EPOCH - "
                    "SWEEP_REPORT_RESERVE_S"
                    if self.guard.deadline_bound()
                    else "budget: the next job would pass cap - margin"
                )
            elif not self._pool.running() and not self._sched.done():
                raise RuntimeError(
                    "no job running, none to start, and the scheduler is not done"
                )
        else:
            raise RuntimeError(
                f"sweep loop passed its bound of {self._max_iters} iterations"
            )
        self.guard.flush()
        self._log_live(force=True)
        _log.info("sweep loop stopped: %s", reason)
        return reason


# --- composition root --------------------------------------------------------------------------


ReportSinkFactory = Callable[[str, str, WandbMode], ReportSink]


def _wandb_sweep_sink(project: str, group: str, mode: WandbMode) -> ReportSink:
    return WandbReportSink(
        project=project, group=group, mode=mode, config={"sweep_id": group}
    )


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


def _defaults(
    plan: Plan, arm: Arm, step_budget: int, settings: Settings, device: str
) -> RunDefaults:
    fp = plan.for_task(arm.task)
    return RunDefaults(
        num_envs=fp.num_envs,
        rollout_len=fp.rollout_len,
        ppo=fp.ppo,
        eval_every_steps=max(1, step_budget // EVALS_PER_RUN),
        eval_n=settings.eval_n,
        device=device,
    )


def _calibration_settings(
    settings: Settings, device: str, cap_usd: float
) -> CalibrationSettings:
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
SWEEP_METRICS_FILE: Final = (
    "sweep_metrics.jsonl"  # local copy of every live sweep metric
)
CALIBRATION_FILE: Final = "calibration.json"


def _arms_sha256(arms: list[Arm]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(a.label() for a in arms)).encode()
    ).hexdigest()


def _check_state(
    state: Path, *, smoke: bool, arms: list[Arm], repo_ref: str | None
) -> str | None:
    """Why `state` cannot be used for this sweep, or None (and the sweep's meta is saved).

    `sweep_meta.json` records the mode, a hash of the initial arm set and the commit
    (`REPO_REF`, null off the pod) on first start; a later start with another mode or arm set
    would resume a different sweep's state, and one on another commit would resume it on
    different code (smoke-patch item 3). The commit check is skipped when either side has no
    value. A scheduler state without its calibration.json cannot rebuild the arm ids."""
    meta = {
        "version": 1,
        "mode": "smoke" if smoke else "full",
        "arms_sha256": _arms_sha256(arms),
    }
    path = state / META_FILE
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return f"{path} is corrupt ({e}); restore it or use another SWEEP_STATE_DIR"
        if not isinstance(saved, dict):
            return f"{path} is corrupt (not a JSON object); restore it or use another SWEEP_STATE_DIR"
        if {k: saved.get(k) for k in meta} != meta:
            return (
                f"{path} is for a {saved.get('mode')} sweep with arm set {saved.get('arms_sha256')}, "
                f"this start is {meta['mode']} with arm set {meta['arms_sha256']}; use another "
                "SWEEP_STATE_DIR or the settings the state was made with"
            )
        saved_ref = saved.get("repo_ref")
        if saved_ref is not None and repo_ref is not None and saved_ref != repo_ref:
            return (
                f"{path} was started on commit {saved_ref}, this start is on commit {repo_ref} "
                f"(REPO_REF); resume with sweep-launch --ref {saved_ref}, or use another "
                "SWEEP_STATE_DIR"
            )
    elif any(
        (state / name).exists()
        for name in (STATE_FILE, CALIBRATION_FILE, "results.jsonl")
    ):
        return f"{state} holds sweep state but no {META_FILE}; refusing to guess what it is"
    if (state / STATE_FILE).exists() and not (state / CALIBRATION_FILE).exists():
        return f"{state / STATE_FILE} exists but {CALIBRATION_FILE} is missing: arm ids cannot be rebuilt"
    if not path.exists():
        path.write_text(
            json.dumps({**meta, "repo_ref": repo_ref}) + "\n", encoding="utf-8"
        )
    return None


def _arm_counts(arms: list[Arm], *, smoke: bool) -> list[dict[ArmKey, float]]:
    """Expected arms per (family, method) at each rung the sweep will run (smoke: rung 0)."""
    ladder = project_ladder(arms, eta=ETA, n_rungs=1 if smoke else len(SEEDS_PER_RUNG))
    out: list[dict[ArmKey, float]] = []
    for counts in ladder.per_rung_by_method:
        keyed: dict[ArmKey, float] = {}
        for (task, method), n in counts.items():
            key = (family_of(task), method)
            keyed[key] = keyed.get(key, 0.0) + n
        out.append(keyed)
    return out


def _plan_rungs(
    settings: Settings, arms: list[Arm], plan: Plan, guard: BudgetGuard
) -> tuple[int, int, int] | None:
    """Rung budgets for a new sweep, logged with the projected cost per rung; None (after
    logging the numbers and the knobs) when r0 would fall below the minimum."""
    counts = _arm_counts(arms, smoke=settings.smoke)
    remaining = guard.remaining()
    if settings.r0_steps is not None:
        r0 = settings.r0_steps
        rungs = (r0, ETA * r0, ETA * ETA * r0)
        factor = smoke_job_factor(plan) if settings.smoke else full_job_factor(plan)
        cost = ladder_cost_usd(counts, plan, r0, guard.usd_per_s, factor=factor)
        source = "SWEEP_R0_STEPS"
    elif settings.smoke:
        sizing = size_smoke_rung(
            arms=counts[0],
            plan=plan,
            remaining_usd=remaining,
            usd_per_s=guard.usd_per_s,
            min_r0_steps=settings.smoke_min_r0_steps,
            max_r0_steps=settings.smoke_r0_max_steps,
        )
        rungs, cost = sizing.rungs, sizing.cost_usd
        source = (
            f"smoke: {SMOKE_SHARE:.0%} of remaining, unprobed jobs x{UNPROBED_FACTOR:g}, "
            f"clamped to [{settings.smoke_min_r0_steps}, {settings.smoke_r0_max_steps}]"
        )
    else:
        try:
            sizing = size_rungs(
                arms_per_rung=counts,
                plan=plan,
                remaining_usd=remaining,
                usd_per_s=guard.usd_per_s,
                min_r0_steps=settings.min_r0_steps,
            )
        except BudgetTooSmallError as e:
            _log.error(
                "refusing to start: %s", e
            )  # obs: allow LOG006 (numbers and knobs; no traceback)
            return None
        rungs, cost, source = (
            sizing.rungs,
            sizing.cost_usd,
            f"{LADDER_SHARE:.0%} of remaining",
        )
    _log.info(
        "ladder: r0=%d (%s); expected arms per rung %s; projected USD per rung %s, total %.2f "
        "of %.2f remaining",
        rungs[0],
        source,
        ", ".join(f"{sum(c.values()):.1f}" for c in counts),
        ", ".join(f"{c:.2f}" for c in cost),
        sum(cost),
        remaining,
    )
    return rungs


def _stop_at(
    settings: Settings, clock: Callable[[], float], epoch_clock: Callable[[], float]
) -> float | None:
    """`SWEEP_DEADLINE_EPOCH - SWEEP_REPORT_RESERVE_S` on the guard's (monotonic) clock."""
    if settings.deadline_epoch is None:
        return None
    left_s = settings.deadline_epoch - settings.report_reserve_s - epoch_clock()
    _log.info(
        "work deadline: stop scheduling and kill jobs in %.0f s (%.0f s before epoch %.0f)",
        left_s,
        settings.report_reserve_s,
        settings.deadline_epoch,
    )
    return clock() + left_s


def _final_report(
    scheduler: ReportSource,
    state: Path,
    sink: ReportSink,
    *,
    smoke: bool,
    extra: Mapping[str, float | str],
    tick: Callable[[], None] = lambda: None,
) -> None:
    write_report(
        scheduler,
        state / "report.md",
        state / "report.json",
        sink,
        rank_rung=0 if smoke else None,
        extra=extra,
        tick=tick,
    )


def _sweep(
    settings: Settings,
    *,
    device: str,
    clock: Callable[[], float],
    epoch_clock: Callable[[], float],
    pool_factory: PoolFactory,
    report_sink_factory: ReportSinkFactory,
) -> int:
    state = settings.state_dir
    state.mkdir(parents=True, exist_ok=True)
    arms = _smoke_arms() if settings.smoke else arm_space("rung0")
    problem = _check_state(
        state, smoke=settings.smoke, arms=arms, repo_ref=settings.repo_ref
    )
    if problem is not None:
        _log.error("state dir mismatch: %s", problem)
        return 4
    sweep_id = _sweep_id(state)
    guard = BudgetGuard(
        cap_usd=settings.cap_usd,
        margin_usd=margin_usd(smoke=settings.smoke),
        gpu_usd_per_h=settings.gpu_usd_per_h,
        volume_usd_per_h=volume_usd_per_h(NETWORK_VOLUME_GB),
        ledger_path=state / "ledger.jsonl",
        clock=clock,
        stop_at=_stop_at(settings, clock, epoch_clock),
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
    start_error: Exception | None = None
    try:
        primary = report_sink_factory(
            settings.wandb_project, sweep_id, settings.wandb_mode
        )
    except Exception as e:  # noqa: BLE001 # W&B down at start (any error type): sweep on the JSONL copy
        primary, start_error = NullReportSink(), e
    sink = GuardedSweepSink(primary, JsonlSink(state / SWEEP_METRICS_FILE))
    if start_error is not None:
        sink.failed("start", start_error)
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
    prior_calib = (
        CalibrationRecord.load(calib_path).spent_usd if calib_path.exists() else 0.0
    )
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
                probe_families=SMOKE_PROBE_FAMILIES if settings.smoke else None,
            )
        plan = choose_plan(
            record, arms_per_family=arms_per_family, settings=calib_settings
        )
    except CalibrationError:
        _log.exception("calibration gave nothing to size the sweep from")
        return 3
    sink.log(
        guard.elapsed_s(),
        {"calibration_usd": record.spent_usd, "spent_usd": guard.spent()},
    )
    _log.info(
        "plan: %d workers; %s",
        plan.n_workers,
        "; ".join(
            f"{f}: N={p.num_envs} lr={p.ppo.lr:g} {p.steps_per_s:.0f} steps/s overhead {p.overhead_s:.0f}s"
            for f, p in plan.families.items()
        ),
    )

    projection: dict[str, float | str] = {}
    if settings.smoke:
        proj = project_full_ladder(
            arms_per_rung=_arm_counts(arm_space("rung0"), smoke=False),
            plan=plan,
            cap_usd=settings.project_cap_usd,
            usd_per_s=guard.usd_per_s,
            min_r0_steps=settings.min_r0_steps,
        )
        projection = _projection_extra(proj, plan)
        line = _projection_line(proj, plan)
        print(line, flush=True)  # stdout: the operator reads this off the pod log
        _log.info("%s", line)

    def arm_id_of(arm: Arm, step_budget: int) -> str:
        defaults = _defaults(plan, arm, step_budget, settings, device)
        return spec_for(
            arm, seed=0, step_budget=step_budget, defaults=defaults
        ).arm_id()

    def make_spec(job: Job) -> RunSpec:
        defaults = _defaults(plan, job.arm, job.step_budget, settings, device)
        return spec_for(
            job.arm, seed=job.seed, step_budget=job.step_budget, defaults=defaults
        )

    if resuming:
        scheduler = Scheduler.load(state, arm_id_of=arm_id_of)
        _log.info("resumed scheduler with rung budgets %s", scheduler.rung_budgets)
    else:
        rungs = _plan_rungs(settings, arms, plan, guard)
        if rungs is None:
            return 5
        scheduler = Scheduler(
            arms=arms, rung_budgets=rungs, state_dir=state, arm_id_of=arm_id_of
        )
        _log.info("new scheduler: %d arms, rung budgets %s", len(arms), rungs)

    safe = _SignalSafeScheduler(scheduler)
    pool = pool_factory(plan.n_workers, device, runner_cfg)
    stopped: SweepTerminated | None = None
    try:
        reason = SweepLoop(
            scheduler=safe,
            pool=pool,
            guard=guard,
            plan=plan,
            make_spec=make_spec,
            clock=clock,
            smoke=settings.smoke,
            sink=sink,
            heartbeat_of=lambda spec: heartbeat_path(runner_cfg.runs_dir, spec),
            hang_s=settings.hang_s,
        ).run()
    except SweepTerminated as e:
        # Workers die in pool.close below; their jobs stay in flight in the saved state, so a
        # resumed sweep reruns them.
        stopped, reason = e, f"terminated by {e.name}"
        _log.warning("%s: killing workers, saving state, writing the report", e.name)
    finally:
        pool.close()
        guard.flush()
    if stopped is not None:
        safe.save()

    extra: dict[str, float | str] = {
        "sweep_id": sweep_id,
        "stop_reason": reason,
        "spent_usd": guard.spent(),
        "cap_usd": settings.cap_usd,
        "calibration_usd": record.spent_usd,
        "n_workers": float(plan.n_workers),
        **projection,
    }
    _final_report(
        scheduler, state, sink, smoke=settings.smoke, extra=extra, tick=guard.heartbeat
    )
    guard.flush()
    _log.info(
        "report written to %s; stopped: %s; spent %.2f of %.2f USD",
        state / "report.md",
        reason,
        guard.spent(),
        settings.cap_usd,
    )
    return 0 if stopped is None else 128 + stopped.signum


def _configure_logging() -> None:
    root = (
        logging.getLogger()
    )  # obs: allow LOG003 (the entry point configures the root)
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
    epoch_clock: Callable[[], float] = time.time,
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
    env = os.environ if environ is None else environ
    try:
        settings = load_settings(env)
    except SettingsError as e:
        _log.error(
            "bad settings: %s", e
        )  # obs: allow LOG006 (a clear message; no traceback)
        return 2
    if device is None:
        from resetsweep.device import pick_device

        device = pick_device()
    if device == "cpu" and env.get("RUNPOD_POD_ID"):
        _log.error(
            "device is cpu on RunPod pod %s: refusing to run the sweep on the CPU at GPU prices",
            env.get("RUNPOD_POD_ID"),
        )
        return 6
    previous = {sig: signal.signal(sig, _on_stop_signal) for sig in _STOP_SIGNALS}
    try:
        return _sweep(
            settings,
            device=device,
            clock=clock,
            epoch_clock=epoch_clock,
            pool_factory=pool_factory,
            report_sink_factory=report_sink_factory,
        )
    except SweepTerminated as e:
        _log.warning("%s before the sweep loop started: no report to write", e.name)
        return 128 + e.signum
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _saved_arm_ids(state: Path) -> Callable[[Arm, int], str]:
    """`arm_id_of` that returns the ids `scheduler.json` recorded, for an offline rebuild that
    has neither the calibration plan nor the run settings."""
    from resetsweep.sweep.scheduler import _arm_from_json  # pyright: ignore[reportPrivateUsage] # the state's own codec

    saved = json.loads((state / STATE_FILE).read_text(encoding="utf-8"))
    arms = [_arm_from_json(row["arm"]) for row in saved["arms"]]
    budgets = saved["config"]["rung_budgets"]
    ids = {(arms[i], budgets[rung]): arm_id for i, rung, arm_id in saved["arm_ids"]}

    def arm_id_of(arm: Arm, step_budget: int) -> str:
        return ids[(arm, step_budget)]

    return arm_id_of


def report_main(argv: list[str] | None = None) -> int:
    """`sweep-report --state-dir DIR`: rebuild report.md and report.json from saved state
    (no W&B). Run it only while no `sweep-run` uses DIR: loading the scheduler rewrites its
    scheduler.json (same content, in-flight jobs kept) and truncates a torn results.jsonl tail."""
    parser = argparse.ArgumentParser(
        prog="sweep-report", description="Rebuild the sweep report from a state dir."
    )
    parser.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    _configure_logging()
    state: Path = args.state_dir
    if not (state / STATE_FILE).exists():
        _log.error("%s has no %s: nothing to rebuild", state, STATE_FILE)
        return 2
    meta_path = state / META_FILE
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    scheduler = Scheduler.load(state, arm_id_of=_saved_arm_ids(state))
    sweep_id_path = state / "sweep_id.txt"
    extra: dict[str, float | str] = {"stop_reason": "rebuilt offline by sweep-report"}
    if sweep_id_path.exists():
        extra["sweep_id"] = sweep_id_path.read_text(encoding="utf-8").strip()
    _final_report(
        scheduler,
        state,
        NullReportSink(),
        smoke=meta.get("mode") == "smoke",
        extra=extra,
    )
    _log.info("report rebuilt in %s", state / "report.md")
    return 0

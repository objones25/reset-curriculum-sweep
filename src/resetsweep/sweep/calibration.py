"""Calibration (spec 5.5 and 7.1; Task 11 ruling; Task 23 contracts): turn dollars into steps.

Per sparse task family (the family is the task's `dense_sibling`), in this order, so the most
useful numbers exist first if the $10 cap (`CALIBRATION_CAP_USD`) stops it early:

1. config: `uniform` on the dense sibling with two PPO configs that differ only in lr
   (`LR_GRID`: 1e-3, the START_CONFIG value whose kl_stopped fired on ~62% of updates in
   Task 11, and 3e-4), `config_steps` each, run side by side. Each records the final eval
   return and its kl_stopped rate; the higher final return wins (spec 5.5: by final
   dense-sibling return).
2. throughput: steps/s of the winning config at each num_envs in {2048, 4096, 8192} and each
   n_workers in {1, 2, 4}: n identical jobs run at once for `throughput_iters` iterations, and
   the aggregate is the sum of their median per-iteration `perf/steps_per_s` (the first
   iteration, which compiles kernels, is dropped). A batch with any failed job (e.g. OOM at
   4 x 8192) is unusable. (base N, 1 worker) runs for every family before the rest of the grid.
3. probes: the known performance risks, measured and recorded (`PROBES`): the arm task's CPU
   contact `is_valid` on every visited offer and reset (`visited` and `buffer_scored` on
   arm_sparse), `buffer_scored._probs` syncing the host each reset, and the population
   methods' per-update IC churn and `walk` waiting for 2 trusted ICs (pendulum_sparse). Each
   records steps/s and the last value of every `method/`, `method_update/` and `ic/` metric.

Per-job overhead (spawn, imports, kernel compile, evals) is measured on the config jobs as
wall clock from submit to result minus ``env_steps / steps_per_s``.

Everything lands in `calibration.json` after each batch (atomic replace), with the spend so far
(guard spend while calibrating, summed over sessions). On restart, finished batches are reused
and calibration continues where it stopped.

The cap is never passed. Each batch gets a budget: what is left of the calibration cap, and
never more than the sweep's own `guard.remaining()`. A batch starts only if that budget pays
for at least one poll interval of pod time; its jobs' timeout is cut to what the budget pays
for (the pod bills at full rate while any job runs, so the budget in seconds is the longest a
batch may run); and the batch loop polls no further than the budget's end and closes the pool
(killing its jobs) once the budget is spent. A batch cut short that way is not stored, and
calibration stops as capped. Jobs killed by their cut timeout are stored as failed.

`Plan.job_factor(family, method)` scales a job's wall-clock estimate (final-review I5): a probed
method's measured slowdown against uniform, else `UNPROBED_FACTOR` (3); uniform is 1.

`choose_plan` picks one worker count for the sweep and, per family, the smallest num_envs whose
aggregate steps/s is within 10% of that family's best at that worker count (smaller N means
more PPO updates per env step); the worker count minimises the rung-0 pod time
``sum_f arms_f / steps_per_s_f``. Families without measurements (e.g. `claw`, or after the cap)
get a conservative fallback: the slowest measured steps/s and the largest overhead.
"""

import json
import logging
import math
import os
import statistics
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from resetsweep.ppo.update import START_CONFIG, PPOConfig
from resetsweep.sweep.arms import SPARSE_TASKS, Arm, arm_space
from resetsweep.sweep.budget import CALIBRATION_CAP_USD, BudgetGuard
from resetsweep.sweep.pool import PoolFactory, RunnerConfig, run_log_path
from resetsweep.sweep.scheduler import Job, RunDefaults, spec_for
from resetsweep.tasks.base import TASKS
from resetsweep.train import RunResult, RunSpec

_log = logging.getLogger(__name__)

NUM_ENVS_GRID: Final = (2048, 4096, 8192)
WORKERS_GRID: Final = (1, 2, 4)
# massively-parallel-rl references/pytorch-gpu-sim.md section 2: rsl_rl 4096 envs x 24 steps.
ROLLOUT_LEN: Final = 24
LR_GRID: Final = (1e-3, 3e-4)
NEAR_BEST: Final = 0.9
CALIBRATION_JOB_RUNG: Final = (
    -1
)  # marks calibration jobs; the scheduler never sees them
CONFIG_STEPS: Final = 10_000_000
THROUGHPUT_ITERS: Final = 10
_VERSION: Final = 1
_DIAG_PREFIXES: Final = ("method/", "method_update/", "ic/")
# family -> (task, method) probes; each uses the rung-0 arm with p_task 0.5 and score
# None or "random", so the score itself costs nothing.
PROBES: Final[Mapping[str, tuple[tuple[str, str], ...]]] = {
    "arm_dense": (("arm_sparse", "visited"), ("arm_sparse", "buffer_scored")),
    "pendulum_dense": (
        ("pendulum_sparse", "buffer_scored"),
        ("pendulum_sparse", "walk"),
    ),
}


class CalibrationError(RuntimeError):
    """Calibration produced nothing the sweep can be sized from."""


def family_of(task: str) -> str:
    return TASKS.get(task).dense_sibling  # pyright: ignore[reportAttributeAccessIssue] # every task declares it


def sparse_families() -> tuple[str, ...]:
    return tuple(sorted({family_of(t) for t in SPARSE_TASKS}))


def minibatch_for(num_envs: int, rollout_len: int) -> int:
    """START_CONFIG's fixed 24,576 when the rollout holds it; else N*T/4 (CPU test sizes)."""
    batch = num_envs * rollout_len
    fixed = START_CONFIG.minibatch_size
    return fixed if batch >= fixed else max(1, batch // 4)


def ppo_for(lr: float, num_envs: int, rollout_len: int) -> PPOConfig:
    return replace(
        START_CONFIG, lr=lr, minibatch_size=minibatch_for(num_envs, rollout_len)
    )


@dataclass(frozen=True, kw_only=True)
class CalibrationSettings:
    num_envs_grid: tuple[int, ...] = NUM_ENVS_GRID
    workers_grid: tuple[int, ...] = WORKERS_GRID
    rollout_len: int = ROLLOUT_LEN
    config_steps: int = CONFIG_STEPS
    throughput_iters: int = THROUGHPUT_ITERS
    eval_n: int = 512
    device: str
    cap_usd: float = CALIBRATION_CAP_USD
    job_timeout_s: float = 3600.0
    poll_s: float = 5.0

    def __post_init__(self) -> None:
        for name in ("num_envs_grid", "workers_grid"):
            grid: tuple[int, ...] = getattr(self, name)
            if not grid or any(v < 1 for v in grid) or len(set(grid)) != len(grid):
                raise ValueError(f"{name} must be distinct positive ints, got {grid}")
        for name in ("rollout_len", "config_steps", "throughput_iters", "eval_n"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("cap_usd", "job_timeout_s", "poll_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")

    @property
    def base_num_envs(self) -> int:
        """The middle of the grid (4096 by default): where configs and probes run."""
        grid = sorted(self.num_envs_grid)
        return grid[len(grid) // 2]


@dataclass(frozen=True, kw_only=True)
class Measurement:
    key: str
    family: str
    label: str
    num_envs: int
    n_workers: int
    lr: float
    seed: int
    status: str
    error: str | None
    steps_per_s: float | None
    final_return: float | None
    kl_stopped_rate: float | None
    elapsed_s: float
    overhead_s: float | None
    diagnostics: dict[str, float | None] = field(
        default_factory=dict[str, float | None]
    )


@dataclass
class CalibrationRecord:
    batches: dict[str, list[Measurement]]
    spent_usd: float = 0.0
    capped: bool = False

    def batch_keys(self) -> list[str]:
        return list(self.batches)

    def config(self, family: str) -> list[Measurement]:
        return list(self.batches.get(f"config/{family}", []))

    def throughput(self, family: str) -> dict[tuple[int, int], float | None]:
        """(num_envs, n_workers) -> aggregate steps/s, None if the batch had a failure."""
        out: dict[tuple[int, int], float | None] = {}
        for key, ms in self.batches.items():
            if not key.startswith(f"throughput/{family}/"):
                continue
            usable = all(m.status == "ok" and m.steps_per_s is not None for m in ms)
            total = sum(m.steps_per_s or 0.0 for m in ms)
            out[(ms[0].num_envs, ms[0].n_workers)] = total if usable and ms else None
        return out

    def probes(self) -> list[Measurement]:
        return [
            m for k, ms in self.batches.items() if k.startswith("probe/") for m in ms
        ]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "spent_usd": self.spent_usd,
            "capped": self.capped,
            "batches": {k: [asdict(m) for m in ms] for k, ms in self.batches.items()},
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "CalibrationRecord":
        if data.get("version") != _VERSION:
            raise CalibrationError(
                f"unsupported calibration.json version {data.get('version')!r}"
            )
        batches = {
            k: [Measurement(**m) for m in ms] for k, ms in data["batches"].items()
        }
        return cls(
            batches=batches,
            spent_usd=float(data["spent_usd"]),
            capped=bool(data["capped"]),
        )

    def save(self, path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as f:
            f.write(json.dumps(self.to_json(), indent=1, allow_nan=False))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "CalibrationRecord":
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))


# --- running -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Batch:
    key: str
    family: str
    n_workers: int
    jobs: tuple[tuple[Job, RunSpec, float], ...]  # (job, spec, lr)


def _uniform_on(family: str) -> Arm:
    return Arm(
        task=family, task_kw=(), method="uniform", method_kw=(), score=None, score_kw=()
    )


def _probe_arm(task: str, method: str) -> Arm:
    for arm in arm_space("rung0"):
        kw = arm.method_kwargs()
        matches = (arm.task, arm.method) == (task, method) and kw.get("p_task") == 0.5
        if matches and arm.score in (None, "random"):
            return arm
    raise ValueError(f"no rung-0 arm for probe {task}/{method}")


def _job(
    arm: Arm,
    *,
    seed: int,
    num_envs: int,
    lr: float,
    budget: int,
    s: CalibrationSettings,
) -> tuple[Job, RunSpec, float]:
    defaults = RunDefaults(
        num_envs=num_envs,
        rollout_len=s.rollout_len,
        ppo=ppo_for(lr, num_envs, s.rollout_len),
        eval_every_steps=budget,
        eval_n=s.eval_n,
        device=s.device,
    )
    spec = spec_for(arm, seed=seed, step_budget=budget, defaults=defaults)
    return (
        Job(arm=arm, seed=seed, rung=CALIBRATION_JOB_RUNG, step_budget=budget),
        spec,
        lr,
    )


def chosen_lr(record: CalibrationRecord, family: str) -> float | None:
    ok = [
        m
        for m in record.config(family)
        if m.status == "ok" and m.final_return is not None
    ]
    if not ok:
        return None
    return max(ok, key=lambda m: (m.final_return, -m.lr)).lr


def _planned(
    families: Sequence[str],
    s: CalibrationSettings,
    record: CalibrationRecord,
    probe_families: Sequence[str],
) -> Iterator[_Batch]:
    """Batches in priority order; lazily, because throughput needs the config winner."""
    base = s.base_num_envs
    for fam in families:
        jobs = tuple(
            _job(
                _uniform_on(fam),
                seed=0,
                num_envs=base,
                lr=lr,
                budget=s.config_steps,
                s=s,
            )
            for lr in LR_GRID
        )
        yield _Batch(f"config/{fam}", fam, len(jobs), jobs)
    grid = [(base, min(s.workers_grid))] + [
        (n_envs, n)
        for n_envs in s.num_envs_grid
        for n in s.workers_grid
        if (n_envs, n) != (base, min(s.workers_grid))
    ]
    for n_envs, n in grid:
        for fam in families:
            lr = chosen_lr(record, fam) or START_CONFIG.lr
            budget = s.throughput_iters * n_envs * s.rollout_len
            jobs = tuple(
                _job(
                    _uniform_on(fam),
                    seed=seed,
                    num_envs=n_envs,
                    lr=lr,
                    budget=budget,
                    s=s,
                )
                for seed in range(n)
            )
            yield _Batch(f"throughput/{fam}/{n_envs}/{n}", fam, n, jobs)
    for fam in probe_families:
        lr = chosen_lr(record, fam) or START_CONFIG.lr
        for task, method in PROBES.get(fam, ()):
            arm = _probe_arm(task, method)
            job = _job(arm, seed=0, num_envs=base, lr=lr, budget=s.config_steps, s=s)
            yield _Batch(f"probe/{fam}/{arm.label()}", fam, 1, (job,))


def _read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return [r for r in rows if r.get("type") == "log"]


def _finite(value: object) -> float | None:
    return (
        float(value)
        if isinstance(value, int | float) and math.isfinite(value)
        else None
    )


def _measure(
    batch: _Batch,
    job: Job,
    spec: RunSpec,
    lr: float,
    result: RunResult,
    elapsed_s: float,
    runs_dir: Path,
) -> Measurement:
    common: dict[str, Any] = {
        "key": batch.key,
        "family": batch.family,
        "label": job.arm.label(),
        "num_envs": spec.num_envs,
        "n_workers": batch.n_workers,
        "lr": lr,
        "seed": spec.seed,
        "elapsed_s": elapsed_s,
    }
    rows = _read_log(run_log_path(runs_dir, spec))
    sps_all = [v for r in rows if (v := _finite(r.get("perf/steps_per_s"))) is not None]
    if result.status != "ok" or not sps_all:
        error = result.error or "no perf/steps_per_s logged"
        return Measurement(
            **common,
            status="failed",
            error=error,
            steps_per_s=None,
            final_return=None,
            kl_stopped_rate=None,
            overhead_s=None,
        )
    sps = statistics.median(sps_all[1:] if len(sps_all) > 1 else sps_all)
    returns = [v for r in rows if (v := _finite(r.get("eval/return"))) is not None]
    kl = [v for r in rows if (v := _finite(r.get("ppo/kl_stopped"))) is not None]
    diag: dict[str, float | None] = {}
    for r in rows:
        for k, v in r.items():
            if k.startswith(_DIAG_PREFIXES):
                diag[k] = _finite(v)
    return Measurement(
        **common,
        status="ok",
        error=None,
        steps_per_s=sps,
        final_return=returns[-1] if returns else None,
        kl_stopped_rate=sum(kl) / len(kl) if kl else None,
        overhead_s=max(0.0, elapsed_s - result.env_steps_train / sps),
        diagnostics=diag,
    )


def _run_batch(
    batch: _Batch,
    *,
    s: CalibrationSettings,
    guard: BudgetGuard,
    pool_factory: PoolFactory,
    runner_cfg: RunnerConfig,
    clock: Callable[[], float],
    budget_usd: float,
) -> list[Measurement] | None:
    """Run one batch within `budget_usd` of pod time; None if the budget ran out first."""
    budget_s = budget_usd / guard.usd_per_s
    if not budget_s > 0:
        raise ValueError(f"batch budget must be > 0, got {budget_usd} USD")
    timeout_s = min(s.job_timeout_s, budget_s)
    pool = pool_factory(batch.n_workers, s.device, runner_cfg)
    pending = {
        (spec.arm_id(), spec.seed): (job, spec, lr) for job, spec, lr in batch.jobs
    }
    if len(pending) != len(batch.jobs):
        raise RuntimeError(f"batch {batch.key} has duplicate (arm_id, seed) jobs")
    done: dict[tuple[str, int], Measurement] = {}
    start_usd = guard.spent()
    try:
        submitted_at: dict[tuple[str, int], float] = {}
        for key, (job, spec, _) in pending.items():
            run_log_path(runner_cfg.runs_dir, spec).unlink(missing_ok=True)
            pool.submit(job, spec, timeout_s=timeout_s)
            submitted_at[key] = clock()
        # Each pass waits up to poll_s or until the budget ends, which closes the batch, so
        # this ends; the bound is a backstop.
        max_polls = math.ceil(budget_s / s.poll_s) + 100
        for _ in range(max_polls):
            if len(done) == len(pending):
                break
            left_s = (budget_usd - (guard.spent() - start_usd)) / guard.usd_per_s
            if left_s <= 0 or guard.remaining() <= 0:
                _log.warning(
                    "calibration batch %s stopped at its budget (%.2f USD); jobs killed",
                    batch.key,
                    budget_usd,
                )
                return None
            for result in pool.poll(timeout_s=min(s.poll_s, left_s)):
                key = (result.arm_id, result.seed)
                job, spec, lr = pending[key]
                elapsed = clock() - submitted_at[key]
                done[key] = _measure(
                    batch, job, spec, lr, result, elapsed, runner_cfg.runs_dir
                )
            guard.heartbeat()
        else:
            raise RuntimeError(
                f"calibration batch {batch.key} did not finish in {max_polls} polls"
            )
    finally:
        pool.close()
    return [done[k] for k in pending]


def calibrate(
    *,
    families: Sequence[str],
    settings: CalibrationSettings,
    guard: BudgetGuard,
    pool_factory: PoolFactory,
    runner_cfg: RunnerConfig,
    path: Path,
    clock: Callable[[], float],
    probe_families: Sequence[str] | None = None,
) -> CalibrationRecord:
    """Run (or resume) calibration for `families`; see the module docstring. `probe_families`
    (default: `families`) selects whose probes run; smoke mode adds `arm_dense` there to
    exercise the arm task's CPU contact check without calibrating that family."""
    if not families:
        raise ValueError("no families to calibrate")
    probe_families = families if probe_families is None else probe_families
    record = (
        CalibrationRecord.load(path) if path.exists() else CalibrationRecord(batches={})
    )
    record.capped = False
    for batch in _planned(families, settings, record, probe_families):
        if batch.key in record.batches:
            continue
        budget_usd = min(settings.cap_usd - record.spent_usd, guard.remaining())
        if budget_usd / guard.usd_per_s < settings.poll_s:
            record.capped = True
            _log.warning(
                "calibration stopped at its cap: %.2f of %.2f USD spent, %.2f USD left in the "
                "sweep budget; next batch %s",
                record.spent_usd,
                settings.cap_usd,
                guard.remaining(),
                batch.key,
            )
            break
        start = guard.spent()
        ms = _run_batch(
            batch,
            s=settings,
            guard=guard,
            pool_factory=pool_factory,
            runner_cfg=runner_cfg,
            clock=clock,
            budget_usd=budget_usd,
        )
        record.spent_usd += guard.spent() - start
        if ms is None:
            record.capped = True
            break
        record.batches[batch.key] = ms
        record.save(path)
        _log.info(
            "calibration %s: %s (%.2f USD so far)",
            batch.key,
            ", ".join(
                f"{m.label} lr={m.lr:g} N={m.num_envs} sps={m.steps_per_s} ret={m.final_return} "
                f"kl_stopped={m.kl_stopped_rate} {m.status}"
                for m in ms
            ),
            record.spent_usd,
        )
    record.save(path)
    return record


# --- choosing the sweep plan -------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class FamilyPlan:
    """How one task family runs in the sweep. `steps_per_s` is the aggregate over the plan's
    `n_workers` concurrent jobs; one job runs at `steps_per_s / n_workers`."""

    family: str
    ppo: PPOConfig
    num_envs: int
    rollout_len: int
    steps_per_s: float
    overhead_s: float
    calibrated: bool

    def __post_init__(self) -> None:
        if not (math.isfinite(self.steps_per_s) and self.steps_per_s > 0):
            raise ValueError(
                f"steps_per_s must be finite and > 0, got {self.steps_per_s}"
            )
        if not (math.isfinite(self.overhead_s) and self.overhead_s >= 0):
            raise ValueError(
                f"overhead_s must be finite and >= 0, got {self.overhead_s}"
            )


UNPROBED_FACTOR: Final = (
    3.0  # final-review I5: a method with no probe may run 3x uniform's wall
)


@dataclass(frozen=True, kw_only=True)
class Plan:
    n_workers: int
    families: Mapping[str, FamilyPlan]
    fallback: FamilyPlan
    # (family, method) -> measured job slowdown vs uniform (>= 1), from the probes
    method_factors: Mapping[tuple[str, str], float] = field(
        default_factory=dict[tuple[str, str], float]
    )

    def job_factor(
        self, family: str, method: str, *, unprobed: float = UNPROBED_FACTOR
    ) -> float:
        """How much longer than `uniform` a job of `method` in `family` is expected to take:
        the probe's measured slowdown when there is one, 1 for uniform, else `unprobed` (3 for
        timeouts; the sweep's budget estimate passes 1 and learns its own correction)."""
        if method == "uniform":
            return 1.0
        return self.method_factors.get((family, method), unprobed)

    def for_family(self, family: str) -> FamilyPlan:
        plan = self.families.get(family)
        return plan if plan is not None else replace(self.fallback, family=family)

    def for_task(self, task: str) -> FamilyPlan:
        return self.for_family(family_of(task))


def _pick_envs(
    grid: Mapping[tuple[int, int], float | None], n: int
) -> tuple[int, float] | None:
    at_n = {n_envs: v for (n_envs, w), v in grid.items() if w == n and v is not None}
    if not at_n:
        return None
    best = max(at_n.values())
    n_envs = min(e for e, v in at_n.items() if v >= NEAR_BEST * best)
    return n_envs, at_n[n_envs]


def _overhead(record: CalibrationRecord, family: str) -> float | None:
    vals = [m.overhead_s for m in record.config(family) if m.overhead_s is not None]
    return statistics.median(vals) if vals else None


def choose_plan(
    record: CalibrationRecord,
    *,
    arms_per_family: Mapping[str, int],
    settings: CalibrationSettings,
) -> Plan:
    """The sweep plan from calibration; see the module docstring."""
    grids = {f: record.throughput(f) for f in arms_per_family}
    measured = [f for f, g in grids.items() if any(v is not None for v in g.values())]
    if not measured:
        raise CalibrationError(
            "no usable steps/s measurement for any family; raise the calibration cap or check "
            "the calibration jobs' errors in calibration.json"
        )
    best: tuple[float, int, dict[str, tuple[int, float]]] | None = None
    for n in sorted(settings.workers_grid):
        picks = {f: _pick_envs(grids[f], n) for f in measured}
        if any(p is None for p in picks.values()):
            continue
        chosen = {f: p for f, p in picks.items() if p is not None}
        cost = sum(arms_per_family[f] / sps for f, (_, sps) in chosen.items())
        if best is None or cost < best[0]:
            best = (cost, n, chosen)
    if best is None:
        raise CalibrationError(
            f"no worker count in {settings.workers_grid} has usable measurements for every "
            f"measured family {measured}"
        )
    _, n_workers, chosen = best
    overheads = [o for f in measured if (o := _overhead(record, f)) is not None]
    max_overhead = max(overheads, default=0.0)
    families: dict[str, FamilyPlan] = {}
    lrs: list[float] = []
    for f, (n_envs, sps) in chosen.items():
        lr = chosen_lr(record, f) or START_CONFIG.lr
        lrs.append(lr)
        overhead = _overhead(record, f)
        families[f] = FamilyPlan(
            family=f,
            ppo=ppo_for(lr, n_envs, settings.rollout_len),
            num_envs=n_envs,
            rollout_len=settings.rollout_len,
            steps_per_s=sps,
            overhead_s=max_overhead if overhead is None else overhead,
            calibrated=True,
        )
    base = settings.base_num_envs
    fallback_lr = Counter(lrs).most_common(1)[0][0]
    fallback = FamilyPlan(
        family="fallback",
        ppo=ppo_for(fallback_lr, base, settings.rollout_len),
        num_envs=base,
        rollout_len=settings.rollout_len,
        steps_per_s=min(p.steps_per_s for p in families.values()),
        overhead_s=max_overhead,
        calibrated=False,
    )
    for f in arms_per_family:
        if f not in families:
            _log.warning(
                "family %s has no calibration measurements; using the fallback plan", f
            )
    return Plan(
        n_workers=n_workers,
        families=families,
        fallback=fallback,
        method_factors=method_factors(record, base_num_envs=base),
    )


def method_factors(
    record: CalibrationRecord, *, base_num_envs: int
) -> dict[tuple[str, str], float]:
    """(family, method) -> uniform's steps/s at (base N, 1 worker) / the probe's steps/s, at
    least 1 (a probe runs one job at base N, so the two are comparable). Probes that failed or
    lack a uniform reference are left out (they get `UNPROBED_FACTOR`)."""
    out: dict[tuple[str, str], float] = {}
    for fam, probes in PROBES.items():
        reference = record.throughput(fam).get((base_num_envs, 1))
        if reference is None:
            continue
        for task, method in probes:
            ms = record.batches.get(
                f"probe/{fam}/{_probe_arm(task, method).label()}", []
            )
            sps = [m.steps_per_s for m in ms if m.status == "ok" and m.steps_per_s]
            if sps:
                out[(fam, method)] = max(1.0, reference / min(sps))
    return out

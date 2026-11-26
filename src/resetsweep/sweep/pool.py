"""Worker pool: long-lived `spawn`ed workers, killed and respawned to stop a job, so any job
can be killed (Review Focus 1).

Design: each of the `n_workers` slots holds one worker process
(`multiprocessing.get_context("spawn")`; CUDA cannot fork) with a duplex `Pipe`. The worker
imports torch, warp, mujoco_warp and wandb and initialises Warp once (`_warm_up`), sends
("ready",), then loops: receive ("job", runner, spec), run it, send the result, reset
process-global state (`_between_jobs`), wait for the next. The constructor returns only once
every worker is ready, so no job pays the startup (about 52 s per job on the GPU pod when each
job had its own process, out of a 94 s per-job overhead). `ProcessPoolExecutor` is not used
because it cannot kill a running task. Here stopping a job (timeout, hang guard, `kill`) kills
its worker and starts a new one in the slot. Only the killed job's slot pays a new startup,
and jobs in other slots never notice. A worker that dies before sending ("ready",) 3 times
in a row in one slot raises RuntimeError instead of respawning forever.

State between jobs in one worker: `run` seeds torch's global RNG and builds its own
generators from `spec.seed`, so a job's result does not depend on what ran before it in the
worker (tested on CPU). After each job the worker runs `gc.collect()`, empties the CUDA cache,
and checks that no W&B sink is still open; a leaked one ends the worker, which is respawned.

Outcomes, all of which end as exactly one `RunResult` from `poll` (except `kill`):
- the runner returns: its result, checked to belong to the submitted spec;
- the runner raises: ``RunResult(status="failed", error=repr(exc))``; the worker stays up
  for the next job;
- the worker dies without a result (signal, OOM kill, segfault, `os._exit`): failed, with the
  exit code and signal name; the slot gets a new worker;
- the job passes its `timeout_s` (the absolute ceiling, from submit): the worker is killed,
  failed "timed out ...", and the slot gets a new worker;
- hang guard (final-review I5): with `heartbeat` and `hang_s`, the job is also killed, failed
  "no progress for ...", once the heartbeat file's mtime has not changed for `hang_s` of the
  pool's clock (the pool notes the time it sees each new mtime, so file and pool clocks are
  never compared; a missing file counts as no progress since submit). `train_job` touches
  the file on every metrics log (`HeartbeatSink`), once per PPO iteration and at each eval;
- `kill(job)`: the worker is killed, the slot gets a new worker, and no result is produced;
  the caller decides what to report (the sweep loop reports a failed result for a job whose
  arm is no longer live, and nothing for jobs it stops at the budget, so they are resubmitted
  on resume).

Every wait is bounded: `poll` waits at most `timeout_s` (`multiprocessing.connection.wait`,
checked against the Python 3.12 docs), joins wait at most `_JOIN_S`, startup at most
`_STARTUP_S`.

Process groups: `multiprocessing.Process` has no `start_new_session` (only `subprocess.Popen`
does, per the Python 3.12 docs), and spawn execs the child, so the parent cannot `setpgid` it.
So the worker calls `os.setsid()` first thing and leads its own group; killing a worker sends
SIGKILL to that whole group (`os.killpg`), which also takes down what its job started. A
worker killed before its `setsid` ran has no group yet; then only the process is killed, and
it has started nothing. Trade-off: the group now outlives a job, so it is NOT killed after a
normal job (that would kill the worker). A job's runner must close what it starts:
`train_job` closes its sinks, which finishes the W&B run. wandb's own service process
(wandb-core) stays up and is reused by the worker's next run; it, and anything a job leaked,
dies with the group when the worker is killed or the pool closes (`close` kills every
worker's group after asking idle workers to exit). Limit: a descendant that calls `setsid`
itself leaves the group.

`train_job` is the real runner: it trains one spec with a JSONL sink (`run_log_path`) and a
W&B sink, and closes both. Workers are daemonic, so they die with the sweep process.
"""

import functools
import gc
import logging
import math
import multiprocessing
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, Self

import torch
import wandb

from resetsweep.metrics import (
    JsonlSink,
    MetricsSink,
    MultiSink,
    WandbMode,
    WandbSink,
    live_wandb_runs,
)
from resetsweep.sweep.scheduler import Job
from resetsweep.train import RunResult, RunSpec, run

_log = logging.getLogger(__name__)

Runner = Callable[[RunSpec], RunResult]
_JOIN_S: Final = 10.0
_STARTUP_S: Final = (
    600.0  # a worker's imports + Warp init take about a minute on the pod
)
_STARTUP_FAILURES_MAX: Final = 3
_ERROR_MAX_CHARS: Final = 2000


class Pool(Protocol):
    """What the sweep loop and calibration need from a pool (the fake in tests implements it)."""

    def submit(
        self,
        job: Job,
        spec: RunSpec,
        *,
        timeout_s: float,
        heartbeat: Path | None = None,
        hang_s: float | None = None,
    ) -> None: ...

    def poll(self, timeout_s: float) -> list[RunResult]: ...

    def kill(self, job: Job) -> None: ...

    def running(self) -> list[Job]: ...

    def free_slots(self) -> int: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RunnerConfig:
    """Picklable settings for `train_job` in the worker process."""

    project: str
    group: str
    wandb_mode: WandbMode
    runs_dir: Path


PoolFactory = Callable[[int, str, RunnerConfig], Pool]


def run_log_path(runs_dir: Path, spec: RunSpec) -> Path:
    """Where a run's JSONL metrics go: ``runs_dir/<arm_id>-s<seed>.jsonl``."""
    return runs_dir / f"{spec.arm_id()}-s{spec.seed}.jsonl"


def heartbeat_path(runs_dir: Path, spec: RunSpec) -> Path:
    """The progress file `train_job` touches: ``runs_dir/<arm_id>-s<seed>.heartbeat``."""
    return runs_dir / f"{spec.arm_id()}-s{spec.seed}.heartbeat"


class HeartbeatSink:
    """Forwards to `inner` and touches `path` after every `log`: the job's progress signal."""

    def __init__(self, path: Path, inner: MetricsSink) -> None:
        self._path = path
        self._inner = inner
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, data: dict[str, float]) -> None:
        self._inner.log(step, data)
        self._path.touch()  # sets mtime to now (pathlib.Path.touch, exist_ok=True)

    def summary(self, data: dict[str, float | str]) -> None:
        self._inner.summary(data)

    def close(self) -> None:
        self._inner.close()


def failed_result(spec: RunSpec, error: str, *, wall_s: float) -> RunResult:
    """A failed RunResult for `spec`: NaN scores, no reward reached, no steps recorded."""
    if not error:
        raise ValueError("error text must not be empty")
    return RunResult(
        arm_id=spec.arm_id(),
        seed=spec.seed,
        status="failed",
        error=error[:_ERROR_MAX_CHARS],
        auc=math.nan,
        final_success=math.nan,
        reached_reward=False,
        steps_to_first_reward=None,
        ever_succeeded_share=math.nan,
        env_steps_train=0,
        env_steps_extra={},
        wall_s=wall_s,
    )


def train_job(cfg: RunnerConfig, spec: RunSpec) -> RunResult:
    """Train `spec` with JSONL + W&B sinks (the real runner, used through functools.partial)."""
    jsonl = JsonlSink(run_log_path(cfg.runs_dir, spec))
    try:
        wb = WandbSink(
            project=cfg.project,
            group=cfg.group,
            name=f"{spec.task}/{spec.method}/{spec.score or '-'}/{spec.arm_id()}-s{spec.seed}",
            config=spec.config(),
            mode=cfg.wandb_mode,
        )
    except BaseException:
        jsonl.close()
        raise
    sink = HeartbeatSink(heartbeat_path(cfg.runs_dir, spec), MultiSink(jsonl, wb))
    try:
        return run(spec, sink)
    finally:
        sink.close()


def _warm_up() -> None:
    """Pay a worker's one-off startup before its first job. Importing this module already
    imported torch, warp, mujoco_warp and wandb (through `resetsweep.train` and
    `resetsweep.metrics`); this initialises Warp (its CUDA driver and devices)."""
    import warp as wp

    wp.init()


def _between_jobs() -> None:
    """Reset process-global state a finished job may leave behind in a reused worker. RNGs
    need nothing: `run` seeds torch's global RNG and builds its own generators from
    `spec.seed`. Raises if the job left a W&B run open, which ends the worker (respawned)."""
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    if live_wandb_runs() != 0 or wandb.run is not None:
        raise RuntimeError(
            f"a job left {live_wandb_runs()} W&B sink(s) open (wandb.run={wandb.run}); "
            "ending this worker"
        )


def _worker_main(conn: Connection, warm_up: Callable[[], None]) -> None:
    """Worker process body: warm up, send ("ready",), then for each ("job", runner, spec)
    send ("ok", result) or ("err", repr(exc)), until ("exit",) or the pool's end closes."""
    os.setsid()  # lead a new process group, so the pool can kill everything the worker starts
    warm_up()
    conn.send(("ready",))
    # Not bounded by a count: each pass blocks on the next message, and the loop ends on
    # ("exit",), on EOF when the pool is gone, or when the pool kills the worker.
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return  # obs: allow LOG007 (the pool closed its end: nothing left to do)
        if message[0] == "exit":
            return
        _, runner, spec = message
        try:
            reply: tuple[str, object] = ("ok", runner(spec))
        except Exception as e:
            logging.getLogger(__name__).exception(
                "job %s seed %d failed", spec.arm_id(), spec.seed
            )
            reply = ("err", repr(e))
        conn.send(reply)
        _between_jobs()


def _exit_text(exitcode: int | None) -> str:
    if exitcode is None:
        return "exit code unknown"
    if exitcode < 0:
        try:
            name = signal.Signals(-exitcode).name
        except (
            ValueError
        ):  # obs: allow LOG007 (not a signal number: fall back to a label)
            name = "unknown signal"
        return f"exit code {exitcode} ({name})"
    return f"exit code {exitcode}"


def _kill_group(pid: int | None) -> None:
    """SIGKILL the process group led by `pid`, if it exists (see the module docstring)."""
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # obs: allow LOG007 (no such group: setsid not reached yet, or all exited)
    except PermissionError:
        # pid was recycled into a group we do not own: never signal it.
        _log.warning("process group %d is not ours; not killed", pid)


@dataclass
class _Slot:
    """One long-lived worker process and the parent's end of its duplex pipe."""

    process: BaseProcess
    conn: Connection
    ready: bool = False  # the worker sent ("ready",): its warm-up is done
    stopped: bool = False  # process gone and pipe closed; replaced by a new _Slot


@dataclass
class _Running:
    job: Job
    spec: RunSpec
    arm_id: str
    slot: int
    start_s: float
    timeout_s: float
    heartbeat: Path | None = None
    hang_s: float | None = None
    last_mtime_ns: int | None = None
    progress_s: float = (
        0.0  # pool-clock time of the last heartbeat change (submit to start)
    )


class WorkerPool:
    """`n_workers` concurrent jobs on `device`, on long-lived spawned workers (see the module
    docstring). The constructor returns once every worker is warm. Use as a context manager
    or call `close()`: it kills whatever still runs and ends the workers."""

    def __init__(
        self,
        *,
        n_workers: int,
        device: str,
        runner: Runner,
        clock: Callable[[], float] = time.monotonic,
        warm_up: Callable[[], None] = _warm_up,
    ) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self._n = n_workers
        self._device = device
        self._runner = runner
        self._clock = clock
        self._warm_up = warm_up
        self._ctx = multiprocessing.get_context("spawn")
        # Keyed by (arm_id, seed), which identifies a run: two calibration jobs can be equal
        # `Job`s that differ only in their spec's PPO config.
        self._running: dict[tuple[str, int], _Running] = {}
        self._slots: list[_Slot] = []
        self._startup_failures = [0] * n_workers
        self._closed = False
        try:
            for _ in range(n_workers):
                self._slots.append(self._spawn())
            self._wait_ready()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- public interface ----------------------------------------------------------------------

    def submit(
        self,
        job: Job,
        spec: RunSpec,
        *,
        timeout_s: float,
        heartbeat: Path | None = None,
        hang_s: float | None = None,
    ) -> None:
        """Send `job` to an idle worker. Raises if no worker is free or `spec` does not match.
        `timeout_s` is the absolute ceiling; `hang_s` (needs `heartbeat`) the hang guard."""
        if self._closed:
            raise RuntimeError("pool is closed")
        if spec.device != self._device:
            raise ValueError(
                f"spec.device {spec.device!r} != pool device {self._device!r}"
            )
        if job.seed != spec.seed or job.step_budget != spec.step_budget:
            raise ValueError(
                f"job seed {job.seed} != spec seed {spec.seed}"
                if job.seed != spec.seed
                else f"job step_budget {job.step_budget} != spec {spec.step_budget}"
            )
        if not (math.isfinite(timeout_s) and timeout_s > 0):
            raise ValueError(f"timeout_s must be finite and > 0, got {timeout_s}")
        if hang_s is not None and (
            heartbeat is None or not (math.isfinite(hang_s) and hang_s > 0)
        ):
            raise ValueError(
                f"hang_s needs a heartbeat path and must be finite and > 0, got {hang_s}"
            )
        key = (spec.arm_id(), spec.seed)
        if key in self._running:
            raise ValueError(f"run {key} is already running")
        if not self.free_slots():
            raise RuntimeError(f"no free worker ({self._n} running)")
        self._service_idle()
        busy = {r.slot for r in self._running.values()}
        # Any idle worker; a warm one first (a respawned one may still be warming up).
        i = min(
            (j for j in range(self._n) if j not in busy),
            key=lambda j: not self._slots[j].ready,
        )
        message = ("job", self._runner, spec)
        try:
            # `send` pickles before it writes: a pickling error sends nothing.
            self._slots[i].conn.send(message)
        except (BrokenPipeError, ConnectionResetError) as e:
            # The worker died after `_service_idle` looked; a new one gets the job.
            _log.warning("worker for slot %d died before taking a job", i, exc_info=e)
            self._replace(i)
            self._slots[i].conn.send(message)
        self._running[key] = _Running(
            job=job,
            spec=spec,
            arm_id=key[0],
            slot=i,
            start_s=self._clock(),
            timeout_s=timeout_s,
            heartbeat=heartbeat,
            hang_s=hang_s,
            last_mtime_ns=_mtime_ns(heartbeat),
            progress_s=self._clock(),
        )

    def poll(self, timeout_s: float) -> list[RunResult]:
        """Results of jobs that finished, waiting at most `timeout_s` for the first one."""
        if not (math.isfinite(timeout_s) and timeout_s >= 0):
            raise ValueError(f"timeout_s must be finite and >= 0, got {timeout_s}")
        self._service_idle()
        if not self._running:
            return []
        now = self._clock()
        until_deadline = min(self._deadline_s(r) - now for r in self._running.values())
        wait_s = max(0.0, min(timeout_s, until_deadline))
        # Sentinels too: a grandchild (e.g. a service the job started) can hold the pipe's
        # write end open after the worker died, so EOF alone would not show the death.
        slots = [self._slots[r.slot] for r in self._running.values()]
        wait(
            [s.conn for s in slots] + [s.process.sentinel for s in slots],
            timeout=wait_s,
        )
        out: list[RunResult] = []
        for key, r in list(self._running.items()):
            result = self._collect(r)
            if result is not None:
                del self._running[key]
                out.append(result)
        return out

    def kill(self, job: Job) -> None:
        """Kill `job`'s worker (and its process group) and start a new one in its slot; no
        result is produced for the job."""
        keys = [k for k, r in self._running.items() if r.job == job]
        if len(keys) != 1:
            raise ValueError(f"job {job} matches {len(keys)} running runs, expected 1")
        self._replace(self._running.pop(keys[0]).slot)

    def running(self) -> list[Job]:
        return [r.job for r in self._running.values()]

    def free_slots(self) -> int:
        free = self._n - len(self._running)
        assert 0 <= free <= self._n, free
        return free

    def close(self) -> None:
        """Kill every running job and end every worker; idempotent. Warm idle workers are
        asked to exit; the rest are killed. Every worker's group is killed at the end."""
        busy = {r.slot for r in self._running.values()}
        self._running.clear()
        asked: set[int] = set()
        for i, slot in enumerate(self._slots):
            if slot.stopped or i in busy or not slot.ready:
                continue
            try:
                slot.conn.send(("exit",))
                asked.add(i)
            except (BrokenPipeError, ConnectionResetError) as e:
                _log.warning("worker for slot %d already gone at close", i, exc_info=e)
        for i, slot in enumerate(self._slots):
            self._stop(slot, expect_exit=i in asked)
        self._closed = True

    # --- internals -----------------------------------------------------------------------------

    def _spawn(self) -> _Slot:
        parent, child = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=_worker_main, args=(child, self._warm_up), daemon=True
        )
        try:
            proc.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()  # the worker holds the only other end: EOF means it is gone
        return _Slot(process=proc, conn=parent)

    def _replace(self, i: int) -> None:
        """Kill slot `i`'s worker and group and start a new worker there (not waited for)."""
        self._stop(self._slots[i], expect_exit=False)
        self._slots[i] = self._spawn()

    def _wait_ready(self) -> None:
        """Wait (at most `_STARTUP_S`) until every worker sent ("ready",); `_service_idle`
        respawns workers that die on the way and raises at the startup-failure bound."""
        deadline = self._clock_s() + _STARTUP_S
        while not all(s.ready for s in self._slots):
            left = deadline - self._clock_s()
            if left <= 0:
                raise RuntimeError(
                    f"workers not ready after {_STARTUP_S:g} s "
                    f"({sum(s.ready for s in self._slots)} of {self._n})"
                )
            cold = [s for s in self._slots if not s.ready]
            wait(
                [s.conn for s in cold] + [s.process.sentinel for s in cold],
                timeout=min(left, 1.0),
            )
            self._service_idle()

    @staticmethod
    def _clock_s() -> float:
        # Startup is real time, not the injectable job clock (tests freeze that one).
        return time.monotonic()

    def _service_idle(self) -> None:
        """Read "ready" from idle workers and respawn idle workers that died."""
        busy = {r.slot for r in self._running.values()}
        for i in range(self._n):
            if i in busy:
                continue
            slot = self._slots[i]
            dead = self._drain(i, slot)
            if dead or slot.process.exitcode is not None:
                self._on_death(i, "idle")

    def _drain(self, i: int, slot: _Slot) -> bool:
        """Mark idle `slot` ready on ("ready",), the only message an idle worker sends;
        return True on EOF (the worker is gone)."""
        while slot.conn.poll():
            try:
                kind = slot.conn.recv()[0]
            except (EOFError, OSError):
                slot.process.join(_JOIN_S)  # bounded: the exit code names the cause
                return True  # obs: allow LOG007 (death is logged by _on_death)
            if kind != "ready":
                raise RuntimeError(f"idle worker in slot {i} sent {kind!r}")
            self._mark_ready(i, slot)
        return False

    def _mark_ready(self, i: int, slot: _Slot) -> None:
        if slot.ready:
            raise RuntimeError(f"worker in slot {i} sent 'ready' twice")
        slot.ready = True
        self._startup_failures[i] = 0

    def _on_death(self, i: int, what: str) -> None:
        """Slot `i`'s worker died on its own: respawn it, or raise at the startup bound."""
        slot = self._slots[i]
        reason = _exit_text(slot.process.exitcode)
        was_ready = slot.ready
        self._stop(slot, expect_exit=True)
        if not was_ready:
            self._startup_failures[i] += 1
            if self._startup_failures[i] >= _STARTUP_FAILURES_MAX:
                raise RuntimeError(
                    f"worker slot {i} died during startup {_STARTUP_FAILURES_MAX} times in "
                    f"a row (last: {reason}); check the worker's imports and Warp init"
                )
        _log.warning(
            "%s worker in slot %d died (%s, %s); respawning",
            what,
            i,
            reason,
            "after startup" if was_ready else "during startup",
        )
        self._slots[i] = self._spawn()

    def _deadline_s(self, r: _Running) -> float:
        """The earliest pool-clock time `r` may be killed, as known now."""
        ceiling = r.start_s + r.timeout_s
        if r.hang_s is None:
            return ceiling
        return min(ceiling, r.progress_s + r.hang_s)

    def _note_progress(self, r: _Running) -> None:
        mtime = _mtime_ns(r.heartbeat)
        if mtime is not None and mtime != r.last_mtime_ns:
            r.last_mtime_ns = mtime
            r.progress_s = self._clock()

    def _died(
        self, r: _Running, elapsed: float, exc: BaseException | None
    ) -> RunResult:
        slot = self._slots[r.slot]
        reason = _exit_text(slot.process.exitcode)
        _log.warning(
            "job %s seed %d: no result (%s)",
            r.arm_id,
            r.spec.seed,
            reason,
            exc_info=exc,
        )
        self._on_death(r.slot, "busy")
        return failed_result(
            r.spec, f"worker died without a result: {reason}", wall_s=elapsed
        )

    def _collect(self, r: _Running) -> RunResult | None:
        """The result of `r` if it finished, timed out or died; None while it still runs."""
        slot = self._slots[r.slot]
        elapsed = self._clock() - r.start_s
        # Exit first, then data: a result sent before exiting is already in the pipe.
        exited = slot.process.exitcode is not None
        while slot.conn.poll():
            try:
                kind, *payload = slot.conn.recv()
            except (EOFError, OSError) as e:
                # The write end closed with nothing (or half a message) in it: the worker died.
                slot.process.join(_JOIN_S)
                return self._died(r, elapsed, e)
            if kind == "ready":
                self._mark_ready(r.slot, slot)
                continue
            return self._checked(r, kind, payload[0] if payload else None, elapsed)
        if exited:
            return self._died(r, elapsed, None)
        if elapsed >= r.timeout_s:
            self._replace(r.slot)
            _log.warning(
                "job %s seed %d timed out after %s s",
                r.arm_id,
                r.spec.seed,
                r.timeout_s,
            )
            return failed_result(
                r.spec,
                f"timed out after {r.timeout_s} s; worker killed",
                wall_s=elapsed,
            )
        if r.hang_s is not None:
            self._note_progress(r)
            stalled = self._clock() - r.progress_s
            if stalled >= r.hang_s:
                self._replace(r.slot)
                _log.warning(
                    "job %s seed %d: no progress for %.0f s (hang guard %s s); killed",
                    r.arm_id,
                    r.spec.seed,
                    stalled,
                    r.hang_s,
                )
                return failed_result(
                    r.spec,
                    f"no progress for {r.hang_s:g} s (heartbeat {r.heartbeat}); worker killed",
                    wall_s=elapsed,
                )
        return None

    def _checked(
        self, r: _Running, kind: object, payload: object, elapsed: float
    ) -> RunResult:
        if kind == "err":
            return failed_result(r.spec, str(payload), wall_s=elapsed)
        if kind != "ok" or not isinstance(payload, RunResult):
            return failed_result(
                r.spec, f"worker sent an unknown message {kind!r}", wall_s=elapsed
            )
        if (payload.arm_id, payload.seed) != (r.arm_id, r.spec.seed):
            return failed_result(
                r.spec,
                f"runner returned a result for arm_id {payload.arm_id} seed {payload.seed}, "
                f"expected {r.arm_id} seed {r.spec.seed}",
                wall_s=elapsed,
            )
        return payload

    def _stop(self, slot: _Slot, *, expect_exit: bool) -> None:
        """Make sure `slot`'s worker and its process group are gone and its pipe is closed
        (bounded waits); idempotent. A worker expected to exit gets `_JOIN_S` first."""
        if slot.stopped:
            return
        if expect_exit:
            slot.process.join(_JOIN_S)
        _kill_group(slot.process.pid)
        if slot.process.is_alive():
            slot.process.kill()
            slot.process.join(_JOIN_S)
        if slot.process.is_alive():
            _log.error(
                "worker pid %s survived SIGKILL for %s s", slot.process.pid, _JOIN_S
            )
        slot.conn.close()
        if slot.process.exitcode is not None:
            slot.process.close()
        slot.stopped = True


def _mtime_ns(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None  # obs: allow LOG007 (no heartbeat yet: the job has not logged)


def default_pool_factory(n_workers: int, device: str, cfg: RunnerConfig) -> Pool:
    """A real `WorkerPool` running `train_job` with `cfg`."""
    return WorkerPool(
        n_workers=n_workers, device=device, runner=functools.partial(train_job, cfg)
    )


__all__ = [
    "HeartbeatSink",
    "Pool",
    "PoolFactory",
    "Runner",
    "RunnerConfig",
    "WorkerPool",
    "default_pool_factory",
    "failed_result",
    "heartbeat_path",
    "run_log_path",
    "train_job",
]

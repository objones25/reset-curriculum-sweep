"""Worker pool: one `spawn`ed process per job, so any job can be killed (Review Focus 1).

Design: `ProcessPoolExecutor` cannot kill a running task, and a long-lived worker that runs
many jobs can only be stopped by killing the worker, which then has to be respawned anyway.
So each job gets its own process (`multiprocessing.get_context("spawn")`; CUDA cannot fork)
and a one-way `Pipe` for its result. The spawn costs a few seconds of imports per job, small
next to jobs that train for minutes; calibration measures it as part of the per-job overhead.

Outcomes, all of which end as exactly one `RunResult` from `poll` (except `kill`):
- the runner returns: its result, checked to belong to the submitted spec;
- the runner raises: ``RunResult(status="failed", error=repr(exc))``;
- the process dies without a result (signal, OOM kill, segfault, `os._exit`): failed, with the
  exit code and signal name;
- the job passes its `timeout_s`: the process is killed, failed "timed out ...";
- `kill(job)`: the process is killed and no result is produced; the caller decides what to
  report (the sweep loop reports a failed result for a job whose arm is no longer live, and
  nothing for jobs it stops at the budget, so they are resubmitted on resume).

Every wait is bounded: `poll` waits at most `timeout_s` (`multiprocessing.connection.wait`,
checked against the Python 3.12 docs), joins wait at most `_JOIN_S`.

Process groups: `multiprocessing.Process` has no `start_new_session` (only `subprocess.Popen`
does, per the Python 3.12 docs), and spawn execs the child, so the parent cannot `setpgid` it.
So the child calls `os.setsid()` first thing and leads its own group; stopping a job sends
SIGKILL to that whole group (`os.killpg`), which also takes down what the job started (e.g.
wandb-core). A job killed before its `setsid` ran has no group yet; then only the process is
killed, and it has started nothing. The group is also killed after a normal exit, to reap any
leftover descendant. Limit: a descendant that calls `setsid` itself leaves the group.

`train_job` is the real runner: it trains one spec with a JSONL sink (`run_log_path`) and a
W&B sink, and closes both. Workers are daemonic, so they die with the sweep process.
"""

import functools
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

from resetsweep.metrics import JsonlSink, MultiSink, WandbMode, WandbSink
from resetsweep.sweep.scheduler import Job
from resetsweep.train import RunResult, RunSpec, run

_log = logging.getLogger(__name__)

Runner = Callable[[RunSpec], RunResult]
_JOIN_S: Final = 10.0
_ERROR_MAX_CHARS: Final = 2000


class Pool(Protocol):
    """What the sweep loop and calibration need from a pool (the fake in tests implements it)."""

    def submit(self, job: Job, spec: RunSpec, *, timeout_s: float) -> None: ...

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
    sink = MultiSink(jsonl, wb)
    try:
        return run(spec, sink)
    finally:
        sink.close()


def _child_main(runner: Runner, spec: RunSpec, conn: Connection) -> None:
    """Worker process body: send ("ok", result) or ("err", repr(exc)), then exit."""
    os.setsid()  # lead a new process group, so the pool can kill everything the job starts
    try:
        try:
            message: tuple[str, object] = ("ok", runner(spec))
        except Exception as e:
            logging.getLogger(__name__).exception("job %s seed %d failed", spec.arm_id(), spec.seed)
            message = ("err", repr(e))
        conn.send(message)
    finally:
        conn.close()


def _exit_text(exitcode: int | None) -> str:
    if exitcode is None:
        return "exit code unknown"
    if exitcode < 0:
        try:
            name = signal.Signals(-exitcode).name
        except ValueError:  # obs: allow LOG007 (not a signal number: fall back to a label)
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
class _Running:
    job: Job
    spec: RunSpec
    arm_id: str
    process: BaseProcess
    conn: Connection
    start_s: float
    timeout_s: float


class WorkerPool:
    """`n_workers` concurrent jobs on `device`, one spawned process each. Use as a context
    manager or call `close()`: it kills whatever still runs."""

    def __init__(
        self,
        *,
        n_workers: int,
        device: str,
        runner: Runner,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self._n = n_workers
        self._device = device
        self._runner = runner
        self._clock = clock
        self._ctx = multiprocessing.get_context("spawn")
        # Keyed by (arm_id, seed), which identifies a run: two calibration jobs can be equal
        # `Job`s that differ only in their spec's PPO config.
        self._running: dict[tuple[str, int], _Running] = {}
        self._closed = False

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

    def submit(self, job: Job, spec: RunSpec, *, timeout_s: float) -> None:
        """Start `job` in a new process. Raises if no worker is free or `spec` does not match."""
        if self._closed:
            raise RuntimeError("pool is closed")
        if spec.device != self._device:
            raise ValueError(f"spec.device {spec.device!r} != pool device {self._device!r}")
        if job.seed != spec.seed or job.step_budget != spec.step_budget:
            raise ValueError(
                f"job seed {job.seed} != spec seed {spec.seed}"
                if job.seed != spec.seed
                else f"job step_budget {job.step_budget} != spec {spec.step_budget}"
            )
        if not (math.isfinite(timeout_s) and timeout_s > 0):
            raise ValueError(f"timeout_s must be finite and > 0, got {timeout_s}")
        key = (spec.arm_id(), spec.seed)
        if key in self._running:
            raise ValueError(f"run {key} is already running")
        if not self.free_slots():
            raise RuntimeError(f"no free worker ({self._n} running)")
        recv, send = self._ctx.Pipe(duplex=False)
        proc = self._ctx.Process(
            target=_child_main, args=(self._runner, spec, send), daemon=True
        )
        try:
            proc.start()
        except BaseException:
            recv.close()
            send.close()
            raise
        send.close()  # the child holds the only write end: EOF means it is gone
        self._running[key] = _Running(
            job=job,
            spec=spec,
            arm_id=key[0],
            process=proc,
            conn=recv,
            start_s=self._clock(),
            timeout_s=timeout_s,
        )

    def poll(self, timeout_s: float) -> list[RunResult]:
        """Results of jobs that finished, waiting at most `timeout_s` for the first one."""
        if not (math.isfinite(timeout_s) and timeout_s >= 0):
            raise ValueError(f"timeout_s must be finite and >= 0, got {timeout_s}")
        if not self._running:
            return []
        now = self._clock()
        until_deadline = min(r.start_s + r.timeout_s - now for r in self._running.values())
        wait_s = max(0.0, min(timeout_s, until_deadline))
        # Sentinels too: a grandchild (e.g. a service the job started) can hold the pipe's
        # write end open after the worker died, so EOF alone would not show the death.
        ready_on = [r.conn for r in self._running.values()]
        ready_on += [r.process.sentinel for r in self._running.values()]
        wait(ready_on, timeout=wait_s)
        out: list[RunResult] = []
        for key, r in list(self._running.items()):
            result = self._collect(r)
            if result is not None:
                del self._running[key]
                out.append(result)
        return out

    def kill(self, job: Job) -> None:
        """Kill `job`'s process; no result is produced for it."""
        keys = [k for k, r in self._running.items() if r.job == job]
        if len(keys) != 1:
            raise ValueError(f"job {job} matches {len(keys)} running runs, expected 1")
        self._stop(self._running.pop(keys[0]), expect_exit=False)

    def running(self) -> list[Job]:
        return [r.job for r in self._running.values()]

    def free_slots(self) -> int:
        free = self._n - len(self._running)
        assert 0 <= free <= self._n, free
        return free

    def close(self) -> None:
        """Kill every running job; idempotent."""
        for key in list(self._running):
            self._stop(self._running.pop(key), expect_exit=False)
        self._closed = True

    # --- internals -----------------------------------------------------------------------------

    def _collect(self, r: _Running) -> RunResult | None:
        """The result of `r` if it finished, timed out or died; None while it still runs."""
        elapsed = self._clock() - r.start_s
        # Exit first, then data: a result sent before exiting is already in the pipe.
        exited = r.process.exitcode is not None
        if not r.conn.poll() and exited:
            reason = _exit_text(r.process.exitcode)
            self._stop(r, expect_exit=True)
            _log.warning("job %s seed %d: no result (%s)", r.arm_id, r.spec.seed, reason)
            return failed_result(r.spec, f"worker died without a result: {reason}", wall_s=elapsed)
        if r.conn.poll():
            try:
                kind, payload = r.conn.recv()
            except (EOFError, OSError) as e:
                # The write end closed with nothing (or half a message) in it: the child died.
                r.process.join(_JOIN_S)
                reason = _exit_text(r.process.exitcode)
                self._stop(r, expect_exit=True)
                _log.warning("job %s seed %d: no result (%s)", r.arm_id, r.spec.seed, reason, exc_info=e)
                return failed_result(r.spec, f"worker died without a result: {reason}", wall_s=elapsed)
            self._stop(r, expect_exit=True)
            return self._checked(r, kind, payload, elapsed)
        if elapsed >= r.timeout_s:
            self._stop(r, expect_exit=False)
            _log.warning("job %s seed %d timed out after %s s", r.arm_id, r.spec.seed, r.timeout_s)
            return failed_result(
                r.spec, f"timed out after {r.timeout_s} s; worker killed", wall_s=elapsed
            )
        return None

    def _checked(self, r: _Running, kind: object, payload: object, elapsed: float) -> RunResult:
        if kind == "err":
            return failed_result(r.spec, str(payload), wall_s=elapsed)
        if kind != "ok" or not isinstance(payload, RunResult):
            return failed_result(r.spec, f"worker sent an unknown message {kind!r}", wall_s=elapsed)
        if (payload.arm_id, payload.seed) != (r.arm_id, r.spec.seed):
            return failed_result(
                r.spec,
                f"runner returned a result for arm_id {payload.arm_id} seed {payload.seed}, "
                f"expected {r.arm_id} seed {r.spec.seed}",
                wall_s=elapsed,
            )
        return payload

    def _stop(self, r: _Running, *, expect_exit: bool) -> None:
        """Make sure `r`'s process is gone and its pipe closed (bounded waits). A worker that
        sent its result gets `_JOIN_S` to exit on its own before it is killed."""
        if expect_exit:
            r.process.join(_JOIN_S)
        _kill_group(r.process.pid)
        if r.process.is_alive():
            r.process.kill()
            r.process.join(_JOIN_S)
        if r.process.is_alive():
            _log.error("worker pid %s survived SIGKILL for %s s", r.process.pid, _JOIN_S)
        r.conn.close()
        if r.process.exitcode is not None:
            r.process.close()


def default_pool_factory(n_workers: int, device: str, cfg: RunnerConfig) -> Pool:
    """A real `WorkerPool` running `train_job` with `cfg`."""
    return WorkerPool(n_workers=n_workers, device=device, runner=functools.partial(train_job, cfg))


__all__ = [
    "Pool",
    "PoolFactory",
    "Runner",
    "RunnerConfig",
    "WorkerPool",
    "default_pool_factory",
    "failed_result",
    "run_log_path",
    "train_job",
]

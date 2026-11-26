"""Worker pool: long-lived spawned workers, killed and respawned to stop a job (Review Focus
1: a crash is a failed result, and any job can be killed).

These tests spawn real processes on CPU. The job functions are module-level so the spawn
child can unpickle them; only the determinism tests train (a tiny CPU config).
"""

import dataclasses
import math
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from resetsweep.metrics import NullSink
from resetsweep.ppo.update import START_CONFIG
from resetsweep.sweep.arms import arm_space
from resetsweep.sweep.pool import (
    RunnerConfig,
    WorkerPool,
    failed_result,
    run_log_path,
)
from resetsweep.sweep.scheduler import Job
from resetsweep.train import RunResult, RunSpec, run

ARM = arm_space("rung0")[0]
POLL_LIMIT_S = 120.0  # a spawn child imports torch and warp: a few seconds each


def _spec(seed: int = 0) -> RunSpec:
    return RunSpec(
        task=ARM.task,
        method=ARM.method,
        score=None,
        seed=seed,
        step_budget=100,
        num_envs=8,
        rollout_len=8,
        ppo=dataclasses.replace(START_CONFIG, minibatch_size=16),
        eval_every_steps=50,
        device="cpu",
    )


def _job(seed: int = 0) -> Job:
    return Job(arm=ARM, seed=seed, rung=0, step_budget=100)


def _ok_result(spec: RunSpec) -> RunResult:
    return RunResult(
        arm_id=spec.arm_id(),
        seed=spec.seed,
        status="ok",
        error=None,
        auc=0.25,
        final_success=0.5,
        reached_reward=True,
        steps_to_first_reward=40,
        ever_succeeded_share=0.75,
        env_steps_train=spec.step_budget,
        env_steps_extra={"walk": 0},
        wall_s=0.01,
    )


def ok_runner(spec: RunSpec) -> RunResult:
    return _ok_result(spec)


def raising_runner(spec: RunSpec) -> RunResult:
    raise FloatingPointError(f"NaN loss in seed {spec.seed}")


def sigkill_runner(spec: RunSpec) -> RunResult:
    del spec
    os.kill(os.getpid(), signal.SIGKILL)  # what the OOM killer does
    raise AssertionError("unreachable")


def hanging_runner(spec: RunSpec) -> RunResult:
    del spec
    threading.Event().wait(600.0)  # a stuck solver; the pool must kill it
    raise AssertionError("not killed")


def orphaning_runner(spec: RunSpec) -> RunResult:
    """Dies, leaving a grandchild that holds the result pipe open for 20 s."""
    del spec
    if os.fork() == 0:
        threading.Event().wait(20.0)
        os._exit(0)
    os._exit(1)


def _grandchild_pid_file(seed: int) -> Path:
    return Path(tempfile.gettempdir()) / f"resetsweep-test-grandchild-{seed}.pid"


def grandchild_runner(spec: RunSpec) -> RunResult:
    """Starts a long-lived grandchild (like wandb-core), records its pid, then hangs."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    _grandchild_pid_file(spec.seed).write_text(str(proc.pid))
    threading.Event().wait(600.0)
    raise AssertionError("not killed")


def wrong_id_runner(spec: RunSpec) -> RunResult:
    return dataclasses.replace(_ok_result(spec), arm_id="0" * 16)


def _poll_until(pool: WorkerPool, n: int) -> list[RunResult]:
    """Poll until `n` results arrived; bounded by POLL_LIMIT_S."""
    out: list[RunResult] = []
    deadline = time.monotonic() + POLL_LIMIT_S
    while len(out) < n:
        if time.monotonic() > deadline:
            raise TimeoutError(f"only {len(out)} of {n} results in {POLL_LIMIT_S} s")
        out += pool.poll(timeout_s=1.0)
    return out


def test_worker_returns_runner_result() -> None:
    spec = _spec()
    with WorkerPool(n_workers=1, device="cpu", runner=ok_runner) as pool:
        pool.submit(_job(), spec, timeout_s=POLL_LIMIT_S)
        (result,) = _poll_until(pool, 1)

        assert pool.running() == []
    assert result == _ok_result(spec)


def test_worker_crash_returns_failed_result() -> None:
    spec = _spec(seed=3)
    with WorkerPool(n_workers=1, device="cpu", runner=raising_runner) as pool:
        pool.submit(_job(seed=3), spec, timeout_s=POLL_LIMIT_S)
        (result,) = _poll_until(pool, 1)

    assert (result.status, result.arm_id, result.seed) == ("failed", spec.arm_id(), 3)
    assert result.error == "FloatingPointError('NaN loss in seed 3')"
    assert math.isnan(result.auc)


def test_worker_killed_by_signal_returns_failed_result() -> None:
    spec = _spec()
    with WorkerPool(n_workers=1, device="cpu", runner=sigkill_runner) as pool:
        pool.submit(_job(), spec, timeout_s=POLL_LIMIT_S)
        (result,) = _poll_until(pool, 1)

    assert result.status == "failed"
    assert result.error == "worker died without a result: exit code -9 (SIGKILL)"


def test_worker_over_timeout_is_killed_and_failed() -> None:
    spec = _spec()
    with WorkerPool(n_workers=1, device="cpu", runner=hanging_runner) as pool:
        pool.submit(_job(), spec, timeout_s=0.5)
        (result,) = _poll_until(pool, 1)

    assert result.status == "failed"
    assert result.error == "timed out after 0.5 s; worker killed"


def test_kill_stops_worker_without_a_result() -> None:
    with WorkerPool(n_workers=1, device="cpu", runner=hanging_runner) as pool:
        pool.submit(_job(), _spec(), timeout_s=POLL_LIMIT_S)
        pool.kill(_job())
        later = pool.poll(timeout_s=0.1)

        assert (pool.running(), later, pool.free_slots()) == ([], [], 1)


def test_result_for_another_spec_is_failed() -> None:
    with WorkerPool(n_workers=1, device="cpu", runner=wrong_id_runner) as pool:
        pool.submit(_job(), _spec(), timeout_s=POLL_LIMIT_S)
        (result,) = _poll_until(pool, 1)

    assert result.status == "failed"
    assert result.error is not None and result.error.startswith(
        "runner returned a result for arm_id 0000000000000000 seed 0"
    )


def test_submit_rejects_when_full() -> None:
    pool = WorkerPool(n_workers=1, device="cpu", runner=hanging_runner)
    try:
        pool.submit(_job(0), _spec(0), timeout_s=POLL_LIMIT_S)
        with pytest.raises(RuntimeError, match="no free worker"):
            pool.submit(_job(1), _spec(1), timeout_s=POLL_LIMIT_S)
    finally:
        pool.close()


def test_submit_rejects_spec_for_another_device() -> None:
    with (
        WorkerPool(n_workers=1, device="cuda:0", runner=ok_runner) as pool,
        pytest.raises(ValueError, match="spec.device 'cpu' != pool device 'cuda:0'"),
    ):
        pool.submit(_job(), _spec(), timeout_s=1.0)


def test_submit_rejects_job_spec_mismatch() -> None:
    with (
        WorkerPool(n_workers=1, device="cpu", runner=ok_runner) as pool,
        pytest.raises(ValueError, match="job seed 1 != spec seed 0"),
    ):
        pool.submit(_job(seed=1), _spec(seed=0), timeout_s=1.0)


def test_constructor_rejects_zero_workers() -> None:
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        WorkerPool(n_workers=0, device="cpu", runner=ok_runner)


def test_failed_result_has_nan_scores_and_given_error() -> None:
    spec = _spec(seed=2)

    result = failed_result(spec, "boom", wall_s=1.5)

    assert (result.arm_id, result.seed, result.status, result.error, result.wall_s) == (
        spec.arm_id(),
        2,
        "failed",
        "boom",
        1.5,
    )
    assert math.isnan(result.final_success)


def test_run_log_path_names_arm_id_and_seed(tmp_path: Path) -> None:
    spec = _spec(seed=4)
    cfg = RunnerConfig(project="p", group="g", wandb_mode="disabled", runs_dir=tmp_path)

    assert run_log_path(cfg.runs_dir, spec) == tmp_path / f"{spec.arm_id()}-s4.jsonl"


def test_worker_death_is_seen_even_if_a_grandchild_holds_the_pipe() -> None:
    with WorkerPool(n_workers=1, device="cpu", runner=orphaning_runner) as pool:
        pool.submit(_job(), _spec(), timeout_s=POLL_LIMIT_S)
        start = time.monotonic()
        (result,) = _poll_until(pool, 1)
        waited = time.monotonic() - start

    assert result.error == "worker died without a result: exit code 1"
    assert waited < 15.0  # not the 20 s until the grandchild lets go of the pipe


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until(cond: Callable[[], bool], limit_s: float) -> bool:
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        if cond():
            return True
        threading.Event().wait(0.1)
    return cond()


def test_kill_also_kills_the_workers_grandchildren() -> None:
    seed = os.getpid()  # unique per test process: names the pid file
    pid_file = _grandchild_pid_file(seed)
    pid_file.unlink(missing_ok=True)
    with WorkerPool(n_workers=1, device="cpu", runner=grandchild_runner) as pool:
        pool.submit(_job(seed), _spec(seed), timeout_s=POLL_LIMIT_S)
        assert _wait_until(
            lambda: pid_file.exists() and pid_file.read_text() != "", POLL_LIMIT_S
        )
        grandchild = int(pid_file.read_text())

        pool.kill(_job(seed))

    pid_file.unlink(missing_ok=True)
    assert _wait_until(lambda: not _alive(grandchild), 10.0)


def _open_fds() -> int:
    return len(os.listdir("/dev/fd"))


def test_failed_start_leaks_no_pipe_and_no_slot() -> None:
    pool = WorkerPool(n_workers=1, device="cpu", runner=lambda spec: _ok_result(spec))
    before = _open_fds()

    with pytest.raises(AttributeError, match="Can.t get local object"):
        pool.submit(_job(), _spec(), timeout_s=1.0)

    assert (_open_fds(), pool.running(), pool.free_slots()) == (before, [], 1)
    pool.close()


# --- hang guard (final-review I5) -----------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _beat(path: Path, n: int) -> None:
    os.utime(path, ns=(n * 10**9, n * 10**9))  # strictly increasing mtimes, no sleep


def test_job_with_progress_is_not_killed_past_hang_s(tmp_path: Path) -> None:
    clock = _Clock()
    beat = tmp_path / "job.heartbeat"
    beat.touch()
    with WorkerPool(
        n_workers=1, device="cpu", runner=hanging_runner, clock=clock
    ) as pool:
        pool.submit(_job(), _spec(), timeout_s=10_000.0, heartbeat=beat, hang_s=100.0)
        for n, t in enumerate((50.0, 140.0, 230.0), start=1):
            clock.t = t
            _beat(beat, n)
            assert pool.poll(timeout_s=0.0) == []
        clock.t = 320.0  # 90 s since the last beat seen at 230
        assert pool.poll(timeout_s=0.0) == []
        assert pool.running() == [_job()]


def test_job_without_progress_for_hang_s_is_killed_and_failed(tmp_path: Path) -> None:
    clock = _Clock()
    beat = tmp_path / "job.heartbeat"
    with WorkerPool(
        n_workers=1, device="cpu", runner=hanging_runner, clock=clock
    ) as pool:
        pool.submit(_job(), _spec(), timeout_s=10_000.0, heartbeat=beat, hang_s=100.0)
        clock.t = 60.0
        beat.touch()  # first progress seen at 60
        assert pool.poll(timeout_s=0.0) == []
        clock.t = 161.0

        (result,) = pool.poll(timeout_s=0.0)

    assert result.status == "failed"
    assert result.error is not None and "no progress for 100" in result.error


def test_progress_does_not_lift_the_absolute_ceiling(tmp_path: Path) -> None:
    clock = _Clock()
    beat = tmp_path / "job.heartbeat"
    with WorkerPool(
        n_workers=1, device="cpu", runner=hanging_runner, clock=clock
    ) as pool:
        pool.submit(_job(), _spec(), timeout_s=150.0, heartbeat=beat, hang_s=100.0)
        clock.t = 80.0
        beat.touch()
        assert pool.poll(timeout_s=0.0) == []
        clock.t = 151.0
        _beat(beat, 2)

        (result,) = pool.poll(timeout_s=0.0)

    assert result.error is not None and "timed out after 150" in result.error


def test_hang_s_needs_a_heartbeat_path() -> None:
    with (
        WorkerPool(n_workers=1, device="cpu", runner=ok_runner) as pool,
        pytest.raises(ValueError, match="heartbeat"),
    ):
        pool.submit(_job(), _spec(), timeout_s=10.0, hang_s=5.0)


def test_heartbeat_sink_touches_the_file_on_every_log(tmp_path: Path) -> None:
    from resetsweep.metrics import JsonlSink
    from resetsweep.sweep.pool import HeartbeatSink, heartbeat_path

    spec = _spec()
    path = heartbeat_path(tmp_path, spec)
    sink = HeartbeatSink(path, JsonlSink(tmp_path / "m.jsonl"))
    assert path.name == f"{spec.arm_id()}-s{spec.seed}.heartbeat"
    assert not path.exists()

    sink.log(0, {"x": 1.0})
    first = path.stat().st_mtime_ns
    _beat(path, 1)  # set an old mtime, then a log must move it forward again
    sink.log(1, {"x": 2.0})
    sink.close()

    assert path.stat().st_mtime_ns > 10**9
    assert first > 0


# --- persistent workers -------------------------------------------------------------------------

SEED_RAISES, SEED_SIGKILL, SEED_HANGS, SEED_SLOW = 3, 4, 5, 6


def _pid_file(seed: int) -> Path:
    """Where `scripted_runner` records its worker's pid; named by the test process's pid."""
    return (
        Path(tempfile.gettempdir())
        / f"resetsweep-test-worker-{os.getppid()}-{seed}.pid"
    )


def _test_pid_file(seed: int) -> Path:
    return (
        Path(tempfile.gettempdir()) / f"resetsweep-test-worker-{os.getpid()}-{seed}.pid"
    )


def scripted_runner(spec: RunSpec) -> RunResult:
    """Behaves by seed; an ok result carries its worker's pid in `env_steps_extra`."""
    if spec.seed == SEED_RAISES:
        raise FloatingPointError(f"NaN loss in seed {spec.seed}")
    if spec.seed == SEED_SIGKILL:
        os.kill(os.getpid(), signal.SIGKILL)
    if spec.seed == SEED_HANGS:
        _pid_file(spec.seed).write_text(str(os.getpid()))
        threading.Event().wait(600.0)
        raise AssertionError("not killed")
    if spec.seed == SEED_SLOW:
        threading.Event().wait(2.0)
    return dataclasses.replace(_ok_result(spec), env_steps_extra={"pid": os.getpid()})


def _pid(result: RunResult) -> int:
    assert result.status == "ok", result
    return result.env_steps_extra["pid"]


def _run_one(
    pool: WorkerPool, seed: int, *, timeout_s: float = POLL_LIMIT_S
) -> RunResult:
    pool.submit(_job(seed), _spec(seed), timeout_s=timeout_s)
    (result,) = _poll_until(pool, 1)
    return result


def test_consecutive_jobs_on_one_slot_reuse_the_worker_process() -> None:
    with WorkerPool(n_workers=1, device="cpu", runner=scripted_runner) as pool:
        first = _pid(_run_one(pool, 0))
        second = _pid(_run_one(pool, 1))

    assert first == second != os.getpid()


def test_failed_runner_keeps_the_worker_for_the_next_job() -> None:
    with WorkerPool(n_workers=1, device="cpu", runner=scripted_runner) as pool:
        before = _pid(_run_one(pool, 0))
        failed = _run_one(pool, SEED_RAISES)
        after = _pid(_run_one(pool, 1))

    assert failed.error == "FloatingPointError('NaN loss in seed 3')"
    assert before == after


def test_worker_crash_fails_the_job_and_respawns_the_slot() -> None:
    with WorkerPool(n_workers=1, device="cpu", runner=scripted_runner) as pool:
        before = _pid(_run_one(pool, 0))
        crashed = _run_one(pool, SEED_SIGKILL)
        after = _pid(_run_one(pool, 1))

        assert pool.free_slots() == 1
    assert crashed.error == "worker died without a result: exit code -9 (SIGKILL)"
    assert before != after


def test_timeout_kills_the_worker_and_respawns_the_slot() -> None:
    with WorkerPool(n_workers=1, device="cpu", runner=scripted_runner) as pool:
        before = _pid(_run_one(pool, 0))
        timed_out = _run_one(pool, SEED_HANGS, timeout_s=0.5)
        after = _pid(_run_one(pool, 1))

    assert timed_out.error == "timed out after 0.5 s; worker killed"
    assert before != after


def test_kill_respawns_the_slot_and_leaves_the_other_slots_job_running() -> None:
    pid_file = _test_pid_file(SEED_HANGS)
    pid_file.unlink(missing_ok=True)
    with WorkerPool(n_workers=2, device="cpu", runner=scripted_runner) as pool:
        pool.submit(_job(SEED_HANGS), _spec(SEED_HANGS), timeout_s=POLL_LIMIT_S)
        pool.submit(_job(SEED_SLOW), _spec(SEED_SLOW), timeout_s=POLL_LIMIT_S)
        assert _wait_until(
            lambda: pid_file.exists() and pid_file.read_text() != "", POLL_LIMIT_S
        )
        killed = int(pid_file.read_text())

        pool.kill(_job(SEED_HANGS))
        (slow,) = _poll_until(pool, 1)
        pool.submit(_job(0), _spec(0), timeout_s=POLL_LIMIT_S)
        pool.submit(_job(1), _spec(1), timeout_s=POLL_LIMIT_S)
        later = _poll_until(pool, 2)

    pid_file.unlink(missing_ok=True)
    assert _wait_until(lambda: not _alive(killed), 10.0)
    assert slow.seed == SEED_SLOW and _pid(slow) != killed
    assert sorted(r.seed for r in later) == [0, 1]
    later_pids = {_pid(r) for r in later}
    # one job per slot: the slow job's worker, reused, and the killed slot's new worker
    assert len(later_pids) == 2 and _pid(slow) in later_pids
    assert killed not in later_pids


def test_hang_guard_kills_the_worker_and_respawns_the_slot(tmp_path: Path) -> None:
    clock = _Clock()
    beat = tmp_path / "job.heartbeat"
    with WorkerPool(
        n_workers=1, device="cpu", runner=scripted_runner, clock=clock
    ) as pool:
        before = _pid(_run_one(pool, 0))
        pool.submit(
            _job(SEED_HANGS),
            _spec(SEED_HANGS),
            timeout_s=10_000.0,
            heartbeat=beat,
            hang_s=100.0,
        )
        clock.t = 101.0
        (hung,) = pool.poll(timeout_s=0.0)
        after = _pid(_run_one(pool, 1))

    assert hung.error is not None and "no progress for 100" in hung.error
    assert before != after


def dying_warm_up() -> None:
    os._exit(3)  # an import or Warp init that kills the worker before it is ready


def test_workers_that_die_during_startup_three_times_raise() -> None:
    with pytest.raises(RuntimeError, match="died during startup 3 times in a row"):
        WorkerPool(
            n_workers=1, device="cpu", runner=scripted_runner, warm_up=dying_warm_up
        )

    assert _wait_until(lambda: multiprocessing.active_children() == [], 10.0)


def _warm_log() -> Path:
    """Where `counting_warm_up` records each warm-up; named by the test process's pid."""
    return Path(tempfile.gettempdir()) / f"resetsweep-test-warm-{os.getppid()}.log"


def counting_warm_up() -> None:
    """Records its worker's pid after a pause standing in for torch/warp imports and Warp init."""
    threading.Event().wait(0.5)
    with _warm_log().open("a") as f:
        f.write(f"{os.getpid()}\n")


def test_the_pool_starts_warm_workers_so_jobs_pay_no_startup() -> None:
    log = Path(tempfile.gettempdir()) / f"resetsweep-test-warm-{os.getpid()}.log"
    log.unlink(missing_ok=True)
    with WorkerPool(
        n_workers=2, device="cpu", runner=scripted_runner, warm_up=counting_warm_up
    ) as pool:
        warmed_at_return = [int(p) for p in log.read_text().split()]
        job_pids = {_pid(_run_one(pool, seed)) for seed in (0, 1, 2, 7)}
        warmed_after_jobs = [int(p) for p in log.read_text().split()]
    log.unlink(missing_ok=True)

    # Both workers finished warming up before the constructor returned ...
    assert len(warmed_at_return) == 2 and len(set(warmed_at_return)) == 2
    # ... and the jobs ran on those workers, with no further warm-up.
    assert warmed_after_jobs == warmed_at_return
    assert job_pids <= set(warmed_at_return)


def test_close_leaves_no_live_children() -> None:
    pool = WorkerPool(n_workers=2, device="cpu", runner=scripted_runner)
    _run_one(pool, 0)
    pool.submit(_job(SEED_HANGS), _spec(SEED_HANGS), timeout_s=POLL_LIMIT_S)
    assert multiprocessing.active_children() != []

    pool.close()
    pool.close()  # idempotent

    assert multiprocessing.active_children() == []
    _test_pid_file(SEED_HANGS).unlink(missing_ok=True)


# --- state isolation between jobs in one worker ---------------------------------------------------

SEED_POISON = 99


class _RaisingSink(NullSink):
    def log(self, step: int, data: dict[str, float]) -> None:
        raise RuntimeError(f"sink failed at step {step}")


def training_runner(spec: RunSpec) -> RunResult:
    """Trains `spec` for real; seed SEED_POISON fails after one PPO iteration, having moved the
    global RNGs and built a sim, so the next job in the worker starts from dirty state."""
    sink = _RaisingSink() if spec.seed == SEED_POISON else NullSink()
    return run(spec, sink)


def _tiny_spec(seed: int) -> RunSpec:
    # `ball` needs a score; `random` draws from a seed derived from spec.seed
    return dataclasses.replace(_spec(seed), score="random", eval_n=4)


def _scores(result: RunResult) -> RunResult:
    assert result.status == "ok", result
    return dataclasses.replace(result, wall_s=0.0)


def test_same_spec_twice_in_one_worker_gives_identical_results() -> None:
    with WorkerPool(n_workers=1, device="cpu", runner=training_runner) as pool:
        results = []
        for seed in (0, 0, SEED_POISON, 0):
            pool.submit(_job(seed), _tiny_spec(seed), timeout_s=POLL_LIMIT_S)
            results += _poll_until(pool, 1)

    first, second, poisoned, after_failure = results
    assert poisoned.status == "failed"
    assert poisoned.error is not None and "sink failed" in poisoned.error
    assert _scores(first) == _scores(second) == _scores(after_failure)
    assert first.env_steps_train > 0

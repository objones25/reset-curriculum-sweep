"""Worker pool: one spawned process per job (Review Focus 1: a crash is a failed result).

These tests spawn real processes on CPU. The job functions are module-level so the spawn
child can unpickle them; none of them trains anything.
"""

import dataclasses
import math
import os
import signal
import threading
import time
from pathlib import Path

import pytest

from resetsweep.ppo.update import START_CONFIG
from resetsweep.sweep.arms import arm_space
from resetsweep.sweep.pool import (
    RunnerConfig,
    WorkerPool,
    failed_result,
    run_log_path,
)
from resetsweep.sweep.scheduler import Job
from resetsweep.train import RunResult, RunSpec

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

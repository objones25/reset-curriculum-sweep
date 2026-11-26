"""GPU checks the pod runs before the sweep (`uv run pytest -m gpu`; final-review I8).

- G2: a GPU smoke of the risky (method, score) paths.
- G3: CPU vs CUDA dynamics agreement for the substep-graph + applied-force path (maze_trap)
  and the contact-heavy arm (arm_sparse).
- G4: the worker pool on CUDA with two concurrent jobs.
- G6: one CUDA worker reused across jobs: the same job twice on one worker gives the same
  result, and a job that raises replaces the worker (recycle on error) for the next job.

Each GPU test has a CPU twin that runs the same harness here, so the harness itself is tested
on a machine without CUDA. G1 is in tests/tasks/test_sample_start_device.py, G5 in
tests/launch/test_bootstrap.py.
"""

import dataclasses
import functools
import logging
import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import Tensor

from resetsweep.metrics import NullSink
from resetsweep.sim.batch import BatchSim
from resetsweep.sim.dynamics import step_task
from resetsweep.sweep.arms import Arm, arm_space
from resetsweep.sweep.calibration import ppo_for
from resetsweep.sweep.pool import RunnerConfig, WorkerPool, train_job
from resetsweep.sweep.scheduler import Job, RunDefaults, spec_for
from resetsweep.tasks.base import TASKS, Task
from resetsweep.train import RunResult, RunSpec, run

TASKS.load("resetsweep.tasks")

# --- G2: GPU smoke --------------------------------------------------------------------------

G2_NUM_ENVS = 256
G2_ROLLOUT_LEN = 24
G2_ITERATIONS = 2
G2_EVAL_N = 16
# Every distinct (method, score) pair runs on pendulum_sparse and arm_sparse (arm: the CPU
# contact check on visited offers and resets). Trimmed for time: on maze_trap and
# maze_coinflip only one pair per method (its first rung-0 score). CPU timings on an M-series
# Mac were 2-3 s per pendulum/arm run and 7-9 s per maze run at these sizes; GPU timings are
# not measured yet (no CUDA here), expected similar because small runs are launch-bound:
# 70 x ~3 s + 20 x ~8 s, about 6 minutes on a 4090.
G2_ALL_PAIRS_TASKS = ("pendulum_sparse", "arm_sparse")
G2_ONE_PER_METHOD_TASKS = ("maze_trap", "maze_coinflip")


def g2_arms() -> list[Arm]:
    """The first rung-0 arm of each (task, method, score) for the all-pairs tasks, and of each
    (task, method) for the trimmed ones."""
    picked: dict[tuple[str, str, str | None], Arm] = {}
    for arm in arm_space("rung0"):
        if arm.task in G2_ALL_PAIRS_TASKS:
            picked.setdefault((arm.task, arm.method, arm.score), arm)
        elif arm.task in G2_ONE_PER_METHOD_TASKS:
            picked.setdefault((arm.task, arm.method, None), arm)
    return list(picked.values())


def _g2_spec(arm: Arm, device: str) -> RunSpec:
    budget = G2_ITERATIONS * G2_NUM_ENVS * G2_ROLLOUT_LEN
    defaults = RunDefaults(
        num_envs=G2_NUM_ENVS,
        rollout_len=G2_ROLLOUT_LEN,
        ppo=ppo_for(3e-4, G2_NUM_ENVS, G2_ROLLOUT_LEN),
        eval_every_steps=budget,
        eval_n=G2_EVAL_N,
        device=device,
    )
    return spec_for(arm, seed=0, step_budget=budget, defaults=defaults)


def test_g2_arm_set_covers_every_method_and_pair() -> None:
    arms = g2_arms()
    methods = {a.method for a in arm_space("rung0")}

    for task in G2_ALL_PAIRS_TASKS:
        want = {(a.method, a.score) for a in arm_space("rung0") if a.task == task}
        assert {(a.method, a.score) for a in arms if a.task == task} == want
    for task in G2_ONE_PER_METHOD_TASKS:
        assert sorted(a.method for a in arms if a.task == task) == sorted(methods)
    assert len(arms) == 2 * 35 + 2 * 10


def test_g2_harness_runs_one_arm_on_cpu() -> None:
    arm = next(a for a in g2_arms() if a.task == "maze_trap" and a.method == "uniform")

    result = run(_g2_spec(arm, "cpu"), NullSink())

    assert result.status == "ok"
    assert result.env_steps_train >= G2_ITERATIONS * G2_NUM_ENVS * G2_ROLLOUT_LEN


@pytest.mark.gpu
@pytest.mark.parametrize("arm", g2_arms(), ids=lambda a: a.label())
def test_g2_gpu_smoke(arm: Arm) -> None:
    result = run(_g2_spec(arm, "cuda:0"), NullSink())

    assert result.status == "ok"
    assert torch.isfinite(torch.tensor(result.auc))


# --- G3: CPU vs CUDA dynamics ----------------------------------------------------------------

G3_WORLDS = 32
G3_STEPS = 20
G3_RESET_EVERY = 5  # rows [0, 8) get a fresh state before every 5th step
G3_RESET_ROWS = 8
# Both devices run float32 MuJoCo Warp; reductions run in a different order on the GPU, so the
# two agree to rounding, not bit for bit. Partial resets every 5 steps bound how far rounding
# can grow. Tolerances (absolute): qpos 1e-3 (m or rad), qvel 1e-2 (m/s or rad/s). Unverified
# on a GPU until the pod runs it; a failure here is a reason to stop and look, not to loosen.
G3_QPOS_ATOL = 1e-3
G3_QVEL_ATOL = 1e-2


def _g3_inputs(
    task: Task,
) -> tuple[list[tuple[Tensor, Tensor, Tensor]], Tensor, Tensor]:
    gen = torch.Generator().manual_seed(0)
    first = task.sample_start(G3_WORLDS, gen)
    resets = [
        task.sample_start(G3_RESET_ROWS, gen) for _ in range(G3_STEPS // G3_RESET_EVERY)
    ]
    actions = torch.rand(G3_STEPS, G3_WORLDS, task.act_dim, generator=gen) * 2 - 1
    states = [(first.qpos, first.qvel, first.extra)] + [
        (r.qpos, r.qvel, r.extra) for r in resets
    ]
    return states, actions, first.extra


def g3_trajectory(task_name: str, device: str) -> tuple[Tensor, Tensor]:
    """(qpos, qvel) after each of G3_STEPS control steps on `device`, copied to the CPU, from
    fixed starts, fixed actions and fixed partial resets of rows [0, G3_RESET_ROWS)."""
    task: Task = TASKS.construct(task_name)  # pyright: ignore[reportAssignmentType] # registry is untyped
    states, actions, extra0 = _g3_inputs(task)
    sim = BatchSim(
        task.mjcf, nworld=G3_WORLDS, device=device, frame_skip=task.frame_skip
    )
    qpos0, qvel0, _ = states[0]
    sim.set_state(
        torch.arange(G3_WORLDS, device=device), qpos0.to(device), qvel0.to(device)
    )
    extra = extra0.to(device).clone()
    rows = torch.arange(G3_RESET_ROWS, device=device)
    qs: list[Tensor] = []
    vs: list[Tensor] = []
    for k in range(G3_STEPS):
        if k > 0 and k % G3_RESET_EVERY == 0:
            q, v, e = states[k // G3_RESET_EVERY]
            sim.set_state(rows, q.to(device), v.to(device))
            extra[:G3_RESET_ROWS] = e.to(device)
        step_task(task, sim, extra, actions[k].to(device))
        qs.append(sim.qpos.detach().to("cpu").clone())
        vs.append(sim.qvel.detach().to("cpu").clone())
    return torch.stack(qs), torch.stack(vs)


@pytest.mark.parametrize("task_name", ["maze_trap", "arm_sparse"])
def test_g3_harness_is_deterministic_on_cpu(task_name: str) -> None:
    a_q, a_v = g3_trajectory(task_name, "cpu")
    b_q, b_v = g3_trajectory(task_name, "cpu")

    assert a_q.shape == (G3_STEPS, G3_WORLDS, a_q.shape[2])
    assert torch.isfinite(a_q).all() and torch.isfinite(a_v).all()
    assert torch.equal(a_q, b_q) and torch.equal(a_v, b_v)
    assert not torch.equal(a_q[0], a_q[-1])  # the states actually moved


@pytest.mark.gpu
@pytest.mark.parametrize("task_name", ["maze_trap", "arm_sparse"])
def test_g3_cpu_and_cuda_dynamics_agree(task_name: str) -> None:
    cpu_q, cpu_v = g3_trajectory(task_name, "cpu")
    gpu_q, gpu_v = g3_trajectory(task_name, "cuda:0")

    torch.testing.assert_close(gpu_q, cpu_q, atol=G3_QPOS_ATOL, rtol=0.0)
    torch.testing.assert_close(gpu_v, cpu_v, atol=G3_QVEL_ATOL, rtol=0.0)


# --- G4: the worker pool on CUDA -------------------------------------------------------------

G4_POLL_LIMIT_S = 600.0


def _g4_jobs(device: str) -> list[tuple[Job, RunSpec]]:
    arm = next(
        a for a in arm_space("rung0") if a.label() == "pendulum_sparse/uniform/-"
    )
    defaults = RunDefaults(
        num_envs=64,
        rollout_len=8,
        ppo=ppo_for(3e-4, 64, 8),
        eval_every_steps=512,
        eval_n=8,
        device=device,
    )
    out = []
    for seed in (0, 1):
        spec = spec_for(arm, seed=seed, step_budget=1024, defaults=defaults)
        out.append((Job(arm=arm, seed=seed, rung=0, step_budget=1024), spec))
    return out


def g4_run_two_jobs(device: str, runs_dir: Path) -> list[RunResult]:
    cfg = RunnerConfig(
        project="g4", group="g4", wandb_mode="disabled", runs_dir=runs_dir
    )
    runner: Callable[[RunSpec], RunResult] = functools.partial(train_job, cfg)
    results: list[RunResult] = []
    with WorkerPool(n_workers=2, device=device, runner=runner) as pool:
        for job, spec in _g4_jobs(device):
            pool.submit(job, spec, timeout_s=G4_POLL_LIMIT_S)
        deadline = time.monotonic() + G4_POLL_LIMIT_S
        while len(results) < 2 and time.monotonic() < deadline:
            results += pool.poll(timeout_s=5.0)
    return results


@pytest.mark.slow
def test_g4_worker_pool_runs_two_jobs_on_cpu(tmp_path: Path) -> None:
    results = g4_run_two_jobs("cpu", tmp_path)

    assert sorted((r.seed, r.status) for r in results) == [(0, "ok"), (1, "ok")], (
        results
    )


@pytest.mark.gpu
def test_g4_worker_pool_runs_two_jobs_on_cuda(tmp_path: Path) -> None:
    results = g4_run_two_jobs("cuda:0", tmp_path)

    assert sorted((r.seed, r.status) for r in results) == [(0, "ok"), (1, "ok")], (
        results
    )


# --- G6: one CUDA worker reused across jobs ----------------------------------------------------

G6_RAISE_SEED = 99


class _G6RaisingSink(NullSink):
    def log(self, step: int, data: dict[str, float]) -> None:
        raise RuntimeError(f"injected failure at step {step} (G6)")


def g6_runner(cfg: RunnerConfig, spec: RunSpec) -> RunResult:
    """`train_job`, with the worker's pid in `env_steps_extra`. Seed G6_RAISE_SEED builds its
    sim on the device and runs until its first metrics log (step 0), then raises."""
    if spec.seed == G6_RAISE_SEED:
        return run(spec, _G6RaisingSink())
    result = train_job(cfg, spec)
    extra = {**result.env_steps_extra, "worker_pid": os.getpid()}
    return dataclasses.replace(result, env_steps_extra=extra)


def g6_reuse_sequence(
    device: str, runs_dir: Path, *, recycle_on_error: bool | None = None
) -> list[RunResult]:
    """On one worker, in order: job A (seed 0), job A again, a job that raises, job B (seed 1)."""
    cfg = RunnerConfig(
        project="g6", group="g6", wandb_mode="disabled", runs_dir=runs_dir
    )
    (job_a, spec_a), (job_b, spec_b) = _g4_jobs(device)
    job_x = dataclasses.replace(job_a, seed=G6_RAISE_SEED)
    spec_x = dataclasses.replace(spec_a, seed=G6_RAISE_SEED)
    results: list[RunResult] = []
    with WorkerPool(
        n_workers=1,
        device=device,
        runner=functools.partial(g6_runner, cfg),
        recycle_on_error=recycle_on_error,
    ) as pool:
        for job, spec in (
            (job_a, spec_a),
            (job_a, spec_a),
            (job_x, spec_x),
            (job_b, spec_b),
        ):
            pool.submit(job, spec, timeout_s=G4_POLL_LIMIT_S)
            deadline = time.monotonic() + G4_POLL_LIMIT_S
            got: list[RunResult] = []
            while not got and time.monotonic() < deadline:
                got = pool.poll(timeout_s=5.0)
            assert len(got) == 1, (
                f"no result for seed {spec.seed} in {G4_POLL_LIMIT_S} s"
            )
            results += got
    return results


def _g6_fields(result: RunResult) -> dict[str, object]:
    """Every field but `wall_s`, with NaN (no start state ever succeeded) spelled "nan" so
    that equal results compare equal."""
    fields = dataclasses.asdict(result)
    del fields["wall_s"]
    return {
        k: "nan" if isinstance(v, float) and math.isnan(v) else v
        for k, v in fields.items()
    }


def _g6_check(results: list[RunResult], caplog: pytest.LogCaptureFixture) -> None:
    first, again, raised, after = results
    assert (first.status, again.status, after.status) == ("ok", "ok", "ok"), results
    assert raised.status == "failed"
    assert raised.error is not None and "injected failure" in raised.error
    # Exact equality (no tolerance). The scores are success shares and step counts, not raw
    # floats, so CUDA rounding differences cannot move them by a little; at this 1024-step
    # budget they are 0 / NaN on CPU (no start state succeeds). A difference means the reused
    # worker ran the job differently, e.g. leaked state.
    assert _g6_fields(first) == _g6_fields(again)
    assert first.env_steps_extra["worker_pid"] == again.env_steps_extra["worker_pid"]
    assert after.env_steps_extra["worker_pid"] != first.env_steps_extra["worker_pid"]
    pool_warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == "resetsweep.sweep.pool" and r.levelno >= logging.WARNING
    ]
    assert pool_warnings == []  # no worker died on its own


@pytest.mark.slow
def test_g6_one_worker_reused_across_jobs_on_cpu(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="resetsweep.sweep.pool")
    # CPU keeps a worker after a failed job by default; turn recycling on to run the CUDA path.
    results = g6_reuse_sequence("cpu", tmp_path, recycle_on_error=True)

    _g6_check(results, caplog)


@pytest.mark.gpu
def test_g6_one_worker_reused_across_jobs_on_cuda(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="resetsweep.sweep.pool")
    results = g6_reuse_sequence(
        "cuda:0", tmp_path
    )  # recycle on error: the CUDA default

    _g6_check(results, caplog)

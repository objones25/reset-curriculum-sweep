"""All-pairs smoke test: every distinct rung 0 (method, score) pair trains two iterations.

A failure here is an integration bug between a method, a score, a task and `train.run`.
"""

import math
from typing import Any

import pytest

from resetsweep.metrics import NullSink
from resetsweep.ppo.update import PPOConfig
from resetsweep.sweep.arms import Arm, arm_space
from resetsweep.train import RunSpec, run

NUM_ENVS = 32
ROLLOUT_LEN = 8  # 256 train steps per iteration
ITERATIONS = 2
TASKS_UNDER_TEST = ("pendulum_sparse", "maze")

# Small overrides so every method works at 32 envs, keyed by method. They only add keys the
# arm leaves unset. Constraints they satisfy:
# - population methods (walk, ball, genetic) need num_envs // R >= 2: R = 4 gives K = 8;
#   update_every=1 makes the plan (rollouts, screens) run inside the two iterations;
# - buffer_scored and visited_count need capacity >= num_envs and ic_capacity <= capacity;
# - own_success keeps its default table (max_trajectories=16), which is small already.
METHOD_OVERRIDES: dict[str, dict[str, Any]] = {
    "visited": {"capacity": 64},
    "buffer_scored": {"capacity": 64, "ic_capacity": 32, "update_every": 1},
    "visited_count": {"capacity": 64, "ic_capacity": 32},
    "walk": {"R": 4, "update_every": 1, "walk_steps": 4},
    "ball": {"R": 4, "update_every": 1},
    "genetic": {"R": 4, "update_every": 1},
}

PPO = PPOConfig(
    gamma=0.99,
    lam=0.95,
    epochs=1,
    minibatch_size=64,
    lr=1e-3,
    ent_coef=0.0,
    vf_coef=1.0,
    target_kl=None,
)


class _PPOSink(NullSink):
    """A `NullSink` that keeps the PPO health metrics, so the test can check they are finite."""

    def __init__(self) -> None:
        self.ppo: dict[str, float] = {}

    def log(self, step: int, data: dict[str, float]) -> None:
        del step
        self.ppo.update({k: v for k, v in data.items() if k.startswith("ppo/")})


def distinct_pairs() -> list[Arm]:
    """One arm per distinct (method, method settings without p_task, score, score_kw),
    taken from the first sparse task; the test re-targets it at each task under test."""
    seen: dict[tuple[Any, ...], Arm] = {}
    for arm in arm_space("rung0"):
        settings = tuple(kv for kv in arm.method_kw if kv[0] != "p_task")
        seen.setdefault((arm.method, settings, arm.score, arm.score_kw), arm)
    return list(seen.values())


def _spec(arm: Arm, task: str) -> RunSpec:
    method_kw = {**METHOD_OVERRIDES.get(arm.method, {}), **arm.method_kwargs()}
    return RunSpec(
        task=task,
        task_kw={},
        method=arm.method,
        method_kw=method_kw,
        score=arm.score,
        score_kw=arm.score_kwargs(),
        seed=0,
        step_budget=NUM_ENVS * ROLLOUT_LEN * ITERATIONS,
        num_envs=NUM_ENVS,
        rollout_len=ROLLOUT_LEN,
        ppo=PPO,
        eval_every_steps=NUM_ENVS * ROLLOUT_LEN,
        eval_n=8,
        device="cpu",
    )


@pytest.mark.slow
@pytest.mark.parametrize("task", TASKS_UNDER_TEST)
def test_every_rung0_method_score_pair_runs_two_iterations(task: str) -> None:
    pairs = distinct_pairs()
    assert len(pairs) == 60  # 704 rung 0 arms collapse to 60 pairs; see test_arms
    failures: list[str] = []
    for arm in pairs:
        try:
            sink = _PPOSink()
            result = run(_spec(arm, task), sink)
        except Exception as exc:  # noqa: BLE001 # collect every failing pair, then report
            failures.append(f"{task}: {arm.label()}: {type(exc).__name__}: {exc}")
            continue
        if result.status != "ok" or not math.isfinite(result.final_success):
            failures.append(
                f"{task}: {arm.label()}: status={result.status} error={result.error}"
            )
        elif not sink.ppo or not all(math.isfinite(v) for v in sink.ppo.values()):
            failures.append(
                f"{task}: {arm.label()}: PPO metrics missing or not finite: {sink.ppo}"
            )
    assert failures == []

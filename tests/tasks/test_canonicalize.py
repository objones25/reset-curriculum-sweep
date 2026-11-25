import math

import pytest
import torch

from resetsweep.sim.batch import BatchSim
from resetsweep.sim.dynamics import step_task
from resetsweep.tasks.base import TASKS, Task, canonicalize, prepare_candidates
from resetsweep.types import StartStates

TASKS.load("resetsweep.tasks")

TWO_PI = 2 * math.pi
STEPS = 5

# Task -> qpos columns that are unlimited hinges (checked against each MJCF: no `limited`).
UNLIMITED_HINGES = {
    "pendulum_sparse": [0],
    "pendulum_dense": [0],
    "cartpole_swingup_sparse": [1],  # the pole; the cart slider is limited
    "cartpole_swingup_dense": [1],
    "acrobot_swingup_sparse": [0, 1],
    "acrobot_swingup_dense": [0, 1],
}
NO_UNLIMITED_HINGES = ["claw", "arm_sparse", "maze", "ramp_deceptive"]


def _task(name: str) -> Task:
    return TASKS.construct(name)  # pyright: ignore[reportReturnType] # registry is untyped


def _in_box_state(task: Task) -> StartStates:
    """Two in-box states with every unlimited hinge near +-pi (where swing-ups cross)."""
    base = task.canonical_start()
    qpos = base.qpos.repeat(2, 1)
    for col in UNLIMITED_HINGES[task.name]:
        qpos[0, col], qpos[1, col] = 2.9, -3.0
    qvel = torch.full_like(base.qvel.repeat(2, 1), 0.5)
    return StartStates(qpos, qvel, torch.zeros(2, task.extra_dim))


def _unwrapped(task: Task, states: StartStates) -> StartStates:
    """The same physical states with each unlimited hinge 2 pi * k away (k = +1, -2)."""
    qpos = states.qpos.clone()
    for col in UNLIMITED_HINGES[task.name]:
        qpos[0, col] += TWO_PI
        qpos[1, col] -= 2 * TWO_PI
    return StartStates(qpos, states.qvel.clone(), states.extra.clone())


def _rollout(task: Task, states: StartStates) -> torch.Tensor:
    sim = BatchSim(task.mjcf, nworld=states.n, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(states.n), states.qpos, states.qvel)
    action = torch.full((states.n, task.act_dim), 0.3)
    for _ in range(STEPS):
        step_task(task, sim, states.extra, action)
    after = StartStates(sim.qpos.clone(), sim.qvel.clone(), states.extra)
    wrapped = canonicalize(task, after)
    return torch.cat([wrapped.qpos, wrapped.qvel], dim=1)


@pytest.mark.parametrize("name", sorted(UNLIMITED_HINGES))
def test_canonicalize_wraps_unlimited_hinges_exactly(name: str) -> None:
    task = _task(name)
    in_box = _in_box_state(task)
    raw = _unwrapped(task, in_box)

    wrapped = canonicalize(task, raw)

    assert torch.allclose(wrapped.qpos, in_box.qpos, atol=1e-5)
    assert torch.equal(wrapped.qvel, raw.qvel)
    assert task.is_valid(wrapped).tolist() == [True, True]
    # Physically the same state: stepping either gives the same (wrapped) trajectory.
    assert torch.allclose(_rollout(task, raw), _rollout(task, wrapped), atol=1e-4)


@pytest.mark.parametrize("name", sorted(UNLIMITED_HINGES))
def test_canonicalize_leaves_other_columns_alone(name: str) -> None:
    task = _task(name)
    base = task.canonical_start()
    qpos = torch.full_like(base.qpos, 7.0)  # every column far outside [-pi, pi)
    states = StartStates(qpos, base.qvel, base.extra)

    wrapped = canonicalize(task, states)

    hinges = UNLIMITED_HINGES[name]
    others = [c for c in range(qpos.shape[1]) if c not in hinges]
    assert wrapped.qpos[0, hinges].tolist() == pytest.approx(
        [7.0 - TWO_PI] * len(hinges)
    )
    assert wrapped.qpos[0, others].tolist() == [7.0] * len(others)


@pytest.mark.parametrize("name", NO_UNLIMITED_HINGES)
def test_canonicalize_is_identity_without_unlimited_hinges(name: str) -> None:
    task = _task(name)
    base = task.canonical_start()
    states = StartStates(base.qpos + 7.0, base.qvel, base.extra)

    assert canonicalize(task, states) is states


def test_prepare_candidates_rejects_non_finite_rows() -> None:
    task = _task("pendulum_sparse")
    states = StartStates(
        torch.tensor([[3.0 + TWO_PI], [float("nan")]]),
        torch.zeros(2, 1),
        torch.zeros(2, 0),
    )

    wrapped, valid = prepare_candidates(task, states)

    assert valid.tolist() == [True, False]
    assert wrapped.qpos[0, 0].item() == pytest.approx(3.0, abs=1e-5)

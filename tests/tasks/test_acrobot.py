import math

import pytest
import torch
from controllers import SamplingPlanner, first_episode_success, run_first_episode
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import TASKS, Task
from resetsweep.types import StartStates

TASKS.load("resetsweep.tasks")

UPRIGHT = [math.pi, 0.0]  # shoulder straight up, elbow straight
NAMES = ["acrobot_swingup_sparse", "acrobot_swingup_dense"]


def _task(name: str) -> Task:
    return TASKS.construct(name)  # pyright: ignore[reportReturnType] # registry is untyped


def _sim_at(task: Task, q: list[list[float]]) -> BatchSim:
    n = len(q)
    sim = BatchSim(task.mjcf, nworld=n, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(n), torch.tensor(q), torch.zeros(n, 2))
    return sim


def test_acrobot_sparse_zero_from_hanging_start() -> None:
    task = _task("acrobot_swingup_sparse")
    starts = task.sample_start(32, torch.Generator().manual_seed(0))
    sim = BatchSim(task.mjcf, nworld=32, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(32), starts.qpos, starts.qvel)

    reward = task.reward(sim, starts.extra, torch.zeros(32, 1))

    assert reward.tolist() == [0.0] * 32
    assert not bool(task.success_now(sim, starts.extra).any())


def _outcome_from(task: Task, qpos: list[float]) -> Tensor:
    """Start at `qpos` with zero velocity and zero action; return the episode's success flags."""
    start = StartStates(torch.tensor([qpos]), torch.zeros(1, 2), torch.zeros(1, 0))
    return run_first_episode(task, lambda _env, _obs: torch.zeros(1, 1), start).success


def test_acrobot_success_when_tip_above_target_and_held() -> None:
    task = _task("acrobot_swingup_sparse")

    # Exactly upright at rest: the unstable balance outlasts hold_steps.
    assert _outcome_from(task, UPRIGHT).tolist() == [True]


def test_acrobot_tip_above_target_but_falling_is_not_held() -> None:
    task = _task("acrobot_swingup_sparse")

    # Above the target height at t=0, but it drops below within a few control steps.
    assert _outcome_from(task, [math.pi - 0.25, 0.0]).tolist() == [False]


def test_acrobot_success_needs_tip_above_target_height() -> None:
    task = _task("acrobot_swingup_sparse")
    sim = _sim_at(
        task,
        [
            UPRIGHT,
            [0.0, 0.0],
            [math.pi / 2, 0.0],
            [math.pi, math.pi],
            [math.pi - 0.2, 0.0],
        ],
    )

    success = task.success_now(sim, torch.zeros(5, 0))

    assert success.tolist() == [True, False, False, False, True]


@pytest.mark.parametrize("name", NAMES)
def test_acrobot_action_cost_is_point_zero_one_times_squared_action(name: str) -> None:
    task = _task(name)
    sim = _sim_at(task, [[0.0, 0.0], [0.0, 0.0]])
    extra = torch.zeros(2, 0)

    free = task.reward(sim, extra, torch.zeros(2, 1))
    paid = task.reward(sim, extra, torch.tensor([[1.0], [0.5]]))

    assert (free - paid).tolist() == pytest.approx([0.01, 0.0025])


def test_acrobot_dense_and_sparse_share_one_mjcf_and_name_the_dense_sibling() -> None:
    sparse, dense = (_task(n) for n in NAMES)

    assert sparse.mjcf == dense.mjcf
    assert sparse.dense_sibling == "acrobot_swingup_dense"
    assert dense.dense_sibling == "acrobot_swingup_dense"


def test_acrobot_dense_reward_is_normalised_tip_height() -> None:
    task = _task("acrobot_swingup_dense")
    sim = _sim_at(task, [[0.0, 0.0], [math.pi / 2, 0.0], UPRIGHT])

    reward = task.reward(sim, torch.zeros(3, 0), torch.zeros(3, 1))

    assert reward.tolist() == pytest.approx([0.0, 0.5, 1.0], abs=1e-4)


def test_acrobot_control_step_is_0_05_s() -> None:
    import mujoco

    task = _task("acrobot_swingup_sparse")
    model = mujoco.MjModel.from_xml_string(task.mjcf)  # pyright: ignore[reportAttributeAccessIssue]

    assert task.frame_skip * model.opt.timestep == pytest.approx(0.05)


def test_acrobot_constant_torque_cannot_reach_the_target() -> None:
    # A swing-up must be necessary: a fixed full torque from hanging never gets there.
    task = _task("acrobot_swingup_sparse")
    sim = _sim_at(task, [[0.0, 0.0], [0.0, 0.0]])
    ctrl = torch.tensor([[100.0], [-100.0]])  # far past the limit: the clip is tested
    reached = torch.zeros(2, dtype=torch.bool)

    for _ in range(task.max_steps):
        sim.step(ctrl)
        reached |= task.success_now(sim, torch.zeros(2, 0))

    assert reached.tolist() == [False, False]


def test_acrobot_observation_is_cos_sin_of_both_joints_then_velocities() -> None:
    task = _task("acrobot_swingup_sparse")
    sim = BatchSim(task.mjcf, nworld=1, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(
        torch.arange(1),
        torch.tensor([[math.pi / 2, 0.0]]),
        torch.tensor([[1.5, -2.0]]),
    )

    obs = task.observe(sim, torch.zeros(1, 0))

    expected = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.5, -2.0]])
    assert torch.allclose(obs, expected, atol=1e-5)


def test_acrobot_never_terminates_early() -> None:
    task = _task("acrobot_swingup_sparse")
    sim = _sim_at(task, [[0.0, 0.0], UPRIGHT])

    assert task.terminated(sim, torch.zeros(2, 0)).tolist() == [False, False]


@pytest.mark.slow
def test_acrobot_sparse_solvable_by_scripted_controller() -> None:
    # A sampling MPC on the dense sibling's reward must reach held success from sample_start.
    n = 4
    task = _task("acrobot_swingup_sparse")
    dense = _task(task.dense_sibling)
    planner = SamplingPlanner(dense, num_envs=n, seed=0)

    success = first_episode_success(
        task,
        lambda env, _obs: planner(*env.sim.get_state(torch.arange(n)), env.extra),
        n,
        seed=0,
    )

    assert int(success.sum()) >= 3

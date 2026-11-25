import math

import pytest
import torch
from controllers import CartpoleSwingup, first_episode_success
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import TASKS, Task
from resetsweep.types import StartStates

TASKS.load("resetsweep.tasks")

UPRIGHT_RAD = math.pi
NAMES = ["cartpole_swingup_sparse", "cartpole_swingup_dense"]


def _task(name: str) -> Task:
    return TASKS.construct(name)  # pyright: ignore[reportReturnType] # registry is untyped


def _sim_at(task: Task, x: list[float], theta: list[float]) -> BatchSim:
    n = len(x)
    sim = BatchSim(task.mjcf, nworld=n, device="cpu", frame_skip=task.frame_skip)
    q = torch.tensor([x, theta]).T.contiguous()
    sim.set_state(torch.arange(n), q, torch.zeros(n, 2))
    return sim


def test_cartpole_sparse_zero_from_hanging_start() -> None:
    task = _task("cartpole_swingup_sparse")
    starts = task.sample_start(32, torch.Generator().manual_seed(0))
    sim = BatchSim(task.mjcf, nworld=32, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(32), starts.qpos, starts.qvel)

    reward = task.reward(sim, starts.extra, torch.zeros(32, 1))

    assert reward.tolist() == [0.0] * 32
    assert not bool(task.success_now(sim, starts.extra).any())


def _upright_env_outcome(task: Task, theta_offset: float) -> Tensor:
    """Start upright (cart at the centre) with zero action; return the episode's success flags."""
    from controllers import run_first_episode

    start = StartStates(
        torch.tensor([[0.0, UPRIGHT_RAD + theta_offset]]),
        torch.zeros(1, 2),
        torch.zeros(1, 0),
    )
    return run_first_episode(task, lambda _env, _obs: torch.zeros(1, 1), start).success


def test_cartpole_success_when_set_upright_and_held() -> None:
    task = _task("cartpole_swingup_sparse")

    # Exactly upright with zero velocity, the pole needs far longer than hold_steps to fall.
    success = _upright_env_outcome(task, 0.0)

    assert success.tolist() == [True]


def test_cartpole_upright_but_falling_is_not_held() -> None:
    task = _task("cartpole_swingup_sparse")

    # Inside the tolerance band at t=0 but falling out of it in about 5 control steps.
    success = _upright_env_outcome(task, -0.1)

    assert success.tolist() == [False]


def test_cartpole_success_needs_upright_and_cart_near_centre() -> None:
    task = _task("cartpole_swingup_sparse")
    sim = _sim_at(
        task,
        x=[0.0, 0.0, 1.2, -1.2, 0.0],
        theta=[UPRIGHT_RAD, UPRIGHT_RAD + 0.1, UPRIGHT_RAD, UPRIGHT_RAD, 0.0],
    )

    success = task.success_now(sim, torch.zeros(5, 0))

    assert success.tolist() == [True, True, False, False, False]


def test_cartpole_success_wraps_angle_past_two_pi() -> None:
    task = _task("cartpole_swingup_sparse")
    sim = _sim_at(
        task,
        x=[0.0, 0.0],
        theta=[UPRIGHT_RAD + 2 * math.pi, UPRIGHT_RAD - 2 * math.pi],
    )

    assert task.success_now(sim, torch.zeros(2, 0)).tolist() == [True, True]


@pytest.mark.parametrize("name", NAMES)
def test_cartpole_action_cost_is_point_zero_one_times_squared_action(
    name: str,
) -> None:
    task = _task(name)
    sim = _sim_at(task, x=[0.0, 0.0], theta=[0.0, 0.0])
    extra = torch.zeros(2, 0)

    free = task.reward(sim, extra, torch.zeros(2, 1))
    paid = task.reward(sim, extra, torch.tensor([[1.0], [0.5]]))

    assert (free - paid).tolist() == pytest.approx([0.01, 0.0025])


def test_cartpole_dense_and_sparse_share_one_mjcf_and_name_the_dense_sibling() -> None:
    sparse, dense = (_task(n) for n in NAMES)

    assert sparse.mjcf == dense.mjcf
    assert sparse.dense_sibling == "cartpole_swingup_dense"
    assert dense.dense_sibling == "cartpole_swingup_dense"


def test_cartpole_dense_reward_increases_with_height() -> None:
    task = _task("cartpole_swingup_dense")
    thetas = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, UPRIGHT_RAD]
    sim = _sim_at(task, x=[0.0] * 5, theta=thetas)

    reward = task.reward(sim, torch.zeros(5, 0), torch.zeros(5, 1))

    assert reward.tolist() == pytest.approx(
        [(1 - math.cos(t)) / 2 for t in thetas], abs=1e-5
    )


def test_cartpole_dense_reward_penalises_cart_off_centre() -> None:
    task = _task("cartpole_swingup_dense")
    sim = _sim_at(task, x=[0.0, 1.0], theta=[UPRIGHT_RAD, UPRIGHT_RAD])

    reward = task.reward(sim, torch.zeros(2, 0), torch.zeros(2, 1))

    assert float(reward[0]) > float(reward[1])


def test_cartpole_control_step_is_0_05_s() -> None:
    import mujoco

    task = _task("cartpole_swingup_sparse")
    model = mujoco.MjModel.from_xml_string(task.mjcf)  # pyright: ignore[reportAttributeAccessIssue]

    assert task.frame_skip * model.opt.timestep == pytest.approx(0.05)


def test_cartpole_full_push_for_one_second_cannot_lift_the_pole() -> None:
    # A swing-up must be necessary: pushing flat out one way never gets near horizontal.
    task = _task("cartpole_swingup_sparse")
    sim = _sim_at(task, x=[0.0, 0.0], theta=[0.0, 0.0])
    ctrl = torch.tensor([[100.0], [-100.0]])  # far past the limit: the clip is tested
    peak = torch.zeros(2)

    for _ in range(round(1.0 / 0.05)):
        sim.step(ctrl)
        peak = torch.maximum(peak, sim.qpos[:, 1].abs())

    assert float(peak.max()) < math.pi / 2


def test_cartpole_actions_beyond_the_limit_are_clipped() -> None:
    task = _task("cartpole_swingup_sparse")
    sim = _sim_at(task, x=[0.0, 0.0], theta=[0.0, 0.0])

    for _ in range(10):
        sim.step(torch.tensor([[1.0], [50.0]]))

    assert float(sim.qpos[0, 0]) == pytest.approx(float(sim.qpos[1, 0]), abs=1e-5)


def test_cartpole_observation_is_x_cos_sin_xdot_thetadot() -> None:
    task = _task("cartpole_swingup_sparse")
    sim = BatchSim(task.mjcf, nworld=1, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(
        torch.arange(1),
        torch.tensor([[0.3, math.pi / 2]]),
        torch.tensor([[0.5, -2.0]]),
    )

    obs = task.observe(sim, torch.zeros(1, 0))

    expected = torch.tensor([[0.3, math.cos(math.pi / 2), 1.0, 0.5, -2.0]])
    assert torch.allclose(obs, expected, atol=1e-5)


def test_cartpole_sample_start_is_hanging_at_the_centre_with_small_noise() -> None:
    task = _task("cartpole_swingup_sparse")

    s = task.sample_start(256, torch.Generator().manual_seed(0))

    assert float(s.qpos.abs().max()) <= 0.1
    assert float(s.qvel.abs().max()) <= 0.1
    assert float(s.qpos.std()) > 0.01


def test_cartpole_start_box_covers_cart_travel_and_every_angle() -> None:
    task = _task("cartpole_swingup_sparse")
    low, high = task.start_box.low.tolist(), task.start_box.high.tolist()

    assert low[1] == pytest.approx(-math.pi)
    assert high[1] == pytest.approx(math.pi)
    assert low[0] == -high[0] < 0


def test_cartpole_never_terminates_early() -> None:
    task = _task("cartpole_swingup_sparse")
    sim = _sim_at(task, x=[0.0, 1.0], theta=[0.0, UPRIGHT_RAD])

    assert task.terminated(sim, torch.zeros(2, 0)).tolist() == [False, False]


def test_cartpole_sparse_solvable_by_scripted_controller() -> None:
    # Without this, a task nothing can solve would pass every constraint test above.
    n = 32
    task = _task("cartpole_swingup_sparse")
    controller = CartpoleSwingup(task.mjcf)

    success = first_episode_success(task, lambda _env, obs: controller(obs), n, seed=0)

    assert float(success.float().mean()) >= 0.75

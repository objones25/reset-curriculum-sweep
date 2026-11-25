import math
from dataclasses import dataclass

import pytest
import torch
from controllers import PendulumSwingup
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.sim.env import VecEnv
from resetsweep.tasks.base import TASKS, Task
from resetsweep.tasks.pendulum import TORQUE_LIMIT_NM
from resetsweep.types import Outcomes, StartStates, StepCounter

TASKS.load("resetsweep.tasks")

UPRIGHT_RAD = math.pi


def _task(name: str) -> Task:
    return TASKS.construct(name)  # pyright: ignore[reportReturnType] # registry is untyped


def _sim_at(
    task: Task, theta: list[float], omega: list[float] | None = None
) -> BatchSim:
    n = len(theta)
    sim = BatchSim(task.mjcf, nworld=n, device="cpu", frame_skip=task.frame_skip)
    q = torch.tensor(theta).unsqueeze(1)
    v = torch.tensor(omega if omega is not None else [0.0] * n).unsqueeze(1)
    sim.set_state(torch.arange(n), q, v)
    return sim


def test_sparse_reward_zero_when_hanging_and_one_when_upright() -> None:
    task = _task("pendulum_sparse")
    sim = _sim_at(task, [0.0, UPRIGHT_RAD])
    extra = torch.zeros(2, 0)

    reward = task.reward(sim, extra, torch.zeros(2, 1))

    assert reward.tolist() == [0.0, 1.0]


def test_sparse_reward_is_zero_just_outside_tolerance() -> None:
    task = _task("pendulum_sparse")
    sim = _sim_at(task, [UPRIGHT_RAD - 0.5, UPRIGHT_RAD + 0.5])

    reward = task.reward(sim, torch.zeros(2, 0), torch.zeros(2, 1))

    assert reward.tolist() == [0.0, 0.0]


def test_sparse_reward_wraps_angle_past_two_pi() -> None:
    task = _task("pendulum_sparse")
    sim = _sim_at(task, [UPRIGHT_RAD + 2 * math.pi, UPRIGHT_RAD - 2 * math.pi])

    reward = task.reward(sim, torch.zeros(2, 0), torch.zeros(2, 1))

    assert reward.tolist() == [1.0, 1.0]


@pytest.mark.parametrize("name", ["pendulum_sparse", "pendulum_dense"])
def test_action_cost_is_point_zero_one_times_squared_action(name: str) -> None:
    task = _task(name)
    sim = _sim_at(task, [0.0, 0.0])
    extra = torch.zeros(2, 0)

    free = task.reward(sim, extra, torch.zeros(2, 1))
    paid = task.reward(sim, extra, torch.tensor([[1.0], [0.5]]))

    assert (free - paid).tolist() == pytest.approx([0.01, 0.0025])


def test_sparse_success_requires_upright_within_tolerance() -> None:
    task = _task("pendulum_sparse")
    sim = _sim_at(task, [0.0, UPRIGHT_RAD, UPRIGHT_RAD + 0.1, UPRIGHT_RAD + 0.5])

    success = task.success_now(sim, torch.zeros(4, 0))

    assert success.tolist() == [False, True, True, False]


def test_dense_and_sparse_share_one_mjcf() -> None:
    assert _task("pendulum_dense").mjcf == _task("pendulum_sparse").mjcf


def test_both_pendulums_name_the_dense_sibling() -> None:
    assert _task("pendulum_sparse").dense_sibling == "pendulum_dense"
    assert _task("pendulum_dense").dense_sibling == "pendulum_dense"


def test_control_step_is_0_05_s_and_episode_is_5_s() -> None:
    import mujoco

    task = _task("pendulum_sparse")
    model = mujoco.MjModel.from_xml_string(task.mjcf)  # pyright: ignore[reportAttributeAccessIssue]

    assert task.frame_skip * model.opt.timestep == pytest.approx(0.05)
    assert task.max_steps * task.frame_skip * model.opt.timestep == pytest.approx(5.0)


def test_torque_limit_cannot_lift_directly() -> None:
    # Full torque from hanging for 1 s must stay below pi/2: a swing-up is required.
    # Commands far past the limit, so the clip (the real torque limit) is what is tested.
    task = _task("pendulum_sparse")
    sim = _sim_at(task, [0.0, 0.0])
    ctrl = torch.tensor([[100.0], [-100.0]])
    steps = round(1.0 / (task.frame_skip * 0.01))
    peak = torch.zeros(2)

    for _ in range(steps):
        sim.step(ctrl)
        peak = torch.maximum(peak, sim.qpos[:, 0].abs())

    assert float(peak.max()) < math.pi / 2


def test_actions_beyond_the_torque_limit_are_clipped() -> None:
    task = _task("pendulum_sparse")
    sim = _sim_at(task, [0.0, 0.0])
    ctrl = torch.tensor([[1.0], [50.0]])

    for _ in range(10):
        sim.step(ctrl)

    assert float(sim.qpos[0, 0]) == pytest.approx(float(sim.qpos[1, 0]), abs=1e-5)


def test_dense_reward_increases_with_height() -> None:
    task = _task("pendulum_dense")
    thetas = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, UPRIGHT_RAD]
    sim = _sim_at(task, thetas)

    reward = task.reward(sim, torch.zeros(5, 0), torch.zeros(5, 1))

    assert reward.tolist() == pytest.approx(
        [(1 - math.cos(t)) / 2 for t in thetas], abs=1e-5
    )
    assert bool((reward[1:] > reward[:-1]).all())


def test_sample_start_is_hanging_with_small_noise() -> None:
    task = _task("pendulum_sparse")

    s = task.sample_start(256, torch.Generator().manual_seed(0))

    assert float(s.qpos.abs().max()) <= 0.1
    assert float(s.qvel.abs().max()) <= 0.1
    assert float(s.qpos.std()) > 0.01


def test_start_box_bounds_are_angle_pi_and_speed_8() -> None:
    task = _task("pendulum_sparse")

    assert task.start_box.low.tolist() == pytest.approx([-math.pi, -8.0])
    assert task.start_box.high.tolist() == pytest.approx([math.pi, 8.0])


def test_observation_is_cos_sin_omega() -> None:
    task = _task("pendulum_sparse")
    sim = _sim_at(task, [0.0, UPRIGHT_RAD / 2], [1.5, -2.0])

    obs = task.observe(sim, torch.zeros(2, 0))

    expected = torch.tensor([[1.0, 0.0, 1.5], [math.cos(math.pi / 2), 1.0, -2.0]])
    assert torch.allclose(obs, expected, atol=1e-5)


def test_pendulum_never_terminates_early() -> None:
    task = _task("pendulum_sparse")
    sim = _sim_at(task, [0.0, UPRIGHT_RAD])

    assert task.terminated(sim, torch.zeros(2, 0)).tolist() == [False, False]


@dataclass
class _TaskStartMethod:
    """Reset rule that draws every start from the task's own distribution."""

    task: Task
    gen: torch.Generator
    name: str = "task_start"

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        return self.task.sample_start(env_ids.numel(), self.gen), env_ids.clone()

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None:
        return None


def test_scripted_swingup_reaches_sparse_success() -> None:
    # Without this, a task that nothing can solve would pass every constraint test above.
    n = 32
    task = _task("pendulum_sparse")
    gen = torch.Generator().manual_seed(0)
    env = VecEnv(
        task,
        num_envs=n,
        device="cpu",
        counter=StepCounter(budget=10 * n * task.max_steps),
        gen=gen,
    )
    env.bind(_TaskStartMethod(task, gen))
    controller = PendulumSwingup(task.mjcf, TORQUE_LIMIT_NM)
    obs = env.reset_all()
    succeeded = torch.zeros(n, dtype=torch.bool)
    seen_full = torch.zeros(n, dtype=torch.bool)

    for _ in range(
        2 * task.max_steps
    ):  # the first full episode ends within 2 * max_steps
        out = env.step(controller(obs))
        obs = out.obs
        o = out.outcomes
        if o is not None:
            first_full = o.full & ~seen_full[o.env_ids]
            ids = o.env_ids[first_full]
            succeeded[ids] = o.success[first_full]
            seen_full[ids] = True

    assert bool(seen_full.all())
    assert float(succeeded.float().mean()) >= 0.9

import math

import mujoco
import numpy as np
import pytest
import torch
from controllers import SamplingPlanner, first_episode_success

from resetsweep.sim.batch import BatchSim
from resetsweep.sim.dynamics import step_task
from resetsweep.tasks.arm import GOAL_RADIUS_M, GOAL_XY, PUCK_RADIUS_M
from resetsweep.tasks.base import TASKS, Task
from resetsweep.types import StartStates

TASKS.load("resetsweep.tasks")

ZERO_VEL = [0.0] * 6
CAPSULE_RADIUS_M = 0.02  # link capsules, from the MJCF


def _task(name: str = "arm_sparse") -> Task:
    return TASKS.construct(name)  # pyright: ignore[reportReturnType] # registry is untyped


def _sim_at(task: Task, q: list[list[float]]) -> BatchSim:
    n = len(q)
    sim = BatchSim(task.mjcf, nworld=n, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(n), torch.tensor(q), torch.zeros(n, 6))
    return sim


def _states(q: list[list[float]]) -> StartStates:
    n = len(q)
    return StartStates(torch.tensor(q), torch.zeros(n, 6), torch.zeros(n, 0))


def _hand_xy(task: Task, arm_q: list[float]) -> np.ndarray:
    """Hand position from MuJoCo's own forward kinematics, independent of the task's code."""
    model = mujoco.MjModel.from_xml_string(task.mjcf)  # pyright: ignore[reportAttributeAccessIssue] # no stub for the C-extension class
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    data.qpos[:3] = arm_q
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    return data.site_xpos[model.site("hand").id, :2].copy()


def test_puck_moves_when_pushed_by_link() -> None:
    task = _task()
    # Arm straight along +x; puck 0.2 m beside it. Torque on joint 1 swings the arm into the puck.
    sim = _sim_at(task, [[0.0, 0.0, 0.0, 0.5, 0.2, 0.0]])

    for _ in range(30):
        step_task(task, sim, torch.zeros(1, 0), torch.tensor([[1.0, 0.0, 0.0]]))

    moved = (sim.qpos[0, 3:5] - torch.tensor([0.5, 0.2])).norm()
    assert float(moved) > 0.3  # shoved well away from where it began


def test_puck_stays_put_when_arm_does_not_touch_it() -> None:
    task = _task()
    sim = _sim_at(task, [[0.0, 0.0, 0.0, 0.5, 0.2, 0.0]])

    for _ in range(30):
        step_task(task, sim, torch.zeros(1, 0), torch.tensor([[-1.0, 0.0, 0.0]]))

    assert sim.qpos[0, 3:5].tolist() == pytest.approx([0.5, 0.2], abs=1e-4)


def test_sparse_reward_only_when_puck_in_goal() -> None:
    task = _task("arm_sparse")
    gx, gy = GOAL_XY
    inside = [0.0, 0.0, 0.0, gx + 0.5 * GOAL_RADIUS_M, gy, 0.0]
    outside = [0.0, 0.0, 0.0, gx - 2 * GOAL_RADIUS_M, gy, 0.0]
    sim = _sim_at(task, [inside, outside])
    extra = torch.zeros(2, 0)

    reward = task.reward(sim, extra, torch.zeros(2, 3))

    assert reward.tolist() == [1.0, 0.0]
    assert task.success_now(sim, extra).tolist() == [True, False]


def test_sparse_reward_subtracts_action_cost() -> None:
    task = _task("arm_sparse")
    gx, gy = GOAL_XY
    sim = _sim_at(task, [[0.0, 0.0, 0.0, gx, gy, 0.0]])
    extra = torch.zeros(1, 0)

    idle = task.reward(sim, extra, torch.zeros(1, 3))
    effort = task.reward(sim, extra, torch.ones(1, 3))

    assert float(idle) == pytest.approx(1.0)
    assert float(effort) == pytest.approx(1.0 - 0.01 * 3)


def test_dense_reward_is_minus_distances_plus_sparse_term() -> None:
    sparse, dense = _task("arm_sparse"), _task("arm_dense")
    arm_q = [0.4, -0.3, 0.2]
    puck = [0.5, -0.2]
    goal = np.array(GOAL_XY)
    sim = _sim_at(dense, [[*arm_q, *puck, 0.0]])
    extra = torch.zeros(1, 0)
    action = torch.full((1, 3), 0.5)
    hand_to_puck = np.linalg.norm(_hand_xy(dense, arm_q) - np.array(puck))
    puck_to_goal = np.linalg.norm(np.array(puck) - goal)

    reward = dense.reward(sim, extra, action)

    expected = -(hand_to_puck + puck_to_goal) + float(sparse.reward(sim, extra, action))
    assert float(reward) == pytest.approx(expected, abs=1e-5)


def test_dense_reward_includes_sparse_bonus_in_goal() -> None:
    dense = _task("arm_dense")
    gx, gy = GOAL_XY
    sim = _sim_at(dense, [[0.0, 0.0, 0.0, gx, gy, 0.0]])
    hand_to_puck = np.linalg.norm(_hand_xy(dense, [0.0, 0.0, 0.0]) - np.array(GOAL_XY))

    reward = dense.reward(sim, torch.zeros(1, 0), torch.zeros(1, 3))

    assert float(reward) == pytest.approx(1.0 - hand_to_puck, abs=1e-5)


def test_sample_starts_do_not_already_succeed() -> None:
    task = _task()
    starts = task.sample_start(64, torch.Generator().manual_seed(0))
    sim = BatchSim(task.mjcf, nworld=64, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(64), starts.qpos, starts.qvel)

    assert not bool(task.success_now(sim, starts.extra).any())


def test_sample_starts_vary_in_puck_position() -> None:
    task = _task()

    starts = task.sample_start(64, torch.Generator().manual_seed(0))

    assert float(starts.qpos[:, 3].std()) > 0.01
    assert float(starts.qpos[:, 4].std()) > 0.01


def test_is_valid_rejects_interpenetrating_start() -> None:
    task = _task()
    arm_q = [0.3, 0.2, -0.1]
    hand = _hand_xy(task, arm_q)
    clear = [*arm_q, 0.5, -0.3, 0.0]
    on_the_hand = [
        *arm_q,
        float(hand[0]),
        float(hand[1]),
        0.0,
    ]  # puck centre on the link tip

    valid = task.is_valid(_states([clear, on_the_hand]))

    assert valid.tolist() == [True, False]


def test_is_valid_accepts_touching_but_rejects_deep_penetration() -> None:
    task = _task()
    arm_q = [0.0, 0.0, 0.0]  # straight along +x, tip at (0.8, 0)
    tip_x = float(_hand_xy(task, arm_q)[0])
    touch = tip_x + CAPSULE_RADIUS_M + PUCK_RADIUS_M  # surfaces exactly in contact
    rows = [
        [*arm_q, touch + 0.01, 0.0, 0.0],  # a 1 cm gap
        [*arm_q, touch, 0.0, 0.0],  # touching
        [
            *arm_q,
            touch - 0.004,
            0.0,
            0.0,
        ],  # 4 mm in: soft-contact sag, not interpenetration
        [*arm_q, touch - 0.03, 0.0, 0.0],  # 3 cm in
        [*arm_q, tip_x - 0.1, 0.0, 0.0],  # the puck sits on the link
    ]

    valid = task.is_valid(_states(rows))

    assert valid.tolist() == [True, True, True, False, False]


def test_is_valid_agrees_row_by_row_across_batch_sizes() -> None:
    task = _task()
    gen = torch.Generator().manual_seed(1)
    starts = task.sample_start(300, gen)
    arm_q = [0.0, 0.0, 0.0]
    hand = _hand_xy(task, arm_q)
    bad = _states([[*arm_q, float(hand[0]), float(hand[1]), 0.0]] * 7)
    mixed = StartStates.cat([starts, bad])

    together = task.is_valid(mixed)
    alone = torch.cat(
        [task.is_valid(mixed.index(torch.tensor([i]))) for i in (0, 150, 299, 300, 306)]
    )

    assert together[:300].all()
    assert not together[300:].any()
    assert alone.tolist() == [True, True, True, False, False]


def test_is_valid_rejects_nan_without_poisoning_other_rows() -> None:
    task = _task()
    good = task.canonical_start()
    nan = StartStates(torch.full_like(good.qpos, float("nan")), good.qvel, good.extra)

    valid = task.is_valid(StartStates.cat([good, nan, good]))

    assert valid.tolist() == [True, False, True]


@pytest.mark.parametrize("name", ["arm_sparse", "arm_dense"])
def test_is_valid_accepts_every_state_random_rollouts_reach(name: str) -> None:
    task = _task(name)
    n = 48
    gen = torch.Generator().manual_seed(3)
    # Arm starts stretched toward the puck's side of the table so random sweeps do hit it.
    arm = (torch.rand(n, 3, generator=gen) * 2 - 1) * 0.8
    puck = torch.stack(
        [
            0.3 + 0.4 * torch.rand(n, generator=gen),
            0.3 * (torch.rand(n, generator=gen) * 2 - 1),
        ],
        dim=1,
    )
    qpos0 = torch.cat([arm, puck, torch.zeros(n, 1)], dim=1)
    start = StartStates(qpos0, torch.zeros(n, 6), torch.zeros(n, 0))
    keep = task.is_valid(start).nonzero(as_tuple=True)[0]
    start = start.index(keep)
    m = start.n
    sim = BatchSim(task.mjcf, nworld=m, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(m), start.qpos, start.qvel)
    visited: list[StartStates] = []
    action = torch.zeros(m, 3)
    for t in range(200):
        if t % 8 == 0:
            action = torch.randint(0, 2, (m, 3), generator=gen).float() * 2 - 1
        step_task(task, sim, torch.zeros(m, 0), action)
        visited.append(
            StartStates(sim.qpos.clone(), sim.qvel.clone(), torch.zeros(m, 0))
        )
    reached = StartStates.cat(visited)

    valid = task.is_valid(reached)

    puck_moved = (sim.qpos[:, 3:5] - start.qpos[:, 3:5]).norm(dim=1)
    assert m >= 24
    assert int((puck_moved > 0.05).sum()) >= 5  # the rollouts really did push pucks
    assert bool(valid.all()), (
        f"{int((~valid).sum())} of {valid.numel()} reached states rejected"
    )


def test_zero_actions_never_solve_from_sample_start() -> None:
    task = _task()
    n = 8

    success = first_episode_success(
        task, lambda env, _obs: torch.zeros(n, 3), n, seed=0
    )

    assert int(success.sum()) == 0


@pytest.mark.slow
def test_arm_sparse_solvable_by_scripted_controller() -> None:
    # A sampling MPC on the dense sibling's reward must reach held success from sample_start.
    n = 8
    task = _task("arm_sparse")
    dense = _task(task.dense_sibling)
    planner = SamplingPlanner(dense, num_envs=n, seed=0)

    success = first_episode_success(
        task,
        lambda env, _obs: planner(*env.sim.get_state(torch.arange(n)), env.extra),
        n,
        seed=0,
    )

    assert int(success.sum()) >= math.ceil(0.75 * n)

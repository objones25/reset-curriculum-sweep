import math
import time

import pytest
import torch
from controllers import SamplingPlanner, first_episode_success, run_first_episode

from resetsweep.sim.batch import BatchSim
from resetsweep.sim.dynamics import step_task
from resetsweep.tasks.base import TASKS, Task
from resetsweep.tasks.claw import (
    DISC_RADIUS_M,
    DISC_XY,
    FINGER_RADIUS_M,
    GOAL_ANGLE_RAD,
    GOAL_TOL_RAD,
    PENETRATION_TOL_M,
)
from resetsweep.types import StartStates

TASKS.load("resetsweep.tasks")

# qpos = (base x, base y, left finger, right finger, disc angle); qvel has the same 5 entries.
# The base carries two fingers 0.12 m either side of it; finger joints slide inward (left +x,
# right -x), so a negative right-finger qpos closes that finger on the disc.
NQ = 5
OPEN_FINGERS = [0.0, 0.0]
# Base x that centres the gripper on the disc, and the disc's centre.
DISC_X = DISC_XY[0]
ACT_DIM = 4


def _task() -> Task:
    return TASKS.construct("claw")  # pyright: ignore[reportReturnType] # registry is untyped


def _states(q: list[list[float]]) -> StartStates:
    n = len(q)
    return StartStates(torch.tensor(q), torch.zeros(n, NQ), torch.zeros(n, 0))


def _sim_at(task: Task, q: list[list[float]]) -> BatchSim:
    n = len(q)
    sim = BatchSim(task.mjcf, nworld=n, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(n), torch.tensor(q), torch.zeros(n, NQ))
    return sim


def test_dense_sibling_is_itself() -> None:
    assert _task().dense_sibling == "claw"
    assert _task().name == "claw"


def test_disc_turns_only_with_finger_contact() -> None:
    task = _task()
    # Both worlds: gripper centred on the disc, 0.1 m below it, fingers open, dragged up (+y).
    # World 1 also closes the right finger onto the rim; world 0 leaves it open, so the fingers
    # slide past the disc without touching it.
    start = [DISC_X, -0.1, *OPEN_FINGERS, 0.0]
    sim = _sim_at(task, [start, start])
    extra = torch.zeros(2, 0)
    drag = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0]])

    for _ in range(40):
        step_task(task, sim, extra, drag)

    assert (
        float(sim.qpos[0, 4]) == 0.0
    )  # fingers never touched: not a flicker of motion
    assert float(sim.qpos[1, 4]) > 0.1  # the closed finger carried the rim round


def test_disc_holds_its_angle_without_contact() -> None:
    task = _task()
    sim = _sim_at(task, [[0.0, 0.3, *OPEN_FINGERS, 0.5]])

    for _ in range(40):
        step_task(task, sim, torch.zeros(1, 0), torch.zeros(1, ACT_DIM))

    assert float(sim.qpos[0, 4]) == pytest.approx(0.5, abs=1e-4)


@pytest.mark.parametrize(
    ("offset_rad", "expected"),
    [
        (0.0, True),
        (0.9 * GOAL_TOL_RAD, True),
        (-0.9 * GOAL_TOL_RAD, True),
        (1.5 * GOAL_TOL_RAD, False),
        (-GOAL_ANGLE_RAD, False),  # back at angle 0
    ],
)
def test_success_at_goal_angle_held(offset_rad: float, expected: bool) -> None:
    task = _task()
    start = _states([[0.0, 0.3, *OPEN_FINGERS, GOAL_ANGLE_RAD + offset_rad]])

    outcomes = run_first_episode(
        task, lambda env, _obs: torch.zeros(1, ACT_DIM), start, seed=0
    )

    assert outcomes.success.tolist() == [expected]


def test_sample_starts_do_not_already_succeed() -> None:
    task = _task()
    starts = task.sample_start(64, torch.Generator().manual_seed(0))
    sim = BatchSim(task.mjcf, nworld=64, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(64), starts.qpos, starts.qvel)

    assert not bool(task.success_now(sim, starts.extra).any())


def test_sample_starts_vary_in_base_position() -> None:
    starts = _task().sample_start(64, torch.Generator().manual_seed(0))

    assert float(starts.qpos[:, 0].std()) > 0.01
    assert float(starts.qpos[:, 1].std()) > 0.01


def test_zero_actions_never_solve_from_sample_start() -> None:
    n = 8

    success = first_episode_success(
        _task(), lambda env, _obs: torch.zeros(n, ACT_DIM), n, seed=0
    )

    assert int(success.sum()) == 0


def test_dense_reward_rises_as_the_disc_nears_the_goal() -> None:
    task = _task()
    rows = [
        [0.0, 0.3, *OPEN_FINGERS, 0.0],
        [0.0, 0.3, *OPEN_FINGERS, GOAL_ANGLE_RAD / 2],
        [0.0, 0.3, *OPEN_FINGERS, GOAL_ANGLE_RAD],
    ]
    sim = _sim_at(task, rows)

    reward = task.reward(sim, torch.zeros(3, 0), torch.zeros(3, ACT_DIM)).tolist()

    assert reward[0] < reward[1] < reward[2]
    assert reward[2] - reward[1] > 1.0  # the success bonus on top of the shaping


def test_dense_reward_rises_as_the_gripper_nears_the_disc() -> None:
    task = _task()
    rows = [[0.0, 0.3, *OPEN_FINGERS, 0.0], [0.1, 0.0, *OPEN_FINGERS, 0.0]]
    sim = _sim_at(task, rows)

    reward = task.reward(sim, torch.zeros(2, 0), torch.zeros(2, ACT_DIM)).tolist()

    assert reward[0] < reward[1]


def test_reward_subtracts_action_cost() -> None:
    task = _task()
    sim = _sim_at(task, [[0.0, 0.3, *OPEN_FINGERS, 0.0]])
    extra = torch.zeros(1, 0)

    idle = task.reward(sim, extra, torch.zeros(1, ACT_DIM))
    effort = task.reward(sim, extra, torch.ones(1, ACT_DIM))

    assert float(idle - effort) == pytest.approx(0.01 * ACT_DIM)


def test_reward_and_success_do_not_raise_on_nan_state() -> None:
    task = _task()
    sim = _sim_at(task, [[0.0, 0.3, *OPEN_FINGERS, 0.0]])
    sim.qpos[0, 4] = float("nan")
    extra = torch.zeros(1, 0)

    reward = task.reward(sim, extra, torch.zeros(1, ACT_DIM))
    success = task.success_now(sim, extra)

    assert reward.shape == (1,)
    assert success.tolist() == [False]  # NaN never counts as success


def test_reward_stays_on_the_sim_device() -> None:
    task = _task()
    sim = _sim_at(task, [[0.0, 0.3, *OPEN_FINGERS, 0.0]])

    reward = task.reward(sim, torch.zeros(1, 0), torch.zeros(1, ACT_DIM))

    assert reward.device == sim.qpos.device
    assert reward.dtype == torch.float32


def test_is_valid_rejects_deep_interpenetration_but_accepts_touching() -> None:
    task = _task()
    # Right finger centre sits at base_x + 0.12 and touches the rim when
    # base_x + 0.12 - FINGER_RADIUS_M = DISC_X + DISC_RADIUS_M, finger open.
    touch_base_x = DISC_X + DISC_RADIUS_M + FINGER_RADIUS_M - 0.12
    rows = [
        [touch_base_x + 0.01, 0.0, *OPEN_FINGERS, 0.0],  # 1 cm gap
        [touch_base_x, 0.0, *OPEN_FINGERS, 0.0],  # touching
        [touch_base_x - 0.004, 0.0, *OPEN_FINGERS, 0.0],  # 4 mm in: soft-contact sag
        [touch_base_x - 0.06, 0.0, *OPEN_FINGERS, 0.0],  # 6 cm in
        [DISC_X - 0.12, 0.0, *OPEN_FINGERS, 0.0],  # the right finger inside the disc
    ]

    valid = task.is_valid(_states(rows))

    assert 0.004 < PENETRATION_TOL_M < 0.06
    assert valid.tolist() == [True, True, True, False, False]


def test_is_valid_rejects_nan_without_poisoning_other_rows() -> None:
    task = _task()
    good = task.canonical_start()
    nan = StartStates(torch.full_like(good.qpos, float("nan")), good.qvel, good.extra)

    valid = task.is_valid(StartStates.cat([good, nan, good]))

    assert valid.tolist() == [True, False, True]


def test_is_valid_agrees_row_by_row_across_batch_sizes() -> None:
    task = _task()
    starts = task.sample_start(300, torch.Generator().manual_seed(1))
    deep = [DISC_X - 0.12, 0.0, *OPEN_FINGERS, 0.0]
    mixed = StartStates.cat([starts, _states([deep] * 7)])

    together = task.is_valid(mixed)
    alone = torch.cat(
        [task.is_valid(mixed.index(torch.tensor([i]))) for i in (0, 150, 299, 300, 306)]
    )

    assert together[:300].all()
    assert not together[300:].any()
    assert alone.tolist() == [True, True, True, False, False]


def test_is_valid_accepts_every_state_random_rollouts_reach() -> None:
    task = _task()
    n = 48
    gen = torch.Generator().manual_seed(3)
    # Start the base near the disc so random fingers do hit it.
    base = torch.stack(
        [
            DISC_X - 0.15 + 0.3 * torch.rand(n, generator=gen),
            0.3 * (torch.rand(n, generator=gen) * 2 - 1),
        ],
        dim=1,
    )
    qpos0 = torch.cat([base, torch.zeros(n, 2), torch.zeros(n, 1)], dim=1)
    start = StartStates(qpos0, torch.zeros(n, NQ), torch.zeros(n, 0))
    keep = task.is_valid(start).nonzero(as_tuple=True)[0]
    start = start.index(keep)
    m = start.n
    sim = BatchSim(task.mjcf, nworld=m, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(m), start.qpos, start.qvel)
    visited: list[StartStates] = []
    action = torch.zeros(m, ACT_DIM)
    for t in range(200):
        if t % 8 == 0:
            action = torch.randint(-1, 2, (m, ACT_DIM), generator=gen).float()
        step_task(task, sim, torch.zeros(m, 0), action)
        visited.append(
            StartStates(sim.qpos.clone(), sim.qvel.clone(), torch.zeros(m, 0))
        )
    reached = StartStates.cat(visited)

    valid = task.is_valid(reached)

    disc_moved = sim.qpos[:, 4].abs()
    assert m >= 24
    assert int((disc_moved > 0.05).sum()) >= 5  # fingers really did turn discs
    assert bool(valid.all()), (
        f"{int((~valid).sum())} of {valid.numel()} reached states rejected"
    )


@pytest.mark.slow
def test_claw_success_reachable_by_planner_or_script() -> None:
    # A sampling MPC on the claw's own dense reward must reach held success from sample_start:
    # at least 3 of 4 starts, within one episode.
    n = 4
    task = _task()
    planner = SamplingPlanner(task, num_envs=n, seed=0)
    t0 = time.perf_counter()

    success = first_episode_success(
        task,
        lambda env, _obs: planner(*env.sim.get_state(torch.arange(n)), env.extra),
        n,
        seed=0,
    )

    print(
        f"claw planner success: {int(success.sum())}/{n} in {time.perf_counter() - t0:.1f}s"
    )
    assert int(success.sum()) >= math.ceil(0.75 * n)

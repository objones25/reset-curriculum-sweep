import math

import pytest
import torch
from controllers import (
    ConstantAction,
    RampGreedyUphill,
    RampPump,
    SamplingPlanner,
    _FixedStartMethod,
    first_episode_success,
    run_first_episode,
)
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.sim.dynamics import step_task
from resetsweep.sim.env import VecEnv
from resetsweep.tasks import ramp
from resetsweep.tasks.base import TASKS, Task
from resetsweep.types import StartStates, StepCounter

TASKS.load("resetsweep.tasks")
_G = 9.81


def _task(name: str = "ramp_deceptive") -> Task:
    return TASKS.construct(name)  # pyright: ignore[reportReturnType] # registry is untyped


def _sim_at(task: Task, s: list[float], v: list[float] | None = None) -> BatchSim:
    n = len(s)
    sim = BatchSim(task.mjcf, nworld=n, device="cpu", frame_skip=task.frame_skip)
    qvel = torch.tensor(v if v is not None else [0.0] * n).unsqueeze(1)
    sim.set_state(torch.arange(n), torch.tensor(s).unsqueeze(1), qvel)
    return sim


def _states(s: list[float], v: list[float] | None = None) -> StartStates:
    n = len(s)
    qvel = torch.tensor(v if v is not None else [0.0] * n).unsqueeze(1)
    return StartStates(torch.tensor(s).unsqueeze(1), qvel, torch.zeros(n, 0))


def _slope(s: Tensor) -> Tensor:
    return ramp.height_slope(s)


def test_registry_has_ramp_pair() -> None:
    assert {"ramp_deceptive", "ramp_dense"} <= set(TASKS.names())
    assert _task().dense_sibling == "ramp_dense"


def test_sparse_reward_zero_until_goal() -> None:
    task = _task()
    s = [-ramp.NEAR_STOP_M, -1.0, 0.0, 1.0, ramp.GOAL_S_M - 0.01]
    sim = _sim_at(task, s)

    reward = task.reward(sim, torch.zeros(5, 0), torch.ones(5, 1))

    assert reward.tolist() == [0.0] * 5


def test_sparse_reward_one_on_goal_plateau() -> None:
    task = _task()
    sim = _sim_at(task, [ramp.GOAL_S_M, ramp.FAR_STOP_M])

    reward = task.reward(sim, torch.zeros(2, 0), torch.zeros(2, 1))

    assert reward.tolist() == [1.0, 1.0]


def test_dense_reward_is_the_carts_height() -> None:
    task = _task("ramp_dense")
    s = [-ramp.NEAR_STOP_M, 0.0, 2.0, ramp.FAR_STOP_M]
    sim = _sim_at(task, s)
    expected = [
        ramp.NEAR_STOP_M * math.sin(math.radians(ramp.NEAR_ANGLE_DEG)),
        0.0,
        2.0 * math.sin(math.radians(ramp.FAR_ANGLE_DEG)),
        ramp.FAR_SLOPE_LEN_M * math.sin(math.radians(ramp.FAR_ANGLE_DEG)),
    ]

    reward = task.reward(sim, torch.zeros(4, 0), torch.zeros(4, 1))

    assert reward.tolist() == pytest.approx(expected, abs=1e-5)


def test_goal_is_higher_than_the_near_slope_top() -> None:
    near_top = ramp.height_m(torch.tensor([-ramp.NEAR_STOP_M]))
    goal = ramp.height_m(torch.tensor([ramp.GOAL_S_M]))

    assert float(goal) > 1.5 * float(near_top)


def test_motor_beats_near_slope_but_loses_to_far_slope() -> None:
    near_pull = ramp.MASS_KG * _G * math.sin(math.radians(ramp.NEAR_ANGLE_DEG))
    far_pull = ramp.MASS_KG * _G * math.sin(math.radians(ramp.FAR_ANGLE_DEG))

    assert near_pull < ramp.FORCE_LIMIT_N < far_pull


def test_applied_force_is_gravity_along_the_profile() -> None:
    task = _task()
    sim = _sim_at(task, [-2.0, 1.0, ramp.GOAL_S_M + 0.1])

    force = task.applied_force(sim, torch.zeros(3, 0))  # pyright: ignore[reportAttributeAccessIssue] # optional hook

    expected = [
        ramp.MASS_KG * _G * math.sin(math.radians(ramp.NEAR_ANGLE_DEG)),
        -ramp.MASS_KG * _G * math.sin(math.radians(ramp.FAR_ANGLE_DEG)),
        0.0,
    ]
    assert force.shape == (3, 1)
    assert force[:, 0].tolist() == pytest.approx(expected, abs=1e-5)


def test_passive_cart_released_on_near_slope_climbs_back_up_the_far_slope() -> None:
    # Physics check that the profile is a valley: energy is conserved up to damping.
    task = _task()
    sim = _sim_at(task, [-ramp.NEAR_STOP_M])
    start_h = float(ramp.height_m(sim.qpos[:, 0])[0])
    peak = 0.0
    for _ in range(60):
        step_task(task, sim, torch.zeros(1, 0), torch.zeros(1, 1))
        peak = max(peak, float(ramp.height_m(sim.qpos[:, 0])[0]))
        if float(sim.qpos[0, 0]) > 0.5:
            far_peak = float(ramp.height_m(sim.qpos[:, 0])[0])
            peak = max(peak, far_peak)

    assert 0.6 * start_h < peak <= start_h * 1.02
    assert float(sim.qpos[0, 0]) > 0.0 or peak > 0.6 * start_h


def test_full_throttle_forward_from_start_never_reaches_goal() -> None:
    task = _task()
    n = 16

    outcomes = run_first_episode(task, ConstantAction(1.0), num_envs=n, seed=0)

    assert int(outcomes.success.sum()) == 0
    assert int(outcomes.reached_reward.sum()) == 0


def test_full_throttle_forward_stays_below_the_goal_for_ten_seconds() -> None:
    task = _task()
    n = 16
    gen = torch.Generator().manual_seed(0)
    env = VecEnv(
        task,
        num_envs=n,
        device="cpu",
        counter=StepCounter(budget=10 * n * task.max_steps),
        gen=gen,
    )
    env.bind(_FixedStartMethod(task, gen))
    env.reset_all()
    env.ep_step.zero_()
    peak = torch.full((n,), -1e9)
    for _ in range(round(10.0 / (task.frame_skip * 0.01))):
        env.step(torch.ones(n, 1))
        peak = torch.maximum(peak, env.sim.qpos[:, 0])

    assert task.max_steps * task.frame_skip * 0.01 >= 10.0
    # Starting up to 0.3 m behind the vertex it coasts some way up the far slope, no further.
    assert float(peak.max()) < ramp.GOAL_S_M - 2.0


def test_greedy_on_dense_reward_never_reaches_goal() -> None:
    task = _task()
    n = 16

    outcomes = run_first_episode(task, RampGreedyUphill(_slope), num_envs=n, seed=0)

    assert int(outcomes.success.sum()) == 0
    assert int(outcomes.reached_reward.sum()) == 0


def test_greedy_on_dense_reward_earns_less_dense_return_than_the_pump() -> None:
    # Greedy is not merely idle (it beats full throttle) but it is trapped at the near slope's
    # local maximum or the vertex, so even the dense reward is better served by going away.
    dense = _task("ramp_dense")

    greedy = run_first_episode(dense, RampGreedyUphill(_slope), num_envs=32, seed=1)
    pump = run_first_episode(dense, RampPump(ramp.GOAL_S_M), num_envs=32, seed=1)
    forward = run_first_episode(dense, ConstantAction(1.0), num_envs=32, seed=1)

    assert float(forward.ret.mean()) < float(greedy.ret.mean())
    assert 1.4 * float(greedy.ret.mean()) < float(pump.ret.mean())


@pytest.mark.slow
def test_myopic_planner_on_dense_reward_never_reaches_goal() -> None:
    # A 1.2 s receding-horizon optimiser of the dense reward (height) is greedy at the scale
    # of the task's deception; the pump needs several seconds of foresight.
    n = 8
    task = _task()
    dense = _task(task.dense_sibling)
    planner = SamplingPlanner(dense, num_envs=n, seed=0)

    success = first_episode_success(
        task,
        lambda env, _obs: planner(*env.sim.get_state(torch.arange(n)), env.extra),
        n,
        seed=0,
    )

    assert int(success.sum()) == 0


def test_back_then_forward_policy_reaches_goal() -> None:
    task = _task()

    success = first_episode_success(task, RampPump(ramp.GOAL_S_M), 1, seed=0)

    assert success.tolist() == [True]


def test_back_then_forward_policy_succeeds_for_75_percent_of_seeded_envs() -> None:
    task = _task()
    n = 64

    success = first_episode_success(task, RampPump(ramp.GOAL_S_M), n, seed=3)

    assert int(success.sum()) >= math.ceil(0.75 * n)


def test_sample_starts_do_not_already_succeed() -> None:
    task = _task()
    starts = task.sample_start(256, torch.Generator().manual_seed(0))
    sim = _sim_at(task, starts.qpos[:, 0].tolist())

    assert not bool(task.success_now(sim, starts.extra).any())


def test_is_valid_accepts_states_at_both_end_stops_even_after_a_hard_hit() -> None:
    task = _task()
    extra = torch.zeros(2, 0)
    pushes = torch.tensor([[-1.0], [1.0]])
    sim = _sim_at(task, [-ramp.NEAR_STOP_M + 0.5, ramp.FAR_STOP_M - 0.5], [-8.0, 8.0])
    seen: list[StartStates] = []
    near_depth = far_depth = 0.0
    for _ in range(120):
        step_task(task, sim, extra, pushes)
        seen.append(StartStates(sim.qpos.clone(), sim.qvel.clone(), extra))
        near_depth = max(near_depth, -float(sim.qpos[0, 0]) - ramp.NEAR_STOP_M)
        far_depth = max(far_depth, float(sim.qpos[1, 0]) - ramp.FAR_STOP_M)

    valid = task.is_valid(StartStates.cat(seen))

    assert near_depth > 0.0 and far_depth > 0.0  # both stops were really hit
    assert (
        max(near_depth, far_depth) < 0.15
    )  # stop give stays inside the stated tolerance
    assert bool(valid.all())


def test_is_valid_accepts_a_cart_resting_against_each_end_stop() -> None:
    task = _task()
    extra = torch.zeros(2, 0)
    sim = _sim_at(task, [-ramp.NEAR_STOP_M + 0.5, ramp.FAR_STOP_M - 0.5], [-3.0, 3.0])
    for _ in range(80):
        step_task(task, sim, extra, torch.tensor([[-1.0], [1.0]]))

    resting = StartStates(sim.qpos.clone(), sim.qvel.clone(), extra)

    assert float(sim.qpos[0, 0]) < -ramp.NEAR_STOP_M + 0.01
    assert float(sim.qpos[1, 0]) > ramp.FAR_STOP_M - 0.01
    assert bool(task.is_valid(resting).all())


def test_is_valid_rejects_nan_far_out_of_range_and_too_fast() -> None:
    task = _task()
    bad = _states(
        [float("nan"), -ramp.NEAR_STOP_M - 1.0, ramp.FAR_STOP_M + 1.0, 0.0],
        [0.0, 0.0, 0.0, 1e3],
    )

    assert task.is_valid(bad).tolist() == [False] * 4


def test_is_valid_accepts_the_start_box_corners_and_interior() -> None:
    task = _task()
    box = task.start_box
    good = _states(
        [float(box.low[0]), float(box.high[0]), 0.0],
        [float(box.low[1]), float(box.high[1]), 0.0],
    )

    assert task.is_valid(good).tolist() == [True] * 3


def test_is_valid_agrees_across_batch_sizes_and_nan_rows() -> None:
    task = _task()
    mixed = _states([0.0, float("nan"), 1.0], [0.0, 0.0, float("nan")])

    together = task.is_valid(mixed)
    alone = [task.is_valid(mixed.index(torch.tensor([i]))).item() for i in range(3)]

    assert together.tolist() == alone == [True, False, False]


@pytest.mark.parametrize("name", ["ramp_deceptive", "ramp_dense"])
def test_is_valid_accepts_every_state_rollouts_reach(name: str) -> None:
    task = _task(name)
    n = 64
    gen = torch.Generator().manual_seed(5)
    start = task.sample_start(n, gen)
    sim = BatchSim(task.mjcf, nworld=n, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(n), start.qpos, start.qvel)
    extra = torch.zeros(n, 0)
    visited: list[StartStates] = []
    action = torch.zeros(n, 1)
    pump = RampPump(ramp.GOAL_S_M)
    for t in range(300):
        if t % 8 == 0:
            action = torch.randint(0, 2, (n, 1), generator=gen).float() * 2 - 1
        if (
            t >= 100
        ):  # second phase: pump hard, so the end stops and the plateau are hit
            obs = torch.cat([sim.qpos, sim.qvel], dim=1)
            action = pump(None, obs)  # pyright: ignore[reportArgumentType] # env unused
        step_task(task, sim, extra, action)
        visited.append(StartStates(sim.qpos.clone(), sim.qvel.clone(), extra))
    reached = StartStates.cat(visited)

    valid = task.is_valid(reached)

    at_stop = (reached.qpos[:, 0] < -ramp.NEAR_STOP_M + 0.05) | (
        reached.qpos[:, 0] > ramp.FAR_STOP_M - 0.05
    )
    assert int(at_stop.sum()) > 0  # the rollouts really did visit an end stop
    assert bool(valid.all()), f"{int((~valid).sum())} of {valid.numel()} rejected"


def test_reward_and_success_stay_finite_for_nan_states() -> None:
    task = _task("ramp_dense")
    sim = _sim_at(task, [0.0, 1.0])
    sim.qpos[0, 0] = float("nan")

    success = task.success_now(sim, torch.zeros(2, 0))
    reward = task.reward(sim, torch.zeros(2, 0), torch.zeros(2, 1))

    assert success.tolist() == [False, False]
    assert reward.shape == (2,)

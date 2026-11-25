"""Trap, twin, infeasible-mass and coin-flip maze variants (spec §5.3 T3-T5)."""

import mujoco
import pytest
import torch
from controllers import MazePathFollower, _FixedStartMethod, run_first_episode
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.sim.env import VecEnv
from resetsweep.tasks.base import TASKS, Task
from resetsweep.tasks.maze import MazeGrid, build_maze
from resetsweep.types import Outcomes, StartStates, StepCounter

TASKS.load("resetsweep.tasks")

TRAPS = ["maze_trap", "maze_trap_twin"]
VARIANTS = [*TRAPS, "maze_infeasible", "maze_coinflip"]
MASS_DRAWS = 100_000


def _task(name: str, **settings: object) -> Task:
    return TASKS.construct(name, **settings)  # pyright: ignore[reportReturnType] # registry is untyped


def _grid(task: Task) -> MazeGrid:
    return task.grid  # pyright: ignore[reportAttributeAccessIssue] # maze tasks expose their grid


def _region(task: Task, qpos: Tensor) -> Tensor:
    return task.diagnostic_region(qpos)  # pyright: ignore[reportAttributeAccessIssue] # variants export it


def _rect_centre(grid: MazeGrid, rect: tuple[int, int, int, int]) -> list[float]:
    row_lo, row_hi, col_lo, col_hi = rect
    return [
        (col_lo + col_hi + 1) / 2 * grid.cell_m,
        (row_lo + row_hi + 1) / 2 * grid.cell_m,
    ]


def _one_start(task: Task, xy: list[float]) -> StartStates:
    return StartStates(
        torch.tensor([xy]), torch.zeros(1, 2), torch.zeros(1, task.extra_dim)
    )


def _push_for(
    task: Task, xy: list[float], direction: list[float], steps: int
) -> Tensor:
    """Final `(x, y)` after pushing at full force in `direction` for `steps` control steps."""
    gen = torch.Generator().manual_seed(0)
    env = VecEnv(
        task,
        num_envs=1,
        device="cpu",
        counter=StepCounter(budget=10 * steps),
        gen=gen,
    )
    env.bind(_FixedStartMethod(task, gen, _one_start(task, xy)))
    env.reset_all()
    env.ep_step.zero_()
    action = torch.tensor([direction])
    for _ in range(steps):
        env.step(action)
    return env.sim.qpos.clone()


# --- trap and twin ---------------------------------------------------------

DIRECTIONS = [[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]]
# Axis pushes and the diagonals that press into the pit's corners.
EXITS = [*DIRECTIONS, [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
LIP_INSET_M = 0.02  # how far inside the lip the "at the lip" start is
PIT_STEPS = 120  # 6 s of full force


def _pit_start(task: Task, direction: list[float], *, at_lip: bool) -> list[float]:
    """Pit centre, or the point `LIP_INSET_M` inside the lip in `direction`."""
    centre = _rect_centre(_grid(task), task.pit_rect)  # pyright: ignore[reportAttributeAccessIssue] # trap tasks expose their pit
    if not at_lip:
        return centre
    half = _grid(task).cell_m * (task.pit_rect[1] - task.pit_rect[0] + 1) / 2  # pyright: ignore[reportAttributeAccessIssue]
    return [
        c + d * (half - LIP_INSET_M) for c, d in zip(centre, direction, strict=True)
    ]


@pytest.mark.parametrize("at_lip", [False, True])
@pytest.mark.parametrize("direction", EXITS)
def test_trap_cannot_be_escaped_at_full_force(
    direction: list[float], at_lip: bool
) -> None:
    task = _task("maze_trap")
    start = _pit_start(task, direction, at_lip=at_lip)

    qpos = _push_for(task, start, direction, steps=PIT_STEPS)

    assert _region(task, qpos).tolist() == [True]


@pytest.mark.parametrize("at_lip", [False, True])
@pytest.mark.parametrize("direction", EXITS)
def test_twin_trap_can_be_escaped(direction: list[float], at_lip: bool) -> None:
    task = _task("maze_trap_twin")
    start = _pit_start(task, direction, at_lip=at_lip)

    qpos = _push_for(task, start, direction, steps=PIT_STEPS)

    assert _region(task, qpos).tolist() == [False]


def test_twin_has_identical_geometry_to_the_trap() -> None:
    trap, twin = _task("maze_trap"), _task("maze_trap_twin")

    assert twin.mjcf == trap.mjcf
    assert torch.equal(_grid(twin).occupied, _grid(trap).occupied)
    assert twin.pit_rect == trap.pit_rect  # pyright: ignore[reportAttributeAccessIssue] # trap tasks expose their pit


def test_control_leaves_the_action_alone_outside_the_pit() -> None:
    task = _task("maze_trap")
    grid = _grid(task)
    outside = grid.centre(grid.start_cell).tolist()
    sim = BatchSim(task.mjcf, nworld=1, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(1), torch.tensor([outside]), torch.zeros(1, 2))
    action = torch.tensor([[0.5, -1.0]])

    ctrl = task.control(sim, torch.zeros(1, 0), action)  # pyright: ignore[reportAttributeAccessIssue] # trap tasks control the action

    assert ctrl.tolist() == [[0.5, -1.0]]


def test_control_inside_the_pit_pulls_back_to_the_pit_centre() -> None:
    task = _task("maze_trap")
    grid = _grid(task)
    centre = _rect_centre(grid, task.pit_rect)  # pyright: ignore[reportAttributeAccessIssue] # trap tasks expose their pit
    above = [centre[0], centre[1] + 0.1]  # inside the pit cell, off its centre line
    sim = BatchSim(task.mjcf, nworld=1, device="cpu", frame_skip=task.frame_skip)
    sim.set_state(torch.arange(1), torch.tensor([above]), torch.zeros(1, 2))

    ctrl = task.control(sim, torch.zeros(1, 0), torch.zeros(1, 2))  # pyright: ignore[reportAttributeAccessIssue] # trap tasks control the action

    assert float(ctrl[0, 0]) == pytest.approx(0.0)
    assert float(ctrl[0, 1]) < -1.0  # pull is stronger than the motor's full force


def test_trap_scale_grows_the_pit() -> None:
    small = _task("maze_trap", trap_scale=0.5)
    large = _task("maze_trap", trap_scale=1.5)
    xy = (
        torch.rand(20_000, 2)
        * torch.tensor([_grid(large).cols, _grid(large).rows])
        * _grid(large).cell_m
    )

    assert int(_region(small, xy).sum()) < int(_region(large, xy).sum())


def test_trap_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="trap_scale must be positive"):
        _task("maze_trap", trap_scale=0.0)


def test_planning_grid_blocks_the_pit() -> None:
    task = _task("maze_trap")
    grid, planning = _grid(task), task.planning_grid  # pyright: ignore[reportAttributeAccessIssue] # trap tasks expose a planning grid
    row_lo, row_hi, col_lo, col_hi = task.pit_rect  # pyright: ignore[reportAttributeAccessIssue]

    assert bool(planning.occupied[row_lo : row_hi + 1, col_lo : col_hi + 1].all())
    assert not bool(grid.occupied[row_lo : row_hi + 1, col_lo : col_hi + 1].any())
    assert planning.shortest_path_cells >= grid.shortest_path_cells


# --- infeasible mass -------------------------------------------------------


def _mass_fractions() -> tuple[float, float, float]:
    task = _task("maze_infeasible")
    grid = _grid(task)
    starts = task.sample_start(MASS_DRAWS, torch.Generator().manual_seed(0))
    steps = grid.dist_cells[grid.cell_index(starts.qpos), grid.flat(grid.goal_cell)]
    infeasible = (steps == -1).float().mean()
    easy = ((steps >= 0) & (steps <= task.easy_radius_cells)).float().mean()  # pyright: ignore[reportAttributeAccessIssue] # infeasible task exposes its easy radius
    hard = (steps > task.easy_radius_cells).float().mean()  # pyright: ignore[reportAttributeAccessIssue]
    return float(easy), float(hard), float(infeasible)


def test_infeasible_mass_fractions_within_tolerance() -> None:
    easy, hard, infeasible = _mass_fractions()

    assert easy == pytest.approx(0.90, abs=0.01)
    assert infeasible == pytest.approx(0.0999, abs=0.005)
    assert hard < 0.001
    assert easy + hard + infeasible == pytest.approx(1.0)


def test_infeasible_pocket_has_no_bfs_path_to_goal() -> None:
    task = _task("maze_infeasible")
    grid = _grid(task)
    starts = task.sample_start(5_000, torch.Generator().manual_seed(1))
    cells = grid.cell_index(starts.qpos)
    pocket = cells[_region(task, starts.qpos)]

    assert pocket.numel() > 0
    assert bool((grid.dist_cells[pocket, grid.flat(grid.goal_cell)] == -1).all())
    assert bool((grid.dist_cells[pocket, grid.flat(grid.start_cell)] == -1).all())


def test_infeasible_diagnostic_region_is_exactly_the_pockets() -> None:
    task = _task("maze_infeasible")
    grid = _grid(task)
    starts = task.sample_start(5_000, torch.Generator().manual_seed(2))
    unreachable = (
        grid.dist_cells[grid.cell_index(starts.qpos), grid.flat(grid.goal_cell)] == -1
    )

    assert torch.equal(_region(task, starts.qpos), unreachable)


def test_infeasible_eval_states_are_the_feasible_start() -> None:
    task = _task("maze_infeasible")
    grid = _grid(task)

    starts = task.eval_states(64, seed=3)

    steps = grid.dist_cells[grid.cell_index(starts.qpos), grid.flat(grid.goal_cell)]
    assert bool((steps == grid.shortest_path_cells).all())
    assert not bool(_region(task, starts.qpos).any())


def test_infeasible_pocket_is_sealed_in_the_physics() -> None:
    task = _task("maze_infeasible")
    grid = _grid(task)
    starts = task.sample_start(2_000, torch.Generator().manual_seed(4))
    in_pocket = _region(task, starts.qpos)
    xy = starts.qpos[in_pocket][0].tolist()

    for direction in DIRECTIONS:
        qpos = _push_for(task, xy, direction, steps=100)

        assert bool(_region(task, qpos).all())
        assert (
            grid.cell_index(qpos).tolist()
            == grid.cell_index(torch.tensor([xy])).tolist()
        )


# --- coin flip -------------------------------------------------------------


def _block_start(task: Task, n: int) -> StartStates:
    grid = _grid(task)
    centre = _rect_centre(grid, task.block_rect)  # pyright: ignore[reportAttributeAccessIssue] # coinflip task exposes its block
    return StartStates(
        torch.tensor([centre]).repeat(n, 1), torch.zeros(n, 2), torch.zeros(n, 2)
    )


def _sim_with(task: Task, xy: Tensor) -> BatchSim:
    sim = BatchSim(
        task.mjcf, nworld=xy.shape[0], device="cpu", frame_skip=task.frame_skip
    )
    sim.set_state(torch.arange(xy.shape[0]), xy, torch.zeros_like(xy))
    return sim


@pytest.mark.parametrize("draw", [0.0, 1.0])
def test_coinflip_success_independent_of_actions(draw: float) -> None:
    task = _task("maze_coinflip")
    xy = _block_start(task, 2).qpos
    extra = torch.tensor([[draw, 1.0]]).repeat(2, 1)
    sim = _sim_with(task, xy)
    gen = torch.Generator().manual_seed(0)
    successes = []
    for _ in range(40):
        sim.step(torch.sign(torch.randn(2, 2, generator=gen)))
        successes.append(task.success_now(sim, extra))
    sim_still = _sim_with(task, xy)
    still = task.success_now(sim_still, extra)

    assert all(s.tolist() == [bool(draw)] * 2 for s in successes)
    assert still.tolist() == [bool(draw)] * 2


@pytest.mark.parametrize("draw", [0.0, 1.0])
def test_coinflip_block_episode_ignores_the_goal(draw: float) -> None:
    task = _task("maze_coinflip")
    grid = _grid(task)
    at_goal = grid.centre(grid.goal_cell).unsqueeze(0)
    extra = torch.tensor([[draw, 1.0]])

    success = task.success_now(_sim_with(task, at_goal), extra)

    assert success.tolist() == [bool(draw)]


def test_coinflip_outside_block_success_is_reaching_the_goal() -> None:
    task = _task("maze_coinflip")
    grid = _grid(task)
    xy = torch.stack([grid.centre(grid.goal_cell), grid.centre(grid.start_cell)])
    extra = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # a stale draw must not matter

    success = task.success_now(_sim_with(task, xy), extra)

    assert success.tolist() == [True, False]


def test_init_extra_draws_a_fair_coin_for_block_starts_and_flags_them() -> None:
    task = _task("maze_coinflip")
    grid = _grid(task)
    inside = _block_start(task, 10_000)
    outside = StartStates(
        grid.centre(grid.start_cell).repeat(5, 1), torch.zeros(5, 2), torch.zeros(5, 2)
    )
    gen = torch.Generator().manual_seed(0)

    drawn = task.init_extra(inside, gen)  # pyright: ignore[reportAttributeAccessIssue] # coinflip task draws extra at reset
    other = task.init_extra(outside, gen)  # pyright: ignore[reportAttributeAccessIssue]

    assert drawn.shape == (10_000, 2)
    assert set(drawn[:, 0].tolist()) == {0.0, 1.0}
    assert float(drawn[:, 0].mean()) == pytest.approx(0.5, abs=0.02)
    assert bool((drawn[:, 1] == 1.0).all())
    assert bool((other[:, 1] == 0.0).all())


def test_coinflip_rate_near_half() -> None:
    task = _task("maze_coinflip")
    gen = torch.Generator().manual_seed(1)
    env = VecEnv(
        task,
        num_envs=400,
        device="cpu",
        counter=StepCounter(budget=10_000),
        gen=gen,
    )
    env.bind(_FixedStartMethod(task, gen, _block_start(task, 1)))

    env.reset_all()

    assert float(env.extra[:, 0].mean()) == pytest.approx(0.5, abs=0.07)
    assert bool((env.extra[:, 1] == 1.0).all())


def _block_episode_success(policy_action: float) -> Tensor:
    task = _task("maze_coinflip")
    outcomes: Outcomes = run_first_episode(
        task,
        lambda _env, obs: torch.full((obs.shape[0], 2), policy_action),
        _block_start(task, 1),
        num_envs=64,
        seed=5,
    )
    return outcomes.success


@pytest.mark.slow
def test_coinflip_episode_success_is_a_fair_coin_whatever_the_policy_does() -> None:
    idle, pushing = _block_episode_success(0.0), _block_episode_success(1.0)

    assert torch.equal(idle, pushing)
    assert float(idle.float().mean()) == pytest.approx(0.5, abs=0.2)


# --- shared across variants -----------------------------------------------


@pytest.mark.parametrize("name", VARIANTS)
def test_variant_registers_with_a_diagnostic_region(name: str) -> None:
    task = _task(name)
    xy = _grid(task).centre(_grid(task).start_cell).unsqueeze(0)

    region = _region(task, xy)

    assert region.dtype == torch.bool
    assert region.shape == (1,)
    assert region.tolist() == [False]  # the task's own start is never in the region


@pytest.mark.parametrize("name", VARIANTS)
def test_mjcf_walls_match_occupancy_grid(name: str) -> None:
    task = _task(name)
    grid = _grid(task)
    model = mujoco.MjModel.from_xml_string(task.mjcf)  # pyright: ignore[reportAttributeAccessIssue] # no stub for the C-extension class
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    boxes = [
        (data.geom_xpos[g], model.geom_size[g])
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX  # pyright: ignore[reportAttributeAccessIssue]
    ]

    def covered(x: float, y: float) -> bool:
        return any(abs(x - p[0]) <= s[0] and abs(y - p[1]) <= s[1] for p, s in boxes)

    for row in range(grid.rows):
        for col in range(grid.cols):
            cx, cy = ((col + 0.5) * grid.cell_m, (row + 0.5) * grid.cell_m)
            assert covered(cx, cy) == bool(grid.occupied[row, col])


class _RowsMethod:
    """Reset rule that hands env `i` row `i % n` of `states`."""

    name = "rows"

    def __init__(self, states: StartStates) -> None:
        self._states = states

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        return self._states.index(env_ids % self._states.n), env_ids.clone()

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None:
        return None


@pytest.mark.parametrize("name", VARIANTS)
def test_is_valid_accepts_every_state_reached_pushing_into_walls_and_regions(
    name: str,
) -> None:
    task = _task(name)
    grid = _grid(task)
    gen = torch.Generator().manual_seed(0)
    # Half the envs start inside the diagnostic region (pit, pocket, block), half at the start.
    candidates = (
        torch.rand(50_000, 2, generator=gen)
        * torch.tensor([grid.cols, grid.rows])
        * grid.cell_m
    )
    keep = _region(task, candidates) & grid.has_clearance(candidates)
    region_xy = candidates[keep][:32]
    assert region_xy.shape[0] == 32
    base = task.sample_start(32, gen)
    states = StartStates(
        torch.cat([region_xy, base.qpos]),
        torch.zeros(64, 2),
        torch.zeros(64, task.extra_dim),
    )
    env = VecEnv(
        task,
        num_envs=64,
        device="cpu",
        counter=StepCounter(budget=100_000),
        gen=gen,
    )
    env.bind(_RowsMethod(states))
    env.reset_all()
    env.ep_step.zero_()
    reached_q, reached_v = [], []
    for _ in range(300):
        # Sign actions: full force in a random diagonal, pressing into walls, corners and lips.
        env.step(torch.sign(torch.randn(64, 2, generator=gen)))
        reached_q.append(env.sim.qpos.clone())
        reached_v.append(env.sim.qvel.clone())
    qpos, qvel = torch.cat(reached_q), torch.cat(reached_v)

    valid = task.is_valid(
        StartStates(qpos, qvel, torch.zeros(qpos.shape[0], task.extra_dim))
    )

    assert bool(valid.all())
    assert bool(_region(task, qpos).any())


# --- solvability from the task's own start --------------------------------

SOLVE_ENVS = 32


def _follower_success(task: Task, grid: MazeGrid | None = None) -> float:
    follower = MazePathFollower(task, grid=grid)
    outcomes = run_first_episode(
        task, lambda _env, obs: follower(obs), num_envs=SOLVE_ENVS, seed=0
    )
    return float(outcomes.success.float().mean())


@pytest.mark.parametrize("name", TRAPS)
def test_trap_sparse_solvable_routing_around_the_pit(name: str) -> None:
    task = _task(name)

    fraction = _follower_success(task, task.planning_grid)  # pyright: ignore[reportAttributeAccessIssue] # trap tasks expose a planning grid

    assert fraction >= 0.75


def test_maze_coinflip_solvable_from_the_normal_start() -> None:
    assert _follower_success(_task("maze_coinflip")) >= 0.75


def _success_from_pit_centre(name: str) -> float:
    task = _task(name)
    follower = MazePathFollower(
        task
    )  # plans on the plain grid: the pit is just floor to it
    start = _one_start(task, _pit_start(task, [0.0, 0.0], at_lip=False))
    outcomes = run_first_episode(
        task, lambda _env, obs: follower(obs), start, num_envs=8, seed=0
    )
    return float(outcomes.success.float().mean())


def test_scripted_controller_started_in_the_pit_is_stuck_only_in_the_trap() -> None:
    assert _success_from_pit_centre("maze_trap") == 0.0
    assert _success_from_pit_centre("maze_trap_twin") >= 0.75


@pytest.mark.slow
def test_maze_infeasible_solvable_from_eval_states() -> None:
    task = _task("maze_infeasible")
    follower = MazePathFollower(task)
    wins = 0
    seeds = 8
    for seed in range(seeds):
        outcomes = run_first_episode(
            task,
            lambda _env, obs: follower(obs),
            task.eval_states(1, seed=seed),
            seed=seed,
        )
        wins += int(outcomes.success.all())

    assert wins / seeds >= 0.75


# --- builder ---------------------------------------------------------------


def test_build_maze_pad_and_pockets_make_sealed_islands() -> None:
    _mjcf, grid = build_maze(
        lanes=3,
        passage_width=0.4,
        layout="horizontal",
        pad_cells=3,
        pockets=((1, 2),),
    )

    pocket = grid.flat((1, 2))
    assert not bool(grid.occupied[1, 2])
    assert int(grid.dist_cells[pocket, grid.flat(grid.start_cell)]) == -1
    assert grid.shortest_path_cells > 0


def test_build_maze_rejects_a_pocket_that_touches_the_maze() -> None:
    with pytest.raises(ValueError, match="not sealed"):
        build_maze(
            lanes=3,
            passage_width=0.4,
            layout="horizontal",
            pockets=((2, 3),),  # the wall between lane 0 and lane 1
        )


def test_build_maze_lane_rows_widens_corridors() -> None:
    _mjcf, thin = build_maze(lanes=3, passage_width=0.4, layout="horizontal")
    _mjcf, wide = build_maze(
        lanes=3, passage_width=0.4, layout="horizontal", lane_rows=3
    )

    assert wide.rows == 3 * (3 + 1) + 1
    assert wide.rows > thin.rows
    assert (
        int(wide.dist_cells[wide.flat(wide.start_cell), wide.flat(wide.goal_cell)]) > 0
    )


def test_blocked_grid_rejects_covering_the_start() -> None:
    _mjcf, grid = build_maze(lanes=3, passage_width=0.4, layout="horizontal")
    mask = torch.zeros_like(grid.occupied)
    mask[grid.start_cell] = True

    with pytest.raises(ValueError, match="start or goal"):
        grid.blocked(mask)

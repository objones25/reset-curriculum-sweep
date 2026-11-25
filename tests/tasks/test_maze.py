import mujoco
import pytest
import torch
from controllers import MazePathFollower, first_episode_success
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import TASKS, Task
from resetsweep.tasks.maze import MazeGrid, build_maze, path_distance
from resetsweep.types import StartStates

TASKS.load("resetsweep.tasks")

NAMES = ["maze", "maze_narrow", "maze_dense"]


def _task(name: str, **settings: object) -> Task:
    return TASKS.construct(name, **settings)  # pyright: ignore[reportReturnType] # registry is untyped


def _grid(task: Task) -> MazeGrid:
    return task.grid  # pyright: ignore[reportAttributeAccessIssue] # maze tasks expose their grid


def _sim_at(
    task: Task, xy: list[list[float]], vel: list[list[float]] | None = None
) -> BatchSim:
    n = len(xy)
    sim = BatchSim(task.mjcf, nworld=n, device="cpu", frame_skip=task.frame_skip)
    qvel = torch.zeros(n, 2) if vel is None else torch.tensor(vel)
    sim.set_state(torch.arange(n), torch.tensor(xy), qvel)
    return sim


def _states(xy: list[list[float]]) -> StartStates:
    n = len(xy)
    return StartStates(torch.tensor(xy), torch.zeros(n, 2), torch.zeros(n, 0))


def _centre(grid: MazeGrid, row: int, col: int) -> list[float]:
    return [(col + 0.5) * grid.cell_m, (row + 0.5) * grid.cell_m]


def test_maze_settings_are_applied() -> None:
    task = _task("maze", lanes=5, passage_width=0.5)

    grid = _grid(task)

    assert grid.cell_m == 0.5
    assert grid.occupied.shape[0] == 2 * 5 + 1


def test_maze_narrow_has_narrow_passages() -> None:
    assert _grid(_task("maze_narrow")).cell_m == 0.15


@pytest.mark.parametrize("settings", [{"lanes": 0}, {"passage_width": 0.0}])
def test_maze_rejects_degenerate_settings(settings: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _task("maze", **settings)


def test_point_cannot_cross_wall_with_full_force() -> None:
    task = _task("maze")
    grid = _grid(task)
    # Lane 0 (row 1), mid-lane; the wall above it (row 2) is solid there. Push +y for 2 s.
    start = _centre(grid, 1, 3)
    wall_face_y = 2 * grid.cell_m
    sim = _sim_at(task, [start])

    for _ in range(40):  # 40 control steps of 0.05 s
        sim.step(torch.tensor([[0.0, 1.0]]))

    assert float(sim.qpos[0, 1]) < wall_face_y
    assert float(sim.qpos[0, 0]) == pytest.approx(start[0], abs=0.05)


def test_point_cannot_cross_outer_wall_with_full_force() -> None:
    task = _task("maze")
    grid = _grid(task)
    start = _centre(grid, 1, 1)  # the start cell; the outer wall is at x < cell_m
    sim = _sim_at(task, [start])

    for _ in range(40):
        sim.step(torch.tensor([[-1.0, 0.0]]))

    assert float(sim.qpos[0, 0]) > 0.0


def test_goal_reward_only_inside_goal() -> None:
    task = _task("maze")
    grid = _grid(task)
    goal = _centre(grid, *grid.goal_cell)
    start = _centre(grid, *grid.start_cell)
    # The cell just across the wall from the goal lane is euclidean-close but not the goal.
    across = _centre(grid, grid.goal_cell[0] - 2, grid.goal_cell[1])
    sim = _sim_at(task, [goal, start, across])
    extra = torch.zeros(3, 0)

    reward = task.reward(sim, extra, torch.zeros(3, 2))

    assert reward.tolist() == [1.0, 0.0, 0.0]
    assert task.success_now(sim, extra).tolist() == [True, False, False]


def test_maze_reward_subtracts_action_cost() -> None:
    task = _task("maze")
    grid = _grid(task)
    sim = _sim_at(task, [_centre(grid, 1, 3)])

    reward = task.reward(sim, torch.zeros(1, 0), torch.tensor([[1.0, 1.0]]))

    assert float(reward[0]) == pytest.approx(-0.02)


def test_maze_never_terminates() -> None:
    task = _task("maze")
    sim = _sim_at(task, [_centre(_grid(task), 1, 3)])

    assert task.terminated(sim, torch.zeros(1, 0)).tolist() == [False]


def test_lanes_setting_changes_path_length() -> None:
    short = _grid(_task("maze", lanes=3)).shortest_path_cells
    long = _grid(_task("maze", lanes=5)).shortest_path_cells

    assert long > short


def test_more_lanes_gives_longer_episode_horizon() -> None:
    assert _task("maze", lanes=5).max_steps > _task("maze", lanes=3).max_steps


def test_is_valid_rejects_point_inside_wall() -> None:
    task = _task("maze")
    grid = _grid(task)
    in_wall = _centre(grid, 2, 3)  # separating wall between lane 0 and lane 1
    in_corridor = _centre(grid, 1, 3)

    valid = task.is_valid(_states([in_wall, in_corridor]))

    assert valid.tolist() == [False, True]


def test_is_valid_accepts_ball_touching_wall() -> None:
    task = _task("maze")
    grid = _grid(task)
    x, _y = _centre(grid, 1, 3)
    # Centre 0.01 m from the wall face: the ball overlaps the wall slightly, as a soft contact allows.
    touching_wall = [x, 2 * grid.cell_m - 0.01]

    assert task.is_valid(_states([touching_wall])).tolist() == [True]


@pytest.mark.parametrize("name", ["maze", "maze_narrow"])
def test_is_valid_accepts_every_state_reached_pushing_into_walls(name: str) -> None:
    task = _task(name)
    sim = BatchSim(task.mjcf, nworld=64, device="cpu", frame_skip=task.frame_skip)
    starts = task.sample_start(64, torch.Generator().manual_seed(0))
    sim.set_state(torch.arange(64), starts.qpos, starts.qvel)
    gen = torch.Generator().manual_seed(1)
    reached = []
    for _ in range(300):
        # Sign actions: full force in a random diagonal, pressing into walls and corners.
        sim.step(torch.sign(torch.randn(64, 2, generator=gen)))
        reached.append((sim.qpos.clone(), sim.qvel.clone()))
    qpos = torch.cat([q for q, _ in reached])
    qvel = torch.cat([v for _, v in reached])

    valid = task.is_valid(StartStates(qpos, qvel, torch.zeros(qpos.shape[0], 0)))

    assert bool(valid.all())


@pytest.mark.parametrize("name", NAMES)
def test_sample_start_keeps_ball_clear_of_walls(name: str) -> None:
    task = _task(name)

    starts = task.sample_start(256, torch.Generator().manual_seed(0))

    assert bool(_grid(task).has_clearance(starts.qpos).all())


@pytest.mark.parametrize("name", NAMES)
def test_sample_start_is_in_the_start_cell(name: str) -> None:
    task = _task(name)
    grid = _grid(task)

    starts = task.sample_start(64, torch.Generator().manual_seed(0))

    rows, cols = grid.cell_of(starts.qpos)
    assert bool((rows == grid.start_cell[0]).all())
    assert bool((cols == grid.start_cell[1]).all())


@pytest.mark.parametrize("name", NAMES)
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

    for row in range(grid.occupied.shape[0]):
        for col in range(grid.occupied.shape[1]):
            cx, cy = _centre(grid, row, col)
            assert covered(cx, cy) == bool(grid.occupied[row, col])


def test_build_maze_returns_mjcf_and_grid() -> None:
    mjcf, grid = build_maze(lanes=3, passage_width=0.4, layout="horizontal")

    assert "<mujoco" in mjcf
    assert grid.start_cell != grid.goal_cell


def test_path_distance_exceeds_euclidean_across_wall() -> None:
    task = _task("maze")
    grid = _grid(task)
    a = torch.tensor([_centre(grid, 1, 3)])  # lane 0
    b = torch.tensor([_centre(grid, 3, 3)])  # lane 1, directly across the wall

    geodesic = path_distance(task, a, b)

    assert float(geodesic[0]) > 3 * float((a - b).norm())


def test_path_distance_is_zero_for_identical_points_and_symmetric() -> None:
    task = _task("maze")
    grid = _grid(task)
    a = torch.tensor([_centre(grid, 1, 2), _centre(grid, 1, 5)])
    b = torch.tensor([_centre(grid, 3, 4), _centre(grid, 5, 7)])

    assert path_distance(task, a, a).tolist() == [0.0, 0.0]
    assert torch.equal(path_distance(task, a, b), path_distance(task, b, a))


def test_path_distance_along_a_straight_lane_is_euclidean() -> None:
    task = _task("maze")
    grid = _grid(task)
    a = torch.tensor([_centre(grid, 1, 1)])
    b = torch.tensor([_centre(grid, 1, 6)])

    assert float(path_distance(task, a, b)[0]) == pytest.approx(5 * grid.cell_m)


def test_path_distance_rejects_points_in_walls() -> None:
    task = _task("maze")
    grid = _grid(task)
    in_wall = torch.tensor([_centre(grid, 2, 3)])

    with pytest.raises(ValueError, match="inside a wall"):
        path_distance(task, in_wall, in_wall)


def _dense_reward_at(xy: list[float]) -> float:
    task = _task("maze_dense")
    sim = _sim_at(task, [xy])
    return float(task.reward(sim, torch.zeros(1, 0), torch.zeros(1, 2))[0])


def test_dense_reward_is_minus_one_at_start_and_zero_at_goal() -> None:
    grid = _grid(_task("maze_dense"))

    assert _dense_reward_at(_centre(grid, *grid.start_cell)) == pytest.approx(-1.0)
    assert _dense_reward_at(_centre(grid, *grid.goal_cell)) == pytest.approx(0.0)


def test_dense_reward_increases_along_the_path() -> None:
    grid = _grid(_task("maze_dense"))

    early = _dense_reward_at(_centre(grid, 1, 2))
    late = _dense_reward_at(_centre(grid, 1, 6))

    assert late > early


def test_dense_reward_prefers_path_progress_over_euclidean_closeness() -> None:
    # (3, 1) is euclidean-close to the start (1, 1), but along the path it is past the end of
    # lane 0, so it must score higher than both. Being near across a wall is not progress.
    grid = _grid(_task("maze_dense"))
    lane0_far_end = _dense_reward_at(_centre(grid, 1, 8))  # end of lane 0, gap above it
    lane1_start = _dense_reward_at(
        _centre(grid, 3, 1)
    )  # far end of lane 1, directly across the wall from the start
    lane0_start = _dense_reward_at(_centre(grid, 1, 1))

    assert lane0_start < lane0_far_end < lane1_start


def test_dense_reward_finite_when_centre_in_wall_cell() -> None:
    task = _task("maze_dense")
    grid = _grid(task)
    in_wall = _centre(grid, 2, 3)
    off_grid = [-1.0, 99.0]
    sim = _sim_at(task, [in_wall, off_grid])

    reward = task.reward(sim, torch.zeros(2, 0), torch.zeros(2, 2))

    assert bool(torch.isfinite(reward).all())
    assert bool((reward <= 0).all())


def test_sparse_and_dense_share_geometry() -> None:
    assert _task("maze").mjcf == _task("maze_dense").mjcf


@pytest.mark.parametrize("name", NAMES)
def test_dense_sibling_is_maze_dense(name: str) -> None:
    assert _task(name).dense_sibling == "maze_dense"


def _solve_fraction(name: str, n: int) -> float:
    task = _task(name)
    controller = MazePathFollower(task)

    success: Tensor = first_episode_success(
        task, lambda _env, obs: controller(obs), n, seed=0
    )

    return float(success.float().mean())


def test_maze_sparse_solvable_by_scripted_controller() -> None:
    # Without this, a maze nothing can solve would pass every constraint test above.
    assert _solve_fraction("maze", 32) >= 0.75


def test_maze_narrow_sparse_solvable_by_scripted_controller() -> None:
    assert _solve_fraction("maze_narrow", 32) >= 0.75


def test_zero_action_never_solves_maze() -> None:
    task = _task("maze")

    success = first_episode_success(
        task, lambda _env, obs: torch.zeros(obs.shape[0], 2), 8, seed=0
    )

    assert not bool(success.any())

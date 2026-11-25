"""Point-mass serpentine maze (T1): sparse goal reward, narrow variant and a dense sibling.

The maze is one occupancy grid. `build_maze` generates the MJCF walls and the `MazeGrid` from
it, so the physics, `is_valid` and `path_distance` can never disagree about where walls are.
Grid cells are square with side `passage_width`: corridors and walls are one cell wide.
Row index grows with y, column index with x. `qpos = (x, y)`, in metres.
"""

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import TASKS
from resetsweep.types import Box, StartStates

Layout = Literal["horizontal", "vertical"]

# Point mass 1 kg with joint damping 2 N*s/m and a motor limited to 4 N: terminal speed is
# 4 / 2 = 2 m/s and the velocity time constant is m / damping = 0.5 s, so the point cannot
# stop on a dime. timestep 0.01 s with frame_skip 5 gives the 0.05 s control step.
FORCE_LIMIT_N = 4.0
_DAMPING_N_S_M = 2.0
MAX_SPEED_M_S = FORCE_LIMIT_N / _DAMPING_N_S_M
_CONTROL_DT_S = 0.05
_FRAME_SKIP = 5
_HOLD_STEPS = 10  # 0.5 s in the goal cell
_LANE_CELLS = 8  # corridor length of each lane, in cells
_BALL_RADIUS_CELLS = (
    0.3  # of the passage width, so the ball is 60% as wide as the corridor
)
_START_NOISE_CELLS = 0.15  # stays clear of the walls: 0.5 - 0.15 = 0.35 > 0.3
_START_SPEED_NOISE_M_S = 0.05
_ACTION_COST = 0.01
# Horizon: 2x the time to walk the shortest path at 1 m/s. Longer mazes get longer episodes.
_REF_SPEED_M_S = 1.0
_HORIZON_SLACK = 2.0


@dataclass(frozen=True)
class MazeGrid:
    """Occupancy grid and its geodesic distances.

    `occupied[row, col]` is True for wall cells. `dist_cells[a, b]` is the 4-connected shortest
    path length in cells between free cells with flat index `row * cols + col`, and -1 if either
    index is a wall.
    """

    occupied: Tensor
    cell_m: float
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    dist_cells: Tensor

    @property
    def rows(self) -> int:
        return self.occupied.shape[0]

    @property
    def cols(self) -> int:
        return self.occupied.shape[1]

    @property
    def ball_radius_m(self) -> float:
        return _BALL_RADIUS_CELLS * self.cell_m

    @property
    def shortest_path_cells(self) -> int:
        return int(
            self.dist_cells[self.flat(self.start_cell), self.flat(self.goal_cell)]
        )

    def flat(self, cell: tuple[int, int]) -> int:
        return cell[0] * self.cols + cell[1]

    def centre(self, cell: tuple[int, int]) -> Tensor:
        return (
            torch.tensor([cell[1], cell[0]], dtype=torch.float32) + 0.5
        ) * self.cell_m

    def cell_of(self, xy: Tensor) -> tuple[Tensor, Tensor]:
        """`(row, col)` of each point, clamped into the grid (callers check `in_free_cell`)."""
        col = (xy[:, 0] / self.cell_m).floor().long().clamp(0, self.cols - 1)
        row = (xy[:, 1] / self.cell_m).floor().long().clamp(0, self.rows - 1)
        return row, col

    def in_free_cell(self, xy: Tensor) -> Tensor:
        """True where `xy` is finite, inside the grid and its centre is in a free cell.

        This is the reachable-state test: soft contacts let the ball press slightly into a wall,
        so states with the ball touching (even a little overlapping) a wall are reachable.
        """
        finite = torch.isfinite(xy).all(dim=1)
        xy = torch.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        inside = (
            (xy[:, 0] >= 0)
            & (xy[:, 1] >= 0)
            & (xy[:, 0] < self.cols * self.cell_m)
            & (xy[:, 1] < self.rows * self.cell_m)
        )
        row, col = self.cell_of(xy)
        return finite & inside & ~self.occupied[row, col]

    def has_clearance(self, xy: Tensor) -> Tensor:
        """True where a ball of the maze's radius at `xy` touches no wall and is inside the grid.

        Stricter than `in_free_cell`; for spawning starts, not for judging reached states."""
        finite = torch.isfinite(xy).all(dim=1)
        xy = torch.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        inside = (
            (xy[:, 0] >= 0)
            & (xy[:, 1] >= 0)
            & (xy[:, 0] < self.cols * self.cell_m)
            & (xy[:, 1] < self.rows * self.cell_m)
        )
        row, col = self.cell_of(xy)
        ok = finite & inside & ~self.occupied[row, col]
        # A ball in a free cell can still overlap a wall cell in the 3x3 neighbourhood.
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                n_row, n_col = row + d_row, col + d_col
                in_grid = (
                    (n_row >= 0)
                    & (n_row < self.rows)
                    & (n_col >= 0)
                    & (n_col < self.cols)
                )
                wall = (
                    in_grid
                    & self.occupied[
                        n_row.clamp(0, self.rows - 1), n_col.clamp(0, self.cols - 1)
                    ]
                )
                lo_x, lo_y = n_col * self.cell_m, n_row * self.cell_m
                gap_x = torch.maximum(
                    lo_x - xy[:, 0], xy[:, 0] - (lo_x + self.cell_m)
                ).clamp(min=0)
                gap_y = torch.maximum(
                    lo_y - xy[:, 1], xy[:, 1] - (lo_y + self.cell_m)
                ).clamp(min=0)
                clear = (
                    torch.sqrt(gap_x.square() + gap_y.square()) >= self.ball_radius_m
                )
                ok &= ~wall | clear
        return ok

    def cell_index(self, xy: Tensor) -> Tensor:
        row, col = self.cell_of(xy)
        return row * self.cols + col

    def blocked(self, mask: Tensor) -> "MazeGrid":
        """Copy of this grid with the cells in `mask` (bool, same shape as `occupied`) treated as
        walls, distances recomputed. For planners that must route around a region the physics
        still lets the ball enter (a pit). Start and goal must stay free."""
        if mask.shape != self.occupied.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)}, expected {tuple(self.occupied.shape)}"
            )
        occupied = self.occupied | mask
        if bool(occupied[self.start_cell]) or bool(occupied[self.goal_cell]):
            raise ValueError("blocked cells must not cover the start or goal cell")
        return replace(self, occupied=occupied, dist_cells=_distances(occupied))


def _horizontal_occupancy(
    lanes: int, lane_rows: int
) -> tuple[list[list[bool]], tuple[int, int], tuple[int, int]]:
    """Serpentine with horizontal lanes `lane_rows` cells thick, one wall row between them, joined
    by a one-cell gap in that wall. Lane i starts at row `i * (lane_rows + 1) + 1`."""
    pitch = lane_rows + 1
    rows, cols = lanes * pitch + 1, _LANE_CELLS + 2
    occupied = [[True] * cols for _ in range(rows)]
    for lane in range(lanes):
        for row in range(lane * pitch + 1, lane * pitch + 1 + lane_rows):
            for col in range(1, _LANE_CELLS + 1):
                occupied[row][col] = False
    for lane in range(1, lanes):
        gap_col = _LANE_CELLS if lane % 2 == 1 else 1
        occupied[lane * pitch][gap_col] = False
    goal_col = _LANE_CELLS if (lanes - 1) % 2 == 0 else 1
    return occupied, (1, 1), ((lanes - 1) * pitch + 1, goal_col)


def _distances(occupied: Tensor) -> Tensor:
    """All-pairs 4-connected BFS distances in cells; -1 for walls and unreachable pairs."""
    rows, cols = occupied.shape
    free = [(r, c) for r in range(rows) for c in range(cols) if not occupied[r, c]]
    dist = torch.full((rows * cols, rows * cols), -1, dtype=torch.long)
    for source in free:
        dist[source[0] * cols + source[1], source[0] * cols + source[1]] = 0
        queue = deque([source])
        steps = {source: 0}
        while queue:
            r, c = queue.popleft()
            for n_r, n_c in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if (
                    0 <= n_r < rows
                    and 0 <= n_c < cols
                    and not occupied[n_r, n_c]
                    and (n_r, n_c) not in steps
                ):
                    steps[(n_r, n_c)] = steps[(r, c)] + 1
                    dist[source[0] * cols + source[1], n_r * cols + n_c] = (
                        steps[(r, c)] + 1
                    )
                    queue.append((n_r, n_c))
    return dist


def _wall_boxes(occupied: Tensor) -> list[tuple[int, int, int, int]]:
    """Cover the wall cells with few boxes: row runs, merged with identical runs in the next row.
    Returns `(row_lo, row_hi, col_lo, col_hi)`, inclusive."""
    rows, cols = occupied.shape
    open_boxes: dict[
        tuple[int, int], int
    ] = {}  # (col_lo, col_hi) -> row_lo of the open box
    boxes: list[tuple[int, int, int, int]] = []
    for row in range(rows + 1):
        runs: list[tuple[int, int]] = []
        if row < rows:
            col = 0
            while col < cols:
                if occupied[row, col]:
                    end = col
                    while end + 1 < cols and occupied[row, end + 1]:
                        end += 1
                    runs.append((col, end))
                    col = end + 1
                else:
                    col += 1
        for key in [k for k in open_boxes if k not in runs]:
            boxes.append((open_boxes.pop(key), row - 1, key[0], key[1]))
        for run in runs:
            open_boxes.setdefault(run, row)
    return boxes


def build_maze(
    *,
    lanes: int,
    passage_width: float,
    layout: Layout,
    lane_rows: int = 1,
    pad_cells: int = 0,
    pockets: tuple[tuple[int, int], ...] = (),
    ctrl_limit: float = 1.0,
) -> tuple[str, MazeGrid]:
    """MJCF and occupancy grid of a serpentine maze with `lanes` lanes.

    `layout="horizontal"` runs lanes along x; `"vertical"` is the transpose (up, down, up: an N
    for three lanes). The start is the first lane's near end, the goal the last lane's far end.
    `lane_rows` is the lane thickness in cells, `pad_cells` extra wall cells all around, and
    `pockets` are `(row, col)` wall cells of the final grid turned into free cells. Each must
    be sealed off from the maze: the grid then has islands with no path to the goal.
    `ctrl_limit` is the motors' control range (1 unless a task adds force through `control`).
    """
    if lanes <= 0 or passage_width <= 0 or lane_rows <= 0 or pad_cells < 0:
        raise ValueError(
            "lanes, passage_width and lane_rows must be positive and pad_cells non-negative, "
            f"got {lanes}, {passage_width}, {lane_rows}, {pad_cells}"
        )
    cells, start, goal = _horizontal_occupancy(lanes, lane_rows)
    occupied = torch.tensor(cells, dtype=torch.bool)
    if layout == "vertical":
        occupied = occupied.T.contiguous()
        start, goal = (start[1], start[0]), (goal[1], goal[0])
    elif layout != "horizontal":
        raise ValueError(f"layout must be 'horizontal' or 'vertical', got {layout!r}")
    if pad_cells:
        padded = torch.ones(
            occupied.shape[0] + 2 * pad_cells,
            occupied.shape[1] + 2 * pad_cells,
            dtype=torch.bool,
        )
        padded[pad_cells:-pad_cells, pad_cells:-pad_cells] = occupied
        occupied = padded
        start = (start[0] + pad_cells, start[1] + pad_cells)
        goal = (goal[0] + pad_cells, goal[1] + pad_cells)
    for row, col in pockets:
        if not (0 <= row < occupied.shape[0] and 0 <= col < occupied.shape[1]):
            raise ValueError(f"pocket {(row, col)} is outside the grid")
        if not bool(occupied[row, col]):
            raise ValueError(f"pocket {(row, col)} is not a wall cell")
        occupied[row, col] = False
    dist = _distances(occupied)
    _check_connectivity(occupied, dist, start, pockets)
    grid = MazeGrid(occupied, passage_width, start, goal, dist)
    return _mjcf(grid, ctrl_limit), grid


def _check_connectivity(
    occupied: Tensor,
    dist: Tensor,
    start: tuple[int, int],
    pockets: tuple[tuple[int, int], ...],
) -> None:
    """Every free cell is reachable from the start, except the pockets, which none may be."""
    rows, cols = occupied.shape
    reachable = dist[start[0] * cols + start[1]] >= 0
    pocket_flat = torch.zeros(rows * cols, dtype=torch.bool)
    for row, col in pockets:
        pocket_flat[row * cols + col] = True
    if bool((reachable & pocket_flat).any()):
        raise ValueError("a pocket is not sealed off from the maze")
    if bool(((~occupied).flatten() & ~reachable & ~pocket_flat).any()):
        raise ValueError("maze has disconnected free cells")


def _mjcf(grid: MazeGrid, ctrl_limit: float) -> str:
    cell = grid.cell_m
    half_height = cell
    walls = []
    for i, (r0, r1, c0, c1) in enumerate(_wall_boxes(grid.occupied)):
        cx, cy = (c0 + c1 + 1) / 2 * cell, (r0 + r1 + 1) / 2 * cell
        hx, hy = (c1 - c0 + 1) / 2 * cell, (r1 - r0 + 1) / 2 * cell
        walls.append(
            f'<geom name="wall{i}" type="box" pos="{cx:.6f} {cy:.6f} 0" '
            f'size="{hx:.6f} {hy:.6f} {half_height:.6f}" condim="1"/>'
        )
    return f"""
<mujoco model="maze">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    {chr(10).join("    " + w for w in walls).strip()}
    <body name="ball" pos="0 0 0">
      <joint name="x" type="slide" axis="1 0 0" damping="{_DAMPING_N_S_M}"/>
      <joint name="y" type="slide" axis="0 1 0" damping="{_DAMPING_N_S_M}"/>
      <geom name="ball" type="sphere" size="{grid.ball_radius_m:.6f}" mass="1" condim="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="x" gear="{FORCE_LIMIT_N}" ctrllimited="true" ctrlrange="-{ctrl_limit} {ctrl_limit}"/>
    <motor joint="y" gear="{FORCE_LIMIT_N}" ctrllimited="true" ctrlrange="-{ctrl_limit} {ctrl_limit}"/>
  </actuator>
</mujoco>
"""


def path_distance(task: "_Maze", xy_a: Tensor, xy_b: Tensor) -> Tensor:
    """Geodesic distance in metres between paired points `(n, 2)`, through the maze's free cells.

    It is the 4-connected grid path length between the points' cells, and never less than the
    straight-line distance (so two points in one cell get their Euclidean distance).
    """
    grid = task.grid
    if xy_a.shape != xy_b.shape or xy_a.ndim != 2 or xy_a.shape[1] != 2:
        raise ValueError(
            f"need two (n, 2) tensors, got {tuple(xy_a.shape)}, {tuple(xy_b.shape)}"
        )
    for xy in (xy_a, xy_b):
        row, col = grid.cell_of(xy)
        if bool(grid.occupied[row, col].any()):
            raise ValueError("path_distance got a point inside a wall")
    steps = grid.dist_cells[grid.cell_index(xy_a), grid.cell_index(xy_b)]
    return torch.maximum(steps.float() * grid.cell_m, (xy_a - xy_b).norm(dim=1))


class _Maze:
    obs_dim = 4  # x, y, vx, vy
    act_dim = 2
    frame_skip = _FRAME_SKIP
    hold_steps = _HOLD_STEPS
    dense_sibling = "maze_dense"
    extra_dim = 0
    name: str

    def __init__(
        self,
        lanes: int = 3,
        passage_width: float = 0.4,
        layout: Layout = "horizontal",
        **build: Any,
    ) -> None:
        self.mjcf, self.grid = build_maze(
            lanes=lanes, passage_width=passage_width, layout=layout, **build
        )
        path_m = self.grid.shortest_path_cells * passage_width
        self.max_steps = math.ceil(
            _HORIZON_SLACK * path_m / (_REF_SPEED_M_S * _CONTROL_DT_S)
        )
        extent = (
            torch.tensor([self.grid.cols, self.grid.rows], dtype=torch.float32)
            * passage_width
        )
        # Margin over the fastest reachable speed: terminal speed at the largest control.
        speed = torch.full((2,), MAX_SPEED_M_S * build.get("ctrl_limit", 1.0) * 1.25)
        self.start_box = Box(
            torch.cat([torch.zeros(2), -speed]), torch.cat([extent, speed])
        )
        self._goal_cell = self.grid.flat(self.grid.goal_cell)

    def sample_start(self, n: int, gen: torch.Generator) -> StartStates:
        return self._with_noise(
            self.grid.centre(self.grid.start_cell).expand(n, 2), gen
        )

    def _with_noise(self, centres: Tensor, gen: torch.Generator) -> StartStates:
        """Start states at `centres` (n, 2), with small position and velocity noise."""
        n = centres.shape[0]
        pos_noise = (
            (torch.rand(n, 2, generator=gen) * 2 - 1)
            * _START_NOISE_CELLS
            * self.grid.cell_m
        )
        vel_noise = (torch.rand(n, 2, generator=gen) * 2 - 1) * _START_SPEED_NOISE_M_S
        return StartStates(
            centres + pos_noise, vel_noise, torch.zeros(n, self.extra_dim)
        )

    def canonical_start(self) -> StartStates:
        return StartStates(
            self.grid.centre(self.grid.start_cell).unsqueeze(0),
            torch.zeros(1, 2),
            torch.zeros(1, 0),
        )

    def observe(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return torch.cat([sim.qpos, sim.qvel], dim=1)

    def terminated(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return torch.zeros(sim.qpos.shape[0], dtype=torch.bool, device=sim.qpos.device)

    def success_now(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return self._in_goal(sim.qpos)

    def state_features(self, qpos: Tensor, qvel: Tensor, extra: Tensor) -> Tensor:
        return torch.cat([qpos, qvel], dim=1)

    def is_valid(self, states: StartStates) -> Tensor:
        x = torch.cat([states.qpos, states.qvel], dim=1)
        low = self.start_box.low.to(x.device)
        high = self.start_box.high.to(x.device)
        in_box = ((x >= low) & (x <= high)).all(dim=1) & torch.isfinite(x).all(dim=1)
        return in_box & self.grid.in_free_cell(states.qpos.cpu()).to(x.device)

    def eval_states(self, n: int, seed: int) -> StartStates:
        # A local generator: never touches the global torch RNG.
        return self.sample_start(n, torch.Generator().manual_seed(seed))

    def _in_goal(self, xy: Tensor) -> Tensor:
        cell = self.grid.cell_index(xy)  # works on any device; clamped into the grid
        extent = torch.tensor(
            [self.grid.cols, self.grid.rows], dtype=xy.dtype, device=xy.device
        )
        inside = ((xy >= 0) & (xy < extent * self.grid.cell_m)).all(dim=1)
        return inside & (cell == self._goal_cell)

    @staticmethod
    def _action_cost(action: Tensor) -> Tensor:
        return _ACTION_COST * action.square().sum(dim=1)


@TASKS.register("maze")
class Maze(_Maze):
    name = "maze"

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        return self.success_now(sim, extra).float() - self._action_cost(action)


@TASKS.register("maze_narrow")
class MazeNarrow(Maze):
    """Narrow N-shaped maze: three vertical lanes (up, down, up) with 0.15 m passages."""

    name = "maze_narrow"

    def __init__(self) -> None:
        super().__init__(lanes=3, passage_width=0.15, layout="vertical")


@TASKS.register("maze_dense")
class MazeDense(_Maze):
    """`maze` plus minus the normalised geodesic distance to the goal (-1 at the start, 0 at the goal)."""

    name = "maze_dense"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        grid = self.grid
        free = (~grid.occupied).flatten().nonzero().squeeze(1)
        centres = torch.stack(
            [
                (torch.arange(grid.rows * grid.cols) % grid.cols + 0.5) * grid.cell_m,
                (torch.arange(grid.rows * grid.cols) // grid.cols + 0.5) * grid.cell_m,
            ],
            dim=1,
        )
        # A wall cell takes the goal distance of the nearest free cell, so a ball pressed into a
        # wall by a soft contact (or off the grid, once clamped) still gets a finite reward.
        nearest = free[torch.cdist(centres, centres[free]).argmin(dim=1)]
        goal_steps = grid.dist_cells[:, grid.flat(grid.goal_cell)][nearest]
        self._to_goal_m = goal_steps.float() * grid.cell_m
        self._goal_xy = grid.centre(grid.goal_cell)
        self._total_m = grid.shortest_path_cells * grid.cell_m

    def reward(self, sim: BatchSim, extra: Tensor, action: Tensor) -> Tensor:
        xy = sim.qpos
        cell = self.grid.cell_index(xy)  # clamped into the grid
        table = self._to_goal_m.to(xy.device)
        euclid = (xy - self._goal_xy.to(xy.device)).norm(dim=1)
        to_go = torch.maximum(table[cell], euclid)
        return -to_go / self._total_m - self._action_cost(action)

"""Maze variants for the reset-curriculum sweep (spec §5.3 T3-T5).

- `maze_trap` / `maze_trap_twin` (T3): a pit in the first lane that pulls the point back harder
  than its motors push, and a twin with the same MJCF and walls whose motors are strong enough
  inside the pit to climb out.
- `maze_infeasible` (T4): start mass split as in ToyLevels: 90% easy (near the goal), 0.01% hard
  but feasible (the maze start), 9.99% infeasible (sealed pockets with no path to the goal).
- `maze_coinflip` (T5): a block of cells where success is a coin flip made at reset.

Each exports `diagnostic_region(qpos) -> Tensor[bool]` (inside the pit, a pocket, the block)
for the wasted-resets metric. All lookups are precomputed tensors indexed on the input's device.
"""

import math

import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.tasks.base import TASKS
from resetsweep.tasks.maze import LANE_CELLS, Maze, MazeGrid
from resetsweep.types import StartStates


class _CellMask:
    """A bool mask over grid cells, looked up by position on the position's device."""

    def __init__(self, grid: MazeGrid, mask: Tensor) -> None:
        if mask.shape != grid.occupied.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)}, expected {tuple(grid.occupied.shape)}"
            )
        self._grid = grid
        self._flat = mask.flatten()
        self._on_device: dict[torch.device, Tensor] = {}

    def __call__(self, xy: Tensor) -> Tensor:
        table = self._on_device.get(xy.device)
        if table is None:
            table = self._on_device[xy.device] = self._flat.to(xy.device)
        return table[self._grid.cell_index(xy)]


def _rect_mask(grid: MazeGrid, rect: tuple[int, int, int, int]) -> Tensor:
    row_lo, row_hi, col_lo, col_hi = rect
    mask = torch.zeros_like(grid.occupied)
    mask[row_lo : row_hi + 1, col_lo : col_hi + 1] = True
    return mask


# --- T3: trap and twin -----------------------------------------------------

# Inside the pit the sim gets an applied force: a constant-magnitude pull toward the pit centre
# (ramped over 5 cm so the point can rest there) and linear drag. The motors give 4 N per axis,
# at most 5.7 N along a diagonal, so against the trap's 6 N pull nothing escapes from rest. The
# pull alone is conservative, though: a controller can pump energy by pushing along its velocity
# and orbit out (a circular orbit at r = 0.4 m needs 1.55 m/s). Drag kills that: sustaining speed
# v costs 10 v newtons, so the motors can hold at most about 0.57 m/s, and an orbit of radius r
# needs sqrt(6 r) m/s, which only exists for r below about 5 cm, inside the pull's ramp.
# The twin has the same walls, MJCF, pit and drag but a 2 N pull, which the motors beat (net 2 N
# outward from rest, terminal speed 0.2 m/s).
_PIT_RAMP_M = 0.05
_PIT_DRAG_N_S_M = 10.0
_TRAP_PULL_N = 6.0
_TWIN_PULL_N = 2.0
_TRAP_LANE_ROWS = (
    5  # wide lanes: a pit up to 3 cells across leaves a free row on each side
)
_PIT_BASE_CELLS = 2  # pit side at trap_scale 1
_PIT_SIDE_CELLS_MAX = 3


class _MazeTrap(Maze):
    """First lane holds a square pit at its middle; `trap_scale` scales its side length."""

    pit_pull_n: float

    def __init__(self, trap_scale: float = 1.0) -> None:
        if trap_scale <= 0:
            raise ValueError(f"trap_scale must be positive, got {trap_scale}")
        super().__init__(
            lanes=3,
            passage_width=0.4,
            layout="horizontal",
            lane_rows=_TRAP_LANE_ROWS,
        )
        side = min(
            max(math.floor(_PIT_BASE_CELLS * trap_scale + 0.5), 1), _PIT_SIDE_CELLS_MAX
        )
        row_lo = 1 + (_TRAP_LANE_ROWS - side) // 2
        col_lo = 1 + (LANE_CELLS - side) // 2
        self.pit_rect = (row_lo, row_lo + side - 1, col_lo, col_lo + side - 1)
        pit = _rect_mask(self.grid, self.pit_rect)
        self._in_pit = _CellMask(self.grid, pit)
        self.planning_grid = self.grid.blocked(pit)
        self._centre_xy = (
            torch.tensor([col_lo + side / 2, row_lo + side / 2]) * self.grid.cell_m
        )
        self._centre_on_device: dict[torch.device, Tensor] = {}

    def diagnostic_region(self, qpos: Tensor) -> Tensor:
        return self._in_pit(qpos)

    def applied_force(self, sim: BatchSim, extra: Tensor) -> Tensor:
        """Generalized force (N, the joints are slides): pull toward the pit centre minus drag,
        inside the pit; zero elsewhere."""
        xy = sim.qpos
        inside = self._in_pit(xy).unsqueeze(1)
        offset = self._centre(xy.device) - xy
        dist = offset.norm(dim=1, keepdim=True)
        pull = (
            self.pit_pull_n
            * (dist / _PIT_RAMP_M).clamp(max=1.0)
            * offset
            / dist.clamp(min=1e-9)
        )
        return torch.where(inside, pull - _PIT_DRAG_N_S_M * sim.qvel, 0.0)

    def _centre(self, device: torch.device) -> Tensor:
        cached = self._centre_on_device.get(device)
        if cached is None:
            cached = self._centre_on_device[device] = self._centre_xy.to(device)
        return cached


@TASKS.register("maze_trap")
class MazeTrap(_MazeTrap):
    """Pit the point cannot climb out of: full motor force (4 N) loses to the 6 N pull."""

    name = "maze_trap"
    pit_pull_n = _TRAP_PULL_N


@TASKS.register("maze_trap_twin")
class MazeTrapTwin(_MazeTrap):
    """`maze_trap` with identical MJCF, walls and pit, but a 2 N pull the motors can beat."""

    name = "maze_trap_twin"
    pit_pull_n = _TWIN_PULL_N


# --- T4: infeasible mass ---------------------------------------------------

_P_EASY = 0.90
_P_HARD = 0.0001  # the infeasible mass is the rest: 0.0999
_EASY_RADIUS_CELLS = 4
_POCKET_PAD_CELLS = 3  # pad of 3: wall, pocket, wall between the pocket and the maze
_POCKET_COLS = (5, 8, 11)


class MazeInfeasible(Maze):
    """`maze` padded with sealed one-cell pockets; `sample_start` mixes easy, hard and infeasible.

    Easy starts are within `easy_radius_cells` of the goal along the path, hard ones are the
    maze start, infeasible ones are in a pocket. `eval_states` draws only the maze start.
    """

    name = "maze_infeasible"
    easy_radius_cells = _EASY_RADIUS_CELLS

    def __init__(self) -> None:
        rows = 3 * 2 + 1 + 2 * _POCKET_PAD_CELLS
        pockets = tuple((row, col) for row in (1, rows - 2) for col in _POCKET_COLS)
        super().__init__(
            lanes=3,
            passage_width=0.4,
            layout="horizontal",
            pad_cells=_POCKET_PAD_CELLS,
            pockets=pockets,
        )
        grid = self.grid
        to_goal = grid.dist_cells[:, grid.flat(grid.goal_cell)]
        self._easy_centres = self._centres(
            (to_goal >= 0) & (to_goal <= _EASY_RADIUS_CELLS)
        )
        self._pocket_mask = (to_goal == -1) & ~grid.occupied.flatten()
        self._pocket_centres = self._centres(self._pocket_mask)
        self._in_pocket = _CellMask(
            grid, self._pocket_mask.reshape(grid.occupied.shape)
        )

    def _centres(self, flat_mask: Tensor) -> Tensor:
        cells = flat_mask.nonzero().squeeze(1)
        return torch.stack(
            [
                self.grid.centre((int(c) // self.grid.cols, int(c) % self.grid.cols))
                for c in cells
            ]
        )

    def sample_start(self, n: int, gen: torch.Generator) -> StartStates:
        u = torch.rand(n, generator=gen)
        easy_pick = torch.randint(self._easy_centres.shape[0], (n,), generator=gen)
        pocket_pick = torch.randint(self._pocket_centres.shape[0], (n,), generator=gen)
        centres = torch.where(
            (u < _P_EASY).unsqueeze(1),
            self._easy_centres[easy_pick],
            torch.where(
                (u >= 1.0 - _P_HARD).unsqueeze(1),
                self.grid.centre(self.grid.start_cell).expand(n, 2),
                self._pocket_centres[pocket_pick],
            ),
        )
        return self._with_noise(centres, gen)

    def eval_states(self, n: int, seed: int) -> StartStates:
        gen = torch.Generator().manual_seed(seed)
        return self._with_noise(
            self.grid.centre(self.grid.start_cell).expand(n, 2), gen
        )

    def diagnostic_region(self, qpos: Tensor) -> Tensor:
        return self._in_pocket(qpos)


TASKS.register("maze_infeasible")(MazeInfeasible)


# --- T5: coin flip ---------------------------------------------------------

_BLOCK_RECT = (3, 3, 3, 6)  # row 3 is lane 1 of the default maze; columns 3 to 6
_P_COIN = 0.5


class MazeCoinflip(Maze):
    """`maze` plus a block of cells in lane 1 where success is a coin flip.

    `extra = (draw, started_in_block)`. `init_extra` makes the draw at every episode start. An
    episode that started in the block has `success_now == draw` whatever the policy does; any
    other episode succeeds by reaching the goal. The block is in lane 1, which the shortest path
    crosses, but only where an episode starts matters, and the task's own start is outside it.
    """

    name = "maze_coinflip"
    extra_dim = 2
    block_rect = _BLOCK_RECT

    def __init__(self) -> None:
        super().__init__()
        block = _rect_mask(self.grid, _BLOCK_RECT)
        if bool((block & self.grid.occupied).any()):
            raise ValueError("coin-flip block covers wall cells")
        self._in_block = _CellMask(self.grid, block)

    def sample_start(self, n: int, gen: torch.Generator) -> StartStates:
        base = super().sample_start(n, gen)
        return StartStates(base.qpos, base.qvel, self.init_extra(base, gen))

    def canonical_start(self) -> StartStates:
        # Draw 0: `init_extra` makes the real draw at every episode start.
        base = super().canonical_start()
        started_in_block = self._in_block(base.qpos).float().unsqueeze(1)
        extra = torch.cat([torch.zeros_like(started_in_block), started_in_block], dim=1)
        return StartStates(base.qpos, base.qvel, extra)

    def init_extra(self, states: StartStates, gen: torch.Generator) -> Tensor:
        draw = (torch.rand(states.n, generator=gen, device=gen.device) < _P_COIN).to(
            states.qpos.device
        )
        started_in_block = self._in_block(states.qpos)
        return torch.stack([draw.float(), started_in_block.float()], dim=1)

    def success_now(self, sim: BatchSim, extra: Tensor) -> Tensor:
        return torch.where(
            extra[:, 1] > 0.5, extra[:, 0] > 0.5, self._in_goal(sim.qpos)
        )

    def diagnostic_region(self, qpos: Tensor) -> Tensor:
        return self._in_block(qpos)


TASKS.register("maze_coinflip")(MazeCoinflip)

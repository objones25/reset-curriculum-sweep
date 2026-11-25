import pytest
import torch
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.sim.rollouts import Rollouts
from resetsweep.tasks.base import TASKS, Task
from resetsweep.types import StartStates, StepCounter

TASKS.load("resetsweep.tasks")


def _rollouts(task: Task, *, nworld_max: int = 16) -> tuple[Rollouts, StepCounter]:
    counter = StepCounter(budget=10**9)
    gen = torch.Generator().manual_seed(0)
    return Rollouts(
        task, device="cpu", counter=counter, gen=gen, nworld_max=nworld_max
    ), counter


def test_random_walk_charges_counter_extra_walk() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped
    rollouts, counter = _rollouts(task, nworld_max=4)
    starts = task.sample_start(10, torch.Generator().manual_seed(1))  # 3 chunks of <= 4

    out = rollouts.random_walk(starts, steps=6, action_hold=3)

    assert out.n == 10
    assert counter.extra["walk"] == 10 * 6
    assert counter.train == 0
    assert not torch.equal(out.qpos, starts.qpos)


def test_random_walk_rejects_bad_arguments() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped
    rollouts, _ = _rollouts(task)
    starts = task.canonical_start()

    with pytest.raises(ValueError, match="steps"):
        rollouts.random_walk(starts, steps=0, action_hold=1)
    with pytest.raises(ValueError, match="action_hold"):
        rollouts.random_walk(starts, steps=4, action_hold=0)


def test_random_walk_does_not_mutate_its_input() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped
    rollouts, _ = _rollouts(task)
    starts = task.sample_start(4, torch.Generator().manual_seed(1))
    before = starts.qpos.clone()

    rollouts.random_walk(starts, steps=5, action_hold=5)

    assert torch.equal(starts.qpos, before)


def test_random_walk_from_inside_the_trap_never_leaves_it() -> None:
    task: Task = TASKS.construct("maze_trap")  # pyright: ignore[reportAssignmentType] # registry is untyped
    rollouts, _ = _rollouts(task, nworld_max=32)
    grid = task.grid  # pyright: ignore[reportAttributeAccessIssue] # maze tasks expose their grid
    row_lo, row_hi, col_lo, col_hi = task.pit_rect  # pyright: ignore[reportAttributeAccessIssue] # trap tasks expose their pit
    half = grid.cell_m * (row_hi - row_lo + 1) / 2
    centre = torch.tensor(
        [
            [
                (col_lo + col_hi + 1) / 2 * grid.cell_m,
                (row_lo + row_hi + 1) / 2 * grid.cell_m,
            ]
        ]
    )
    # Start near each of the four lips, 0.02 m inside, where an unforced walk escapes fastest.
    offsets = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    qpos = (centre + offsets * (half - 0.02)).repeat(8, 1)
    starts = StartStates(qpos, torch.zeros_like(qpos), torch.zeros(qpos.shape[0], 0))
    assert bool(task.diagnostic_region(starts.qpos).all())  # pyright: ignore[reportAttributeAccessIssue] # variants export it

    out = rollouts.random_walk(starts, steps=400, action_hold=5)

    assert bool(task.diagnostic_region(out.qpos).all())  # pyright: ignore[reportAttributeAccessIssue] # variants export it


def test_hold_screen_rejects_falling_state() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped
    rollouts, counter = _rollouts(task)
    # Row 0 hangs at rest. Row 1 is nearly upright, so with zero torque it falls and swings fast.
    starts = StartStates(
        torch.tensor([[0.0], [3.0]]), torch.zeros(2, 1), torch.zeros(2, 0)
    )

    keep = rollouts.hold_screen(starts, seconds=2.0)

    assert keep.tolist() == [True, False]
    assert keep.dtype == torch.bool
    assert (
        counter.extra["screen"] == 2 * 40
    )  # 2 s / (5 substeps * 0.01 s) = 40 control steps


def test_hold_screen_rejects_state_that_terminates() -> None:
    task: Task = TASKS.construct("pendulum_sparse")  # pyright: ignore[reportAssignmentType] # registry is untyped

    class _TerminatesWhenRaised(type(task)):  # pyright: ignore[reportGeneralTypeIssues] # subclass of a registry-built task
        def terminated(self, sim: BatchSim, extra: Tensor) -> Tensor:
            return sim.qpos[:, 0] > 0.5

    rollouts, _ = _rollouts(_TerminatesWhenRaised())
    # Both rows hang at rest; row 1 starts above the threshold, so it terminates immediately.
    starts = StartStates(
        torch.tensor([[0.0], [1.0]]), torch.zeros(2, 1), torch.zeros(2, 0)
    )

    keep = rollouts.hold_screen(starts, seconds=0.5)

    assert keep.tolist() == [True, False]

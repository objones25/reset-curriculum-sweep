"""`reward_max`: the largest per-step reward a task can give (bounds the critic's vmax)."""

import math

import pytest

from resetsweep.tasks.base import TASKS, reward_max
from resetsweep.tasks.ramp import FAR_SLOPE_LEN_M

TASKS.load("resetsweep.tasks")


def test_reward_max_defaults_to_one_for_a_sparse_task() -> None:
    assert reward_max(TASKS.construct("pendulum_sparse")) == 1.0  # pyright: ignore[reportArgumentType] # registry returns object


def test_reward_max_of_ramp_dense_is_the_far_slope_top() -> None:
    task = TASKS.construct("ramp_dense")
    assert reward_max(task) == pytest.approx(  # pyright: ignore[reportArgumentType] # registry returns object
        FAR_SLOPE_LEN_M * math.sin(math.radians(25.0))
    )


def test_reward_max_of_maze_dense_is_zero() -> None:
    assert reward_max(TASKS.construct("maze_dense")) == 0.0  # pyright: ignore[reportArgumentType] # registry returns object

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from resetsweep.methods.base import P_TASK_DEFAULT, MethodContext
from resetsweep.scores.base import BaseScore, Score, median_fill
from resetsweep.sim.rollouts import Rollouts
from resetsweep.tasks.base import TASKS, Task
from resetsweep.types import Outcomes, RolloutStats, StepCounter

TASKS.load("resetsweep.tasks")

NUM_ENVS = 8


@pytest.fixture
def pendulum() -> Task:
    return TASKS.construct("pendulum_sparse")  # pyright: ignore[reportReturnType] # registry is untyped


@pytest.fixture
def make_ctx(pendulum: Task) -> Callable[..., MethodContext]:
    def factory(
        *,
        task: Task | None = None,
        p_task: float = P_TASK_DEFAULT,
        num_envs: int = NUM_ENVS,
        score: Score | None = None,
    ) -> MethodContext:
        task = task or pendulum
        gen = torch.Generator().manual_seed(0)
        counter = StepCounter(budget=10**9)
        rollouts = Rollouts(task, device="cpu", counter=counter, gen=gen, nworld_max=16)
        return MethodContext(
            task=task,
            num_envs=num_envs,
            device="cpu",
            gen=gen,
            score=score,
            rollouts=rollouts,
            counter=counter,
            p_task=p_task,
        )

    return factory


class FixedScore(BaseScore):
    """Score fake with settable `lp` / `trusted`; it logs every call it gets."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self.lp = torch.zeros(capacity)
        self.trusted = torch.ones(capacity, dtype=torch.bool)
        self.recorded: list[Outcomes] = []
        self.observed: list[RolloutStats] = []
        self.forgotten: list[list[int]] = []

    def record(self, outcomes: Outcomes) -> None:
        self.recorded.append(outcomes)

    def observe_rollout(self, stats: RolloutStats) -> None:
        self.observed.append(stats)

    def values(self, ic_ids: Tensor) -> tuple[Tensor, Tensor]:
        trusted = self.trusted[ic_ids]
        return median_fill(self.lp[ic_ids], trusted), trusted

    def forget(self, ic_ids: Tensor) -> None:
        self.forgotten.append(sorted(ic_ids.tolist()))


@pytest.fixture
def fixed_score() -> Callable[[int], FixedScore]:
    return FixedScore

from collections.abc import Callable

import pytest
import torch

from resetsweep.methods.base import P_TASK_DEFAULT, MethodContext
from resetsweep.sim.rollouts import Rollouts
from resetsweep.tasks.base import TASKS, Task
from resetsweep.types import StepCounter

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
            score=None,
            rollouts=rollouts,
            counter=counter,
            p_task=p_task,
        )

    return factory

"""M1 `uniform`: every reset draws from the task's own start distribution."""

import torch
from torch import Tensor

from resetsweep.methods.base import (
    METHODS,
    TASK_START_IC_ID,
    MethodBase,
    MethodContext,
    to_device,
)
from resetsweep.types import Outcomes, StartStates


@METHODS.register("uniform")
class Uniform(MethodBase):
    name = "uniform"

    def __init__(self, ctx: MethodContext) -> None:
        super().__init__(ctx)

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        del outcomes
        n = env_ids.numel()
        states = to_device(self.ctx.task.sample_start(n, self.ctx.gen), self.ctx.device)
        return states, torch.full(
            (n,), TASK_START_IC_ID, dtype=torch.long, device=self.ctx.device
        )

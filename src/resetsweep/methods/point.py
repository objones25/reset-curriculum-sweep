"""M2 `point`: every reset starts from one fixed initial condition (ic_id 0)."""

from typing import Literal

import torch
from torch import Tensor

from resetsweep.methods.base import METHODS, MethodBase, MethodContext, to_device
from resetsweep.types import Outcomes, StartStates

_SOURCES = ("canonical", "random")


@METHODS.register("point")
class Point(MethodBase):
    """`source="canonical"` uses `task.canonical_start()`; `"random"` draws one state from
    `task.sample_start` at construction. Either way the same state is used for the whole run."""

    name = "point"

    def __init__(
        self,
        ctx: MethodContext,
        *,
        source: Literal["canonical", "random"] = "canonical",
    ) -> None:
        super().__init__(ctx)
        if source not in _SOURCES:
            raise ValueError(f"source must be one of {_SOURCES}, got {source!r}")
        start = (
            ctx.task.canonical_start()
            if source == "canonical"
            else ctx.task.sample_start(1, ctx.gen)
        )
        if start.n != 1:
            raise ValueError(f"expected a single start state, got {start.n}")
        self._start = to_device(start, ctx.device)

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        del outcomes
        n = env_ids.numel()
        rows = torch.zeros(n, dtype=torch.long, device=self.ctx.device)
        return self._start.index(rows), torch.zeros(
            n, dtype=torch.long, device=self.ctx.device
        )

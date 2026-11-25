"""S3: learning progress from the trend of an IC's recent outcomes."""

from typing import Literal

import torch
from torch import Tensor

from resetsweep.scores.base import SCORES, TableScore

Form = Literal["signed", "abs", "fast_minus_slow"]
_FORMS: tuple[Form, ...] = ("signed", "abs", "fast_minus_slow")


@SCORES.register("slope")
class Slope(TableScore):
    """`signed`: least-squares slope of return over the last `window` outcomes; `abs`: its
    absolute value (1707.00183); `fast_minus_slow`: difference of two exponentially
    weighted means of success over the stored history, with per-outcome rates `fast` and
    `slow` (2106.14876). Weights are normalised over the outcomes that exist, so young ICs
    are not biased toward zero. An IC needs at least `min_count` outcomes (two for the
    regression forms) to be trusted.
    """

    def __init__(
        self,
        *,
        capacity: int,
        device: str,
        form: Form = "signed",
        window: int = 20,
        fast: float = 0.1,
        slow: float = 0.01,
        history: int = 100,
        min_count: int = 2,
    ) -> None:
        if form not in _FORMS:
            raise ValueError(f"form must be one of {_FORMS}, got {form!r}")
        # `window` only matters to the regression forms; fast_minus_slow reads the whole history.
        if window < 2 or (form != "fast_minus_slow" and window > history):
            raise ValueError(f"window must be in [2, history={history}], got {window}")
        if not 0.0 < slow < 1.0:
            raise ValueError(f"slow must be in (0, 1), got {slow}")
        if not slow < fast <= 1.0:
            raise ValueError(f"fast must be in (slow={slow}, 1], got {fast}")
        if min_count < 2:
            raise ValueError(f"min_count must be at least 2, got {min_count}")
        super().__init__(
            capacity=capacity, device=device, history=history, min_count=min_count
        )
        self._form: Form = form
        self._window = window
        self._fast = fast
        self._slow = slow

    def _raw_values(self, ic_ids: Tensor) -> Tensor:
        if self._form == "fast_minus_slow":
            return self._ewma(ic_ids, self._fast) - self._ewma(ic_ids, self._slow)
        slope = self._return_slope(ic_ids)
        return slope.abs() if self._form == "abs" else slope

    def _return_slope(self, ic_ids: Tensor) -> Tensor:
        ret, _, valid = self.table.recent(ic_ids, self._window)
        mask = valid.to(torch.float32)
        x = torch.arange(
            self._window, dtype=torch.float32, device=ret.device
        ).expand_as(ret)
        n = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        dx = x - (mask * x).sum(dim=1, keepdim=True) / n
        dy = ret - (mask * ret).sum(dim=1, keepdim=True) / n
        var = (mask * dx * dx).sum(dim=1)
        # var is 0 with fewer than two outcomes; those ICs are untrusted and median-filled.
        return (mask * dx * dy).sum(dim=1) / var.clamp(min=1e-12)

    def _ewma(self, ic_ids: Tensor, rate: float) -> Tensor:
        _, success, valid = self.table.recent(ic_ids, self.table.history)
        age = torch.arange(self.table.history - 1, -1, -1, device=success.device)
        weight = valid.to(torch.float32) * (1.0 - rate) ** age.to(torch.float32)
        total = weight.sum(dim=1).clamp(min=1e-12)
        return (weight * success.to(torch.float32)).sum(dim=1) / total

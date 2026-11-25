"""S1: success band. Peaks where an IC's recent success rate is near a target."""

from typing import Literal

import torch
from torch import Tensor

from resetsweep.scores.base import SCORES, TableScore

Form = Literal["sgs", "pq"]
_FORMS: tuple[Form, ...] = ("sgs", "pq")


@SCORES.register("success_band")
class SuccessBand(TableScore):
    """`pq`: lp = p(1 - p). `sgs`: the Beta-shaped logit of SGS (arXiv 2610.12465), with
    p the mean of the last `history` outcomes (H = 100 there):

        w = (p + eps)^(kappa t) (1 - p + eps)^(kappa (1 - t))   # paper file L206
        l = log w + eps                                           # L208
        prob(i) = softmax(l_i / T)                                # L215; T = 2 (L230)

    Source: papers/text/2610.12465_a-balanced-data-diet-...txt L203-231. `target` is t.
    Defaults are the paper's manipulation setting (t = 0.5, eps = 1e-4, kappa = 1); the
    locomotion setting is t = 0.66, eps = 1e-8, kappa = 5. `log w` is computed as the sum of
    the two logs, which is the same number without underflow for large kappa.

    For `sgs`, lp is l and `probs` is the softmax; for `pq`, `probs` is the base rank rule.
    """

    def __init__(
        self,
        *,
        capacity: int,
        device: str,
        history: int = 100,
        target: float = 0.5,
        form: Form = "sgs",
        min_count: int = 1,
        kappa: float = 1.0,
        eps: float = 1e-4,
        temperature: float = 2.0,
    ) -> None:
        if form not in _FORMS:
            raise ValueError(f"form must be one of {_FORMS}, got {form!r}")
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"target must be in [0, 1], got {target}")
        if kappa < 0.0:
            raise ValueError(f"kappa must be non-negative, got {kappa}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        super().__init__(
            capacity=capacity, device=device, history=history, min_count=min_count
        )
        self._form: Form = form
        self._target = target
        self._kappa = kappa
        self._eps = eps
        self._temperature = temperature

    def _raw_values(self, ic_ids: Tensor) -> Tensor:
        p = self.table.success_rate(ic_ids)
        if self._form == "pq":
            return p * (1.0 - p)
        t, kappa, eps = self._target, self._kappa, self._eps
        log_w = kappa * t * torch.log(p + eps) + kappa * (1.0 - t) * torch.log(
            1.0 - p + eps
        )
        return log_w + eps

    def probs(self, ic_ids: Tensor) -> Tensor:
        if self._form == "pq":
            return super().probs(ic_ids)
        lp, _ = self.values(ic_ids)
        return torch.softmax(lp / self._temperature, dim=0)

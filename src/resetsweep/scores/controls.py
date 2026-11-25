"""Control scores: an iid random score and the reversed ranking of another score."""

import torch
from torch import Tensor

from resetsweep.scores.base import SCORES, BaseScore
from resetsweep.types import Outcomes, RolloutStats


@SCORES.register("random")
class Random(BaseScore):
    """An iid uniform value per IC, fixed until that IC is forgotten. Always trusted."""

    def __init__(
        self, *, capacity: int, device: str, seed: int = 0, beta: float = 0.1
    ) -> None:
        self._set_beta(beta)
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._generator = torch.Generator(device=device).manual_seed(seed)
        self._device = device
        self._capacity = capacity
        self._lp = torch.rand(capacity, generator=self._generator, device=device)

    def record(self, outcomes: Outcomes) -> None:
        del outcomes

    def _check_ids(self, ic_ids: Tensor) -> None:
        if ic_ids.numel() and bool(((ic_ids < 0) | (ic_ids >= self._capacity)).any()):
            raise ValueError(f"ic_ids must be in [0, {self._capacity})")

    def values(self, ic_ids: Tensor) -> tuple[Tensor, Tensor]:
        self._check_ids(ic_ids)
        return self._lp[ic_ids], torch.ones_like(ic_ids, dtype=torch.bool)

    def forget(self, ic_ids: Tensor) -> None:
        self._check_ids(ic_ids)
        self._lp[ic_ids] = torch.rand(
            ic_ids.numel(), generator=self._generator, device=self._device
        )


@SCORES.register("inverted")
class Inverted(BaseScore):
    """`max(lp) - lp` of an inner score built through the registry, with the max taken over
    the queried ICs. Trust flags pass through. `probs` is the base rank rule on the
    inverted values."""

    def __init__(
        self,
        *,
        capacity: int,
        device: str,
        inner: str,
        inner_kw: dict[str, object] | None = None,
        beta: float = 0.1,
    ) -> None:
        self._set_beta(beta)
        built = SCORES.construct(
            inner, capacity=capacity, device=device, **(inner_kw or {})
        )
        if not isinstance(built, BaseScore):
            raise TypeError(f"inner score {inner!r} must derive from BaseScore")
        self._inner = built

    def record(self, outcomes: Outcomes) -> None:
        self._inner.record(outcomes)

    def observe_rollout(self, stats: RolloutStats) -> None:
        self._inner.observe_rollout(stats)

    def values(self, ic_ids: Tensor) -> tuple[Tensor, Tensor]:
        lp, trusted = self._inner.values(ic_ids)
        return lp.max() - lp, trusted

    def forget(self, ic_ids: Tensor) -> None:
        self._inner.forget(ic_ids)

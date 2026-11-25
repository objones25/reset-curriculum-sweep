"""Shared value types: start states, state boxes and the step-budget counter."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

ExtraKind = Literal["walk", "screen", "scout"]
_EXTRA_KINDS: tuple[ExtraKind, ...] = ("walk", "screen", "scout")


class BudgetExceeded(RuntimeError):
    """Total simulator steps went past the budget."""


class InvalidStartState(ValueError):
    """A reset method proposed a start state the env must not train on."""


@dataclass(frozen=True)
class StartStates:
    """Batch of start states; every field has `n` rows. `extra` may be `(n, 0)`."""

    qpos: Tensor
    qvel: Tensor
    extra: Tensor

    def __post_init__(self) -> None:
        if not (self.qpos.shape[0] == self.qvel.shape[0] == self.extra.shape[0]):
            raise ValueError(
                "qpos, qvel and extra must have the same number of rows, got "
                f"{self.qpos.shape[0]}, {self.qvel.shape[0]}, {self.extra.shape[0]}"
            )

    @property
    def n(self) -> int:
        return self.qpos.shape[0]

    def index(self, idx: Tensor) -> "StartStates":
        return StartStates(self.qpos[idx], self.qvel[idx], self.extra[idx])

    @staticmethod
    def cat(states: list["StartStates"]) -> "StartStates":
        if not states:
            raise ValueError("cat needs at least one StartStates")
        return StartStates(
            torch.cat([s.qpos for s in states]),
            torch.cat([s.qvel for s in states]),
            torch.cat([s.extra for s in states]),
        )


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounds over `concat(qpos, qvel)`."""

    low: Tensor
    high: Tensor


class StepCounter:
    """Counts simulator steps against a hard budget; train and extra steps both count."""

    def __init__(self, *, budget: int) -> None:
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        self._budget = budget
        self._train = 0
        self._extra: dict[str, int] = dict.fromkeys(_EXTRA_KINDS, 0)

    @property
    def train(self) -> int:
        return self._train

    @property
    def extra(self) -> dict[str, int]:
        return dict(self._extra)

    @property
    def total(self) -> int:
        return self._train + sum(self._extra.values())

    def add_train(self, n: int) -> None:
        self._check_count(n)
        self._train += n
        self._check_budget()

    def add_extra(self, n: int, kind: ExtraKind) -> None:
        self._check_count(n)
        if kind not in self._extra:
            raise ValueError(
                f"unknown extra kind {kind!r}; expected one of {_EXTRA_KINDS}"
            )
        self._extra[kind] += n
        self._check_budget()

    @staticmethod
    def _check_count(n: int) -> None:
        if n < 0:
            raise ValueError(f"step count must be non-negative, got {n}")

    def _check_budget(self) -> None:
        if self.total > self._budget:
            raise BudgetExceeded(f"step budget exceeded: {self.total} > {self._budget}")

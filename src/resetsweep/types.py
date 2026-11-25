"""Shared value types: start states, state boxes and the step-budget counter."""

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Literal

import torch
from torch import Tensor

ExtraKind = Literal["walk", "screen", "scout"]
_EXTRA_KINDS: tuple[ExtraKind, ...] = ("walk", "screen", "scout")


class BudgetExceeded(RuntimeError):
    """Total simulator steps went past the budget."""


class InvalidStartState(ValueError):
    """A reset method proposed a start state the env must not train on."""

    def __init__(self, method_name: str, row: int, reason: str) -> None:
        super().__init__(
            f"method {method_name!r} returned an invalid start state at row {row}: {reason}"
        )
        self.method_name = method_name
        self.row = row


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


@dataclass(frozen=True)
class Outcomes:
    """Finished episodes, one row each. `ep_len` counts the env clock, so it includes the
    staggered offset for first episodes; `full` is False for those. `start_step` is the
    number of vector steps completed before the episode began (negative for first episodes)."""

    env_ids: Tensor
    ic_ids: Tensor
    ret: Tensor
    success: Tensor
    ep_len: Tensor
    reached_reward: Tensor
    full: Tensor
    start_step: Tensor

    @staticmethod
    def cat(outcomes: list["Outcomes"]) -> "Outcomes":
        if not outcomes:
            raise ValueError("cat needs at least one Outcomes")
        return Outcomes(
            *(
                torch.cat([getattr(o, f.name) for o in outcomes])
                for f in fields(Outcomes)
            )
        )


@dataclass(frozen=True)
class StepOut:
    """One vector step. `obs` is after any reset; `final_obs` is before it (bootstrap value
    for truncated envs). `ic_ids` are the ids of each env's current episode, after reset."""

    obs: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    final_obs: Tensor
    ic_ids: Tensor
    outcomes: Outcomes | None


@dataclass(frozen=True)
class RolloutStats:
    """PPO rollout data handed to scores. `ic_ids`, `adv`, `returns` and `values` are `(T, N)`
    (`T` vector steps, `N` envs); `start_obs` holds the observation at each IC's start, and
    `critic` maps a batch of observations to values in one call. `adv` must be raw GAE, not
    normalised per batch: the critic-loss scores read its sign and scale."""

    ic_ids: Tensor
    adv: Tensor
    returns: Tensor
    values: Tensor
    start_obs: Tensor
    critic: Callable[[Tensor], Tensor]

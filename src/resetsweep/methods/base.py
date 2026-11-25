"""The reset-method protocol, its shared context and the task-start mixing helper.

A method decides where each episode starts. `VecEnv` calls `reset` for every finished
episode (and once for all envs at the start) and `offer_visited` every `visit_every` steps;
the training loop calls `update` once per PPO iteration and logs `diagnostics`.

`ic_id` convention: `-1` marks a task-start row (from `task.sample_start`); scores ignore
those rows. Ids `>= 0` name an initial condition a score may track.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch
from torch import Tensor

from resetsweep.registry import make_registry
from resetsweep.scores.base import Score
from resetsweep.sim.rollouts import Rollouts
from resetsweep.tasks.base import Task
from resetsweep.types import Outcomes, RolloutStats, StartStates, StepCounter

METHODS = make_registry("method")

P_TASK_DEFAULT = 0.5
P_TASK_SWEEP = (0.0, 0.2, 0.5, 0.8, 1.0)
TASK_START_IC_ID = -1

DrawFn = Callable[[int], tuple[StartStates, Tensor]]


@dataclass(frozen=True)
class MethodContext:
    """Everything a method is built from. `p_task` is the share of resets sent to the
    task's own start distribution; it must be in [0, 1]."""

    task: Task
    num_envs: int
    device: str
    gen: torch.Generator
    score: Score | None
    rollouts: Rollouts
    counter: StepCounter
    p_task: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_task <= 1.0:
            raise ValueError(f"p_task must be in [0, 1], got {self.p_task}")
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")


class Method(Protocol):
    """Contract between a reset method and `VecEnv` / the training loop.

    `diagnostics()` must return at least these keys (all floats):

    - `sampling_entropy`: Shannon entropy in nats of the discrete distribution the next
      reset draws from, over the method's pool entries plus one atom for "the task start
      distribution". 0.0 for a method with no pool.
    - `buffer_size`: number of states or ICs currently held in the method's pool (0 if none).
    - `never_succeeded_share`: share of tracked pool entries with no recorded success; NaN
      when the method tracks no per-entry outcomes.
    """

    name: str
    accepts_scores: ClassVar[frozenset[str]]

    def __init__(self, ctx: MethodContext, **settings: object) -> None: ...

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]: ...

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None: ...

    def update(self, it: int, stats: RolloutStats) -> dict[str, float]: ...

    def diagnostics(self) -> dict[str, float]: ...

    def ic_states(self) -> StartStates | None:
        """The IC table, or None for a method without stable IC ids.

        When not None it has exactly `capacity` rows (the score's capacity) and row `i` is the
        start state of IC `i`, so `reset` only returns ic ids `-1` or in `[0, capacity)`. Rows
        of ICs not yet filled hold zeros: they are placeholders, never drawn and never valid
        states. train.py builds `RolloutStats.start_obs` and `start_features` from this table
        (via `task.observe` / `task.state_features`) so both stay aligned by IC id. The table
        is a copy; it changes whenever an IC is replaced, so rebuild it once per update.
        """
        ...


class MethodBase:
    """Defaults for methods with no visited-state intake, no learning and no pool."""

    name: str
    accepts_scores: ClassVar[frozenset[str]] = frozenset({"none"})

    def __init__(self, ctx: MethodContext) -> None:
        self.ctx = ctx

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None:
        del env_ids, qpos, qvel, extra, ep_step

    def update(self, it: int, stats: RolloutStats) -> dict[str, float]:
        del it, stats
        return {}

    def ic_states(self) -> StartStates | None:
        return None

    def diagnostics(self) -> dict[str, float]:
        return {
            "sampling_entropy": 0.0,
            "buffer_size": 0.0,
            "never_succeeded_share": float("nan"),
        }


def mix_task_start(
    ctx: MethodContext, env_ids: Tensor, draw_fn: DrawFn
) -> tuple[StartStates, Tensor]:
    """Start states for `env_ids`: each row independently comes from `task.sample_start`
    (with `ic_id = -1`) with probability `ctx.p_task`, otherwise from `draw_fn(k)`, which
    returns `k` states and their `k` ic_ids. Row order follows `env_ids`.

    Host-syncs once to size the two groups, which `VecEnv` already does on reset steps.
    """
    n = env_ids.numel()
    to_task = torch.rand(n, generator=ctx.gen, device=ctx.device) < ctx.p_task
    task_rows = to_task.nonzero(as_tuple=True)[0]
    other_rows = (~to_task).nonzero(as_tuple=True)[0]
    parts: list[StartStates] = []
    ids: list[Tensor] = []
    if task_rows.numel() > 0:
        k = task_rows.numel()
        parts.append(to_device(ctx.task.sample_start(k, ctx.gen), ctx.device))
        ids.append(
            torch.full((k,), TASK_START_IC_ID, dtype=torch.long, device=ctx.device)
        )
    if other_rows.numel() > 0:
        states, ic_ids = draw_fn(other_rows.numel())
        _check_draw(states, ic_ids, other_rows.numel())
        parts.append(states)
        ids.append(ic_ids)
    order = torch.cat([task_rows, other_rows])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(n, device=ctx.device)
    return StartStates.cat(parts).index(inverse), torch.cat(ids)[inverse]


def _check_draw(states: StartStates, ic_ids: Tensor, k: int) -> None:
    if states.n != k or ic_ids.shape != (k,):
        raise ValueError(
            f"draw_fn returned {states.n} states and ic_ids {tuple(ic_ids.shape)}, expected {k}"
        )


def to_device(states: StartStates, device: str) -> StartStates:
    return StartStates(
        states.qpos.to(device), states.qvel.to(device), states.extra.to(device)
    )

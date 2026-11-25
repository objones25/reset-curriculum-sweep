"""M8 `visited`: restart from states the training envs themselves passed through.

Every `visit_every` env steps `VecEnv` offers all envs' current states; the valid ones go
into a fixed-capacity ring buffer (all time offsets, so no bias towards episode starts), and
resets draw uniformly from it. Until the buffer holds `min_fill` states, resets use the task start.
"""

import math

import torch
from torch import Tensor

from resetsweep.methods.base import (
    METHODS,
    TASK_START_IC_ID,
    MethodBase,
    MethodContext,
    mix_task_start,
    to_device,
)
from resetsweep.types import Outcomes, StartStates

CAPACITY_DEFAULT = 65536


@METHODS.register("visited")
class Visited(MethodBase):
    """Ring buffer of visited states, all on device.

    `offer_visited` adds no host syncs of its own, but it calls `task.is_valid`, which may
    sync (contact tasks such as `arm` run a CPU contact check). Rows `is_valid` rejects are
    routed to a spare slot at index `capacity` instead of being filtered, and the write
    pointer advances by the device-side count of valid rows. `is_valid` includes the
    `start_box` check, so high-velocity visited states are excluded on purpose: `VecEnv`
    would reject them as starts.
    `reset` syncs once to read the fill level. Buffer rows hold `qpos`, `qvel` and `extra`
    (so a restored state carries e.g. the trap), copied at offer time. Rows drawn from the
    buffer get `ic_id = -1`: they are not stable ICs (slots get overwritten), so scores ignore them.
    """

    name = "visited"

    def __init__(
        self,
        ctx: MethodContext,
        *,
        capacity: int = CAPACITY_DEFAULT,
        min_fill: int | None = None,
    ) -> None:
        super().__init__(ctx)
        min_fill = ctx.num_envs if min_fill is None else min_fill
        if capacity < ctx.num_envs:
            raise ValueError(
                f"capacity ({capacity}) must be at least num_envs ({ctx.num_envs}): one offer writes num_envs rows"
            )
        if not 1 <= min_fill <= capacity:
            raise ValueError(
                f"min_fill must be in [1, capacity={capacity}], got {min_fill}"
            )
        shape = ctx.task.canonical_start()
        self._capacity = capacity
        self._min_fill = min_fill
        d = ctx.device
        rows = capacity + 1  # last row absorbs rejected offers
        self._qpos = torch.zeros(rows, shape.qpos.shape[1], device=d)
        self._qvel = torch.zeros(rows, shape.qvel.shape[1], device=d)
        self._extra = torch.zeros(rows, ctx.task.extra_dim, device=d)
        self._ptr = torch.zeros((), dtype=torch.long, device=d)
        self._size = torch.zeros((), dtype=torch.long, device=d)

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None:
        del env_ids, ep_step
        states = StartStates(qpos, qvel, extra)
        valid = (
            self.ctx.task.is_valid(states)
            & torch.isfinite(qpos).all(dim=1)
            & torch.isfinite(qvel).all(dim=1)
            & torch.isfinite(extra).all(dim=1)
        )
        rank = valid.long().cumsum(0) - 1
        slot = torch.where(
            valid,
            (self._ptr + rank) % self._capacity,
            torch.full_like(rank, self._capacity),
        )
        self._qpos[slot] = qpos
        self._qvel[slot] = qvel
        self._extra[slot] = extra
        added = valid.sum()
        self._ptr = (self._ptr + added) % self._capacity
        self._size = (self._size + added).clamp(max=self._capacity)

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        del outcomes
        size = int(self._size)
        if size < self._min_fill:
            n = env_ids.numel()
            states = to_device(
                self.ctx.task.sample_start(n, self.ctx.gen), self.ctx.device
            )
            ids = torch.full(
                (n,), TASK_START_IC_ID, dtype=torch.long, device=self.ctx.device
            )
            return states, ids
        return mix_task_start(self.ctx, env_ids, lambda k: self._draw(k, size))

    def _draw(self, k: int, size: int) -> tuple[StartStates, Tensor]:
        idx = torch.randint(
            0, size, (k,), generator=self.ctx.gen, device=self.ctx.device
        )
        states = StartStates(self._qpos[idx], self._qvel[idx], self._extra[idx])
        return states, torch.full(
            (k,), TASK_START_IC_ID, dtype=torch.long, device=self.ctx.device
        )

    def buffer(self) -> StartStates:
        """Copy of the states currently stored, in slot order."""
        size = int(self._size)
        return StartStates(
            self._qpos[:size].clone(),
            self._qvel[:size].clone(),
            self._extra[:size].clone(),
        )

    def diagnostics(self) -> dict[str, float]:
        """`buffer_size`: stored states. `sampling_entropy`: entropy in nats over one atom
        (the task start, probability p_task) plus `buffer_size` equally likely states (each
        (1 - p_task) / buffer_size); 0.0 while below `min_fill`. `never_succeeded_share` is NaN:
        buffer slots are not tracked ICs."""
        size = int(self._size)
        if size < self._min_fill:
            entropy = 0.0
        else:
            p = self.ctx.p_task
            entropy = (1 - p) * math.log(size / (1 - p)) if p < 1 else 0.0
            entropy -= p * math.log(p) if p > 0 else 0.0
        return {
            "sampling_entropy": entropy,
            "buffer_size": float(size),
            "never_succeeded_share": float("nan"),
        }

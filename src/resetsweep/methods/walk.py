"""M6 `walk`: a population of K ICs whose low-scored members are replaced by states reached
from a kept parent by a random-action walk (spec section 5.3; shortlist section 3, method 6).

The reverse-curriculum paper (1707.05300) and the Voronoi-biased curriculum (2610.03395) grow
their IC sets this way, so children are reached states, not points moved in a box. Walk
output is unfiltered (`Rollouts.random_walk`), so every child still goes through the
population filter (finite, inside `start_box`, `task.is_valid`); a rejected child is walked
again from the same parent, up to `tries` times, then falls back to the task start.

Every `update_every` iterations: the top `ceil(keep_top * K)` ICs by score are kept and
every other IC is replaced (ours: `keep_top` is the share kept, which is also the parent
pool). Each replaced IC gets a parent drawn from the kept ones with weight
`rank^-parent_power` (rank 1 = best; 0 makes the draw uniform), and its child is the state
after `walk_steps` control steps of uniform random actions held for `action_hold` steps.
Every walk step is charged to the step counter as kind "walk". There is no spread rule
here (the spec gives it to `ball` and `genetic` only): before any IC is trusted all scores
equal the median, ties break at random, and the walk keeps expanding the population from
the task start.
"""

from torch import Tensor

from resetsweep.methods._population import (
    R_DEFAULT,
    TRIES_DEFAULT,
    UPDATE_EVERY_DEFAULT,
    PopulationMixin,
    ProposeFn,
)
from resetsweep.methods.base import METHODS, MethodContext
from resetsweep.types import StartStates


@METHODS.register("walk")
class Walk(PopulationMixin):
    """Random-walk children of kept ICs; see the module docstring and `PopulationMixin`."""

    name = "walk"

    def __init__(
        self,
        ctx: MethodContext,
        *,
        R: int = R_DEFAULT,
        walk_steps: int = 20,
        action_hold: int = 5,
        parent_power: float = 1.0,
        keep_top: float = 0.2,
        update_every: int = UPDATE_EVERY_DEFAULT,
        tries: int = TRIES_DEFAULT,
    ) -> None:
        if walk_steps < 1 or action_hold < 1:
            raise ValueError(
                f"walk_steps and action_hold must be at least 1, got {walk_steps}, {action_hold}"
            )
        if parent_power < 0:
            raise ValueError(f"parent_power must be >= 0, got {parent_power}")
        if not 0.0 < keep_top <= 1.0:
            raise ValueError(f"keep_top must be in (0, 1], got {keep_top}")
        super().__init__(
            ctx, R=R, update_every=update_every, tries=tries, screen_seconds=0.0
        )
        self._walk_steps = walk_steps
        self._action_hold = action_hold
        self._parent_power = parent_power
        self._keep_top = keep_top

    def _plan(self) -> tuple[Tensor, ProposeFn] | None:
        order, _, _ = self._ranked()
        plan = self._keep_plan(
            order, keep_top=self._keep_top, parent_power=self._parent_power
        )
        if plan is None:
            return None
        targets, parents, _ = plan

        def propose(rows: Tensor) -> StartStates:
            return self.ctx.rollouts.random_walk(
                self._rows(parents[rows]), self._walk_steps, self._action_hold
            )

        return targets, propose

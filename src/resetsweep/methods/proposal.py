"""M7 `ball` and `genetic`: population methods whose children are points moved in normalised
`start_box` coordinates (spec section 5.3; shortlist section 3, method 7).

Children here are not visited states (shortlist finding E), so each one is filtered by the
population filter (finite, inside `start_box`, `task.is_valid`), redrawn up to `tries`
times, then replaced by a task-start draw; an invalid row is never admitted (Review Focus
5). With `screen_seconds > 0` each child must also pass a zero-action hold rollout of that
length, charged as kind "screen", with `speed_max` = the largest |qvel| bound of
`start_box` (2609.35715 rejects on termination or excess speed; 2609.13761 holds 2 s). A
child keeps its parent's `extra`.

Both cull only when at least two ICs are trusted and the trusted scores' relative spread
`(max - min) / max(|median|, 1e-8)` reaches `spread_threshold` (an analogue of 2503.19037
L298-312, which selects only when fitness spread exceeds a threshold relative to the
median), so neither culls while every score is near zero.

Selection reads measured ICs only: untrusted ICs (a score's median placeholder, not a
measurement) are never culled, never parents and never ranked, so a child survives until
its score trusts it.

`ball`: of the `n` trusted ICs, keeps the top `ceil(keep_top * n)` and replaces the other
trusted ones. Each child's parent
is a kept IC drawn with weight `rank^-parent_power`; the child is uniform in the ball of
radius `r = r_min + (r_max - r_min) * (1 - q)^radius_power` around the parent, where `q` is
the parent's quality in the kept pool (1 best, 0 worst): better parents search closer
(ours, mirroring `genetic`'s "high-score parents mutate less").

`genetic`: with `n` trusted ICs, the bottom `m = min(ceil(p * n), n // 2)` trusted ICs
are replaced by children of the top `m` trusted ones, each parent drawn uniformly from the
top `m`. A child is the parent plus Gaussian
noise of scale `sigma = sigma_min + (sigma_max - sigma_min) * (1 - q)`, `q` the parent's
quality within the top `m`. Parents in the lower half of that pool (`q < 0.5`, ours) draw
`n_candidates` mutations and keep the most novel valid one: the largest mean distance to
its `novelty_k` nearest ICs of the current population in `task.state_features` space.
Other parents keep their first valid mutation.
"""

import math

import torch
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


def knn_novelty(candidates: Tensor, population: Tensor, *, k: int) -> Tensor:
    """`(n,)` mean Euclidean distance from each of `candidates` `(n, f)` to its `k` nearest
    rows of `population` `(p, f)`."""
    if not 1 <= k <= population.shape[0]:
        raise ValueError(f"k must be in [1, {population.shape[0]}], got {k}")
    distance = torch.cdist(candidates, population)
    return distance.topk(k, dim=1, largest=False).values.mean(dim=1)


def _check_spread(spread_threshold: float) -> None:
    if spread_threshold < 0:
        raise ValueError(f"spread_threshold must be >= 0, got {spread_threshold}")


@METHODS.register("ball")
class Ball(PopulationMixin):
    """Children uniform in a score-sized ball around kept ICs; see the module docstring."""

    name = "ball"

    def __init__(
        self,
        ctx: MethodContext,
        *,
        R: int = R_DEFAULT,
        parent_power: float = 1.0,
        radius_power: float = 1.0,
        r_min: float = 0.02,
        r_max: float = 0.3,
        keep_top: float = 0.2,
        tries: int = TRIES_DEFAULT,
        screen_seconds: float = 0.0,
        spread_threshold: float = 0.1,
        update_every: int = UPDATE_EVERY_DEFAULT,
    ) -> None:
        if parent_power < 0 or radius_power < 0:
            raise ValueError(
                f"parent_power and radius_power must be >= 0, got {parent_power}, {radius_power}"
            )
        if not 0.0 < r_min <= r_max:
            raise ValueError(f"need 0 < r_min <= r_max, got {r_min}, {r_max}")
        if not 0.0 < keep_top <= 1.0:
            raise ValueError(f"keep_top must be in (0, 1], got {keep_top}")
        _check_spread(spread_threshold)
        super().__init__(
            ctx,
            R=R,
            update_every=update_every,
            tries=tries,
            screen_seconds=screen_seconds,
        )
        self._parent_power = parent_power
        self._radius_power = radius_power
        self._r_min = r_min
        self._r_max = r_max
        self._keep_top = keep_top
        self._spread_threshold = spread_threshold

    def _plan(self) -> tuple[Tensor, ProposeFn] | None:
        order, lp, trusted = self._ranked()
        if not self._spread_ok(lp, trusted, self._spread_threshold):
            return None
        plan = self._keep_plan(
            order, keep_top=self._keep_top, parent_power=self._parent_power
        )
        if plan is None:
            return None
        targets, parents, quality = plan
        radius = self._r_min + (self._r_max - self._r_min) * (1.0 - quality).pow(
            self._radius_power
        )
        gen = self.ctx.gen

        def propose(rows: Tensor) -> StartStates:
            centre = self._unit(parents[rows])
            n, dim = centre.shape
            direction = torch.randn(n, dim, generator=gen, device=centre.device)
            direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-12)
            # Uniform in the ball: radius scales as u^(1/dim).
            u = torch.rand(n, generator=gen, device=centre.device)
            length = radius[rows] * u.pow(1.0 / dim)
            child = centre + direction * length.unsqueeze(1)
            return self._from_unit(child, self._extra[parents[rows]])

        return targets, propose


@METHODS.register("genetic")
class Genetic(PopulationMixin):
    """Bottom `p` replaced by mutated children of the top `p`; see the module docstring."""

    name = "genetic"

    def __init__(
        self,
        ctx: MethodContext,
        *,
        R: int = R_DEFAULT,
        p: float = 0.2,
        sigma_min: float = 0.02,
        sigma_max: float = 0.3,
        n_candidates: int = 4,
        novelty_k: int = 5,
        tries: int = TRIES_DEFAULT,
        screen_seconds: float = 0.0,
        spread_threshold: float = 0.1,
        update_every: int = UPDATE_EVERY_DEFAULT,
    ) -> None:
        if not 0.0 < p <= 0.5:
            raise ValueError(f"p must be in (0, 0.5], got {p}")
        if not 0.0 < sigma_min <= sigma_max:
            raise ValueError(
                f"need 0 < sigma_min <= sigma_max, got {sigma_min}, {sigma_max}"
            )
        if n_candidates < 1 or novelty_k < 1:
            raise ValueError(
                f"n_candidates and novelty_k must be at least 1, got {n_candidates}, {novelty_k}"
            )
        _check_spread(spread_threshold)
        super().__init__(
            ctx,
            R=R,
            update_every=update_every,
            tries=tries,
            screen_seconds=screen_seconds,
        )
        self._p = p
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._n_candidates = n_candidates
        self._novelty_k = min(novelty_k, self._k)
        self._spread_threshold = spread_threshold

    def _plan(self) -> tuple[Tensor, ProposeFn] | None:
        order, lp, trusted = self._ranked()
        if not self._spread_ok(lp, trusted, self._spread_threshold):
            return None
        n = order.numel()  # trusted slots only (see `_ranked`)
        m = min(math.ceil(self._p * n), n // 2)
        if m == 0:
            return None
        targets = order[-m:].sort().values
        top = order[:m]  # best first
        pool_index = torch.randint(
            m, (m,), generator=self.ctx.gen, device=self.ctx.device
        )
        parents = top[pool_index]
        quality = self._pool_quality(pool_index, m)
        sigma = self._sigma_min + (self._sigma_max - self._sigma_min) * (1.0 - quality)
        novel = quality < 0.5
        population = self._features(self.ic_states())

        def propose(rows: Tensor) -> StartStates:
            return self._mutate(parents[rows], sigma[rows], novel[rows], population)

        return targets, propose

    def _features(self, states: StartStates) -> Tensor:
        return self.ctx.task.state_features(states.qpos, states.qvel, states.extra)

    def _mutate(
        self, parents: Tensor, sigma: Tensor, novel: Tensor, population: Tensor
    ) -> StartStates:
        """One child per parent: the most novel valid of `n_candidates` mutations where
        `novel`, else the first valid one (candidate 0 when none is valid, so the
        population filter rejects it and the row is redrawn)."""
        n, c = parents.numel(), self._n_candidates
        centre = self._unit(parents)
        noise = torch.randn(
            n, c, centre.shape[1], generator=self.ctx.gen, device=centre.device
        )
        unit = (centre.unsqueeze(1) + sigma.view(n, 1, 1) * noise).reshape(n * c, -1)
        cands = self._from_unit(unit, self._extra[parents].repeat_interleave(c, dim=0))
        valid = self._valid(cands).view(n, c)
        novelty = knn_novelty(self._features(cands), population, k=self._novelty_k)
        most_novel = torch.where(valid, novelty.view(n, c), -torch.inf).argmax(dim=1)
        position = torch.arange(c, device=centre.device, dtype=torch.float32)
        first_valid = torch.where(valid, -position, -torch.inf).argmax(dim=1)
        pick = torch.where(novel, most_novel, first_valid)
        return cands.index(torch.arange(n, device=centre.device) * c + pick)

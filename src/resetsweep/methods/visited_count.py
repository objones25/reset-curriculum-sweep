"""M3 `visited_count`: draw from the IC table (promoted from the visit ring, see `SlotBuffer`)
with probability proportional to the `visit_count` score's lp, 1 / sqrt(count + 1)
(Go-Explore, 2004.12919 L1632-1633), where the count comes from one of three rules
(shortlist section 3, method 3):

- `neighbour`: visit-ring states within `radius` of the IC, on state features standardised
  by running mean and std (2609.08642 L257-260 counts this way for goals). The IC's own copy
  in the ring, when still there, counts too, so a state seen once has count 1. Counting
  against the ring keeps the count a density of what the policy visits.
- `lineage_depth`: the number of restarts along the chain from the task start to the episode
  the state was visited in (a state visited in an episode that began at the task start has
  depth 0; one visited after restarting from a depth-d IC has depth d + 1). A ring entry
  stores its depth and the IC promoted from it carries it. Deeper states are preferred: depth
  is a metric-free stand-in for "further from the start / rarer" (reset_curriculum_shortlist.md
  L100, section 3 method 3). Each recompute feeds the score the pseudo-count
  `max_depth - depth` (max over the filled ICs), so lp = 1 / sqrt(max_depth - depth + 1) rises
  monotonically with depth and the deepest ICs get lp = 1. The known weakness is the
  shortlist's: chains that wander in place also grow deep.
- `embedding`: ring neighbours within `radius` in the embedding of an internal
  `step_distance` model (2407.08464), trained on consecutive visited states of one episode.

Cost: counts are recomputed right after each promotion (every `update_every` updates),
never in `reset`. A neighbour count is chunked `torch.cdist` of the IC table against the
ring on device, O(ic_capacity * ring) per recompute. `reset` draws by inverse CDF from a
table rebuilt at the same time, O(k log ic_capacity) for k resetting envs; the IC table
itself only changes at promotion, so the draw table never points at a replaced state.
"""

import math
import sys
from typing import Literal

import torch
from torch import Tensor

from resetsweep.methods._slots import (
    IC_CAPACITY_DEFAULT,
    PROMOTE_FRAC_DEFAULT,
    SlotBuffer,
)
from resetsweep.methods.base import METHODS, MethodContext
from resetsweep.methods.visited import CAPACITY_DEFAULT
from resetsweep.scores.base import SCORES
from resetsweep.scores.side_models import StepDistance
from resetsweep.scores.visit_count import VisitCount
from resetsweep.types import Outcomes, StartStates

CountRule = Literal["neighbour", "lineage_depth", "embedding"]
_COUNT_RULES: tuple[CountRule, ...] = ("neighbour", "lineage_depth", "embedding")

_CDIST_ELEMENTS_MAX = 2**25  # distance-matrix entries per chunk: 128 MiB in float32
_STD_MIN = 1e-6  # constant feature columns standardise to 0 instead of dividing by 0
_PAIR_CAPACITY = 16384  # consecutive-state pairs kept for the step-distance model
_PAIRS_PER_UPDATE = 4096
_DIAG_ANCHORS = 256  # ICs whose ring neighbours the across-wall diagnostic inspects
_DIAG_PAIRS_MAX = 65536
_SAME_STATE_DIST = 1e-5  # below this a ring entry is the IC's own copy (cdist rounding)


@METHODS.register("visited_count")
class VisitedCount(SlotBuffer):
    """Count-weighted draw over the IC table; see the module docstring.

    Takes only the `visit_count` score and feeds it with `set_counts`. When any outcome first
    has `reached_reward`, it calls `score.notify_first_reward()` and logs the step
    (`first_reward_step`, the total step count at that reset). For tasks whose module defines
    `path_distance` (the maze family) each recompute also logs `across_wall_neighbour_share`
    (neighbour and embedding rules only): over pairs of a sampled IC and a ring neighbour
    (pairs at feature distance below 1e-5, the IC's own ring copy, excluded), the share whose path
    distance exceeds 2 x radius, with the radius turned into metres by multiplying with the
    mean standard deviation of the ring's x and y. Embedding mode logs the step-distance
    training time as `side/wall_ms_per_update`.

    `offer_visited` adds no host sync of its own (beyond `task.is_valid`, as in `Visited`).
    `slot_probs()` is the distribution the score implies now; `reset` draws from the table
    built at the last recompute (after the first reward the learnability factor can move lp
    between recomputes).
    """

    name = "visited_count"
    accepts_scores = frozenset({"visit_count"})

    def __init__(
        self,
        ctx: MethodContext,
        *,
        capacity: int = CAPACITY_DEFAULT,
        min_fill: int | None = None,
        count: CountRule = "neighbour",
        radius: float = 0.5,
        update_every: int = 1,
        ic_capacity: int = IC_CAPACITY_DEFAULT,
        promote_frac: float = PROMOTE_FRAC_DEFAULT,
    ) -> None:
        score = ctx.score
        if not isinstance(score, VisitCount):
            raise TypeError(
                f"visited_count accepts only the visit_count score, got {type(score).__name__}"
            )
        if count not in _COUNT_RULES:
            raise ValueError(f"count must be one of {_COUNT_RULES}, got {count!r}")
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}")
        super().__init__(
            ctx,
            capacity=capacity,
            min_fill=min_fill,
            ic_capacity=ic_capacity,
            promote_frac=promote_frac,
            update_every=update_every,
        )
        self._visit_score: VisitCount = score
        self._rule: CountRule = count
        self._radius = radius
        d = ctx.device
        canon = ctx.task.canonical_start()
        width = ctx.task.state_features(canon.qpos, canon.qvel, canon.extra).shape[1]
        self._counts = torch.zeros(ic_capacity, dtype=torch.long, device=d)
        self._depth = torch.zeros(capacity + 1, dtype=torch.long, device=d)  # ring
        self._ic_depth = torch.zeros(ic_capacity, dtype=torch.long, device=d)
        self._env_depth = torch.zeros(ctx.num_envs, dtype=torch.long, device=d)
        # Running feature statistics (Chan et al. parallel update), all on device.
        self._feat_n = torch.zeros((), device=d)
        self._feat_mean = torch.zeros(width, device=d)
        self._feat_m2 = torch.zeros(width, device=d)
        self._first_reward_step = float("nan")
        self._across_share = float("nan")
        task_module = sys.modules[type(ctx.task).__module__]
        self._path_distance = getattr(task_module, "path_distance", None)
        self._cdf: Tensor | None = None  # draw table, rebuilt by each recompute
        self._step_model: StepDistance | None = None
        if count == "embedding":
            model = SCORES.construct("step_distance", capacity=ic_capacity, device=d)
            if not isinstance(model, StepDistance):
                raise TypeError(f"step_distance built a {type(model).__name__}")
            self._step_model = model
            self._prev_feat = torch.zeros(ctx.num_envs, width, device=d)
            self._prev_ok = torch.zeros(ctx.num_envs, dtype=torch.bool, device=d)
            self._pair_a = torch.zeros(_PAIR_CAPACITY + 1, width, device=d)
            self._pair_b = torch.zeros(_PAIR_CAPACITY + 1, width, device=d)
            self._pair_ptr = torch.zeros((), dtype=torch.long, device=d)
            self._pair_size = torch.zeros((), dtype=torch.long, device=d)

    def depths(self) -> Tensor:
        """`(buffer_size,)` lineage depth per filled IC (tracked under every count rule)."""
        return self._ic_depth[: self._ic_size].clone()

    def counts(self) -> Tensor:
        """`(buffer_size,)` count fed to the score at the last recompute: the ring neighbour
        count, or the lineage pseudo-count `max_depth - depth`."""
        return self._counts[: self._ic_size].clone()

    def diagnostics(self) -> dict[str, float]:
        diag = super().diagnostics()
        diag["first_reward_step"] = self._first_reward_step
        if self._path_distance is not None:
            diag["across_wall_neighbour_share"] = self._across_share
        if self._step_model is not None:
            wall_ms = self._step_model.extra_metrics()["wall_ms_per_update"]
            diag["side/wall_ms_per_update"] = wall_ms
        return diag

    def _after_promotion(self) -> None:
        size = self._ic_size
        ids = torch.arange(size, device=self.ctx.device)
        if self._rule == "lineage_depth":
            depth = self._ic_depth[:size]
            counts = depth.max() - depth  # deepest IC: count 0, the largest lp
        else:
            ring = int(self._size)
            z_ic = self._standardise(self._features(self._ic_rows(ids)))
            z_ring = self._standardise(
                self._features(
                    StartStates(
                        self._qpos[:ring], self._qvel[:ring], self._extra[:ring]
                    )
                )
            )
            if self._step_model is not None:
                self._train_step_model()
                z_ic, z_ring = (
                    self._step_model.embed(z_ic),
                    self._step_model.embed(z_ring),
                )
            counts = self._ring_neighbour_counts(z_ic, z_ring)
            if self._path_distance is not None:
                self._across_share = self._across_wall_share(z_ic, z_ring)
        self._counts[:size] = counts
        self._visit_score.set_counts(ids, counts)
        self._cdf = torch.cumsum(self._probs(size), dim=0)

    def _draw_ics(self, k: int) -> tuple[StartStates, Tensor]:
        """Inverse-CDF draw from the table built by the last recompute (or, before one, at
        the first draw): O(k log n) per reset."""
        if self._cdf is None:
            self._cdf = torch.cumsum(self._probs(self._ic_size), dim=0)
        u = torch.rand(k, generator=self.ctx.gen, device=self.ctx.device)
        idx = torch.searchsorted(self._cdf, u * self._cdf[-1], right=True)
        idx = idx.clamp(max=self._cdf.numel() - 1)
        return self._ic_rows(idx), idx

    def _probs(self, size: int) -> Tensor:
        lp, _ = self._visit_score.values(torch.arange(size, device=self.ctx.device))
        return lp / lp.sum()

    def _on_write(
        self, env_ids: Tensor, slot: Tensor, qpos: Tensor, qvel: Tensor, extra: Tensor
    ) -> None:
        feats = self.ctx.task.state_features(qpos, qvel, extra).float()
        self._observe_features(feats, slot < self._capacity)
        self._depth[slot] = self._env_depth[env_ids]
        if self._step_model is not None:
            self._store_pairs(env_ids, feats)

    def _on_reset(self, env_ids: Tensor, ids: Tensor) -> None:
        parent = self._ic_depth[ids.clamp(min=0)] + 1
        self._env_depth[env_ids] = torch.where(
            ids >= 0, parent, torch.zeros_like(parent)
        )
        if self._step_model is not None:
            self._prev_ok[env_ids] = False  # never pair states across a reset

    def _on_outcomes(self, outcomes: Outcomes) -> None:
        if math.isnan(self._first_reward_step) and bool(outcomes.reached_reward.any()):
            self._visit_score.notify_first_reward()
            self._first_reward_step = float(self.ctx.counter.total)

    def _on_promote(self, targets: Tensor, candidates: Tensor) -> None:
        self._ic_depth[targets] = self._depth[candidates]

    def _observe_features(self, feats: Tensor, valid: Tensor) -> None:
        w = valid.float().unsqueeze(1)
        x = torch.where(valid.unsqueeze(1), feats, torch.zeros_like(feats))
        n_b = w.sum()
        mean_b = (x * w).sum(dim=0) / n_b.clamp(min=1.0)
        m2_b = (((x - mean_b) ** 2) * w).sum(dim=0)
        n = self._feat_n + n_b
        delta = mean_b - self._feat_mean
        self._feat_mean = self._feat_mean + delta * n_b / n.clamp(min=1.0)
        self._feat_m2 = (
            self._feat_m2 + m2_b + delta**2 * self._feat_n * n_b / n.clamp(min=1.0)
        )
        self._feat_n = n

    def _standardise(self, feats: Tensor) -> Tensor:
        var = self._feat_m2 / self._feat_n.clamp(min=1.0)
        return (feats - self._feat_mean) / var.sqrt().clamp(min=_STD_MIN)

    def _features(self, states: StartStates) -> Tensor:
        return self.ctx.task.state_features(
            states.qpos, states.qvel, states.extra
        ).float()

    def _ring_neighbour_counts(self, z_ic: Tensor, z_ring: Tensor) -> Tensor:
        rows = max(1, _CDIST_ELEMENTS_MAX // z_ring.shape[0])
        counts = torch.empty(z_ic.shape[0], dtype=torch.long, device=z_ic.device)
        for start in range(0, z_ic.shape[0], rows):
            dist = torch.cdist(z_ic[start : start + rows], z_ring)
            counts[start : start + rows] = (dist <= self._radius).sum(dim=1)
        return counts

    def _store_pairs(self, env_ids: Tensor, feats: Tensor) -> None:
        ok = torch.isfinite(feats).all(dim=1)
        pair = ok & self._prev_ok[env_ids]
        rank = pair.long().cumsum(0) - 1
        slot = torch.where(
            pair,
            (self._pair_ptr + rank) % _PAIR_CAPACITY,
            torch.full_like(rank, _PAIR_CAPACITY),
        )
        self._pair_a[slot] = self._prev_feat[env_ids]
        self._pair_b[slot] = feats
        added = pair.sum()
        self._pair_ptr = (self._pair_ptr + added) % _PAIR_CAPACITY
        self._pair_size = (self._pair_size + added).clamp(max=_PAIR_CAPACITY)
        self._prev_feat[env_ids] = feats
        self._prev_ok[env_ids] = ok

    def _train_step_model(self) -> None:
        assert self._step_model is not None
        n_pairs = int(self._pair_size)
        if n_pairs == 0:
            return
        pick = torch.randperm(n_pairs, generator=self.ctx.gen, device=self.ctx.device)
        pick = pick[:_PAIRS_PER_UPDATE]
        self._step_model.observe_transitions(
            self._standardise(self._pair_a[pick]), self._standardise(self._pair_b[pick])
        )

    def _across_wall_share(self, z_ic: Tensor, z_ring: Tensor) -> float:
        assert self._path_distance is not None
        gen, device = self.ctx.gen, self.ctx.device
        n_ic = z_ic.shape[0]
        anchors = torch.randperm(n_ic, generator=gen, device=device)[:_DIAG_ANCHORS]
        dist = torch.cdist(z_ic[anchors], z_ring)
        pairs = ((dist <= self._radius) & (dist > _SAME_STATE_DIST)).nonzero()
        if pairs.shape[0] == 0:
            return float("nan")
        if pairs.shape[0] > _DIAG_PAIRS_MAX:
            keep = torch.randperm(pairs.shape[0], generator=gen, device=device)
            pairs = pairs[keep[:_DIAG_PAIRS_MAX]]
        ring_xy = self._qpos[: z_ring.shape[0], :2]
        radius_m = self._radius * float(ring_xy.std(dim=0, unbiased=False).mean())
        path = self._path_distance(
            self.ctx.task,
            self._ic_qpos[anchors[pairs[:, 0]], :2].cpu(),
            ring_xy[pairs[:, 1]].cpu(),
        )
        return float((path > 2.0 * radius_m).float().mean())

"""M3 `visited_count`: draw from the visited buffer with probability proportional to the
`visit_count` score's lp, 1 / sqrt(count + 1) (Go-Explore, 2004.12919 L1632-1633), where the
count comes from one of three rules (shortlist section 3, method 3):

- `neighbour`: buffer states within `radius` of the state, on state features standardised by
  running mean and std (2609.08642 L257-260 counts this way for goals).
- `lineage_depth`: the number of restarts along the chain from the task start to the episode
  the state was visited in (a state visited in an episode that began at the task start has
  depth 0; one visited after restarting from a depth-d slot has depth d + 1). Deeper states
  are preferred: depth is a metric-free stand-in for "further from the start / rarer"
  (reset_curriculum_shortlist.md L100, section 3 method 3). Each recompute feeds the score
  the pseudo-count `max_depth - depth` (max over the filled slots), so lp = 1 / sqrt(max_depth
  - depth + 1) rises monotonically with depth and the deepest slots get lp = 1. The known
  weakness is the shortlist's: chains that wander in place also grow deep.
- `embedding`: neighbours within `radius` in the embedding of an internal `step_distance`
  model (2407.08464), trained on consecutive visited states of one episode.

Cost: counts are recomputed in `update` every `update_every` iterations, never in `reset`.
A neighbour count is chunked `torch.cdist` on device, O(buffer_size^2) per recompute. `reset`
draws by inverse CDF from a table built once per recompute, O(k log buffer_size) for k
resetting envs, so slots written since the last recompute are not drawn until the next one.
"""

import math
import sys
from typing import Literal

import torch
from torch import Tensor

from resetsweep.methods._slots import SlotBuffer
from resetsweep.methods.base import METHODS, MethodContext
from resetsweep.methods.visited import CAPACITY_DEFAULT
from resetsweep.scores.base import SCORES
from resetsweep.scores.side_models import StepDistance
from resetsweep.scores.visit_count import VisitCount
from resetsweep.types import Outcomes, RolloutStats, StartStates

CountRule = Literal["neighbour", "lineage_depth", "embedding"]
_COUNT_RULES: tuple[CountRule, ...] = ("neighbour", "lineage_depth", "embedding")

_CDIST_ELEMENTS_MAX = 2**25  # distance-matrix entries per chunk: 128 MiB in float32
_STD_MIN = 1e-6  # constant feature columns standardise to 0 instead of dividing by 0
_PAIR_CAPACITY = 16384  # consecutive-state pairs kept for the step-distance model
_PAIRS_PER_UPDATE = 4096
_DIAG_ANCHORS = 256  # states whose neighbours the across-wall diagnostic inspects
_DIAG_PAIRS_MAX = 65536


@METHODS.register("visited_count")
class VisitedCount(SlotBuffer):
    """Count-weighted draw over the visited buffer; see the module docstring.

    Takes only the `visit_count` score and feeds it with `set_counts`. When any outcome first
    has `reached_reward`, it calls `score.notify_first_reward()` and logs the step
    (`first_reward_step`, the total step count at that reset). For tasks whose module defines
    `path_distance` (the maze family) each recompute also logs `across_wall_neighbour_share`
    (only for the neighbour and embedding rules): over sampled neighbour pairs, the share whose
    path distance exceeds 2 x radius, with the radius turned into metres by multiplying with
    the mean standard deviation of the buffer's x and y. Embedding mode logs the step-distance
    training time as `side/wall_ms_per_update`.

    `offer_visited` adds no host sync of its own (beyond `task.is_valid`, as in `Visited`).
    `slot_probs()` is the distribution the score implies now; `reset` draws from the table
    built at the last recompute (see the module docstring).
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
        if update_every < 1:
            raise ValueError(f"update_every must be at least 1, got {update_every}")
        super().__init__(ctx, capacity=capacity, min_fill=min_fill)
        self._visit_score: VisitCount = score
        self._rule: CountRule = count
        self._radius = radius
        self._update_every = update_every
        d = ctx.device
        rows = capacity + 1
        canon = ctx.task.canonical_start()
        width = ctx.task.state_features(canon.qpos, canon.qvel, canon.extra).shape[1]
        self._counts = torch.zeros(rows, dtype=torch.long, device=d)
        self._depth = torch.zeros(rows, dtype=torch.long, device=d)
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
            model = SCORES.construct("step_distance", capacity=capacity, device=d)
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
        """`(buffer_size,)` lineage depth per filled slot (tracked under every count rule)."""
        return self._depth[: int(self._size)].clone()

    def counts(self) -> Tensor:
        """`(buffer_size,)` count fed to the score at the last recompute (0 for slots written
        since): the neighbour count, or the lineage pseudo-count `max_depth - depth`."""
        return self._counts[: int(self._size)].clone()

    def update(self, it: int, stats: RolloutStats) -> dict[str, float]:
        super().update(it, stats)
        if it % self._update_every != 0:
            return {}
        self._cdf = None
        size = int(self._size)
        if size == 0:
            return {}
        ids = torch.arange(size, device=self.ctx.device)
        if self._rule == "lineage_depth":
            depth = self._depth[:size]
            pseudo = depth.max() - depth  # deepest slot: count 0, the largest lp
            self._counts[:size] = pseudo
            self._visit_score.set_counts(ids, pseudo)
            self._cdf = self._draw_table(size)
            return {}
        z = self._standardise(self._buffer_features(size))
        if self._step_model is not None:
            self._train_step_model()
            z = self._step_model.embed(z)
        counts = self._neighbour_counts(z)
        self._counts[:size] = counts
        self._visit_score.set_counts(ids, counts)
        self._cdf = self._draw_table(size)
        if self._path_distance is not None:
            self._across_share = self._across_wall_share(z, size)
        return {}

    def diagnostics(self) -> dict[str, float]:
        diag = super().diagnostics()
        diag["first_reward_step"] = self._first_reward_step
        if self._path_distance is not None:
            diag["across_wall_neighbour_share"] = self._across_share
        if self._step_model is not None:
            wall_ms = self._step_model.extra_metrics()["wall_ms_per_update"]
            diag["side/wall_ms_per_update"] = wall_ms
        return diag

    def _draw(self, k: int, size: int) -> tuple[StartStates, Tensor]:
        """Inverse-CDF draw from the table built by the last recompute (or, before the first
        one, at the first draw): O(k log n) per reset. Slots written later join at the next
        recompute."""
        if self._cdf is None:
            self._cdf = self._draw_table(size)
        u = torch.rand(k, generator=self.ctx.gen, device=self.ctx.device)
        idx = torch.searchsorted(self._cdf, u * self._cdf[-1], right=True)
        idx = idx.clamp(max=self._cdf.numel() - 1)
        return StartStates(self._qpos[idx], self._qvel[idx], self._extra[idx]), idx

    def _draw_table(self, size: int) -> Tensor:
        return torch.cumsum(self._probs(size), dim=0)

    def _probs(self, size: int) -> Tensor:
        lp, _ = self._visit_score.values(torch.arange(size, device=self.ctx.device))
        return lp / lp.sum()

    def _on_write(
        self, env_ids: Tensor, slot: Tensor, qpos: Tensor, qvel: Tensor, extra: Tensor
    ) -> None:
        feats = self.ctx.task.state_features(qpos, qvel, extra).float()
        self._observe_features(feats, slot < self._capacity)
        self._counts[slot] = 0
        self._depth[slot] = self._env_depth[env_ids]
        if self._step_model is not None:
            self._store_pairs(env_ids, feats)

    def _on_reset(self, env_ids: Tensor, ids: Tensor) -> None:
        parent = self._depth[ids.clamp(min=0)] + 1
        self._env_depth[env_ids] = torch.where(
            ids >= 0, parent, torch.zeros_like(parent)
        )
        if self._step_model is not None:
            self._prev_ok[env_ids] = False  # never pair states across a reset

    def _on_outcomes(self, outcomes: Outcomes) -> None:
        if math.isnan(self._first_reward_step) and bool(outcomes.reached_reward.any()):
            self._visit_score.notify_first_reward()
            self._first_reward_step = float(self.ctx.counter.total)

    def _forget_slots(self, ids: Tensor) -> None:
        self._counts[ids] = 0

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

    def _buffer_features(self, size: int) -> Tensor:
        return self.ctx.task.state_features(
            self._qpos[:size], self._qvel[:size], self._extra[:size]
        ).float()

    def _neighbour_counts(self, z: Tensor) -> Tensor:
        size = z.shape[0]
        rows = max(1, _CDIST_ELEMENTS_MAX // size)
        counts = torch.empty(size, dtype=torch.long, device=z.device)
        for start in range(0, size, rows):
            dist = torch.cdist(z[start : start + rows], z)
            counts[start : start + rows] = (dist <= self._radius).sum(dim=1) - 1
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

    def _across_wall_share(self, z: Tensor, size: int) -> float:
        assert self._path_distance is not None
        gen, device = self.ctx.gen, self.ctx.device
        anchors = torch.randperm(size, generator=gen, device=device)[:_DIAG_ANCHORS]
        near = torch.cdist(z[anchors], z) <= self._radius
        near[torch.arange(anchors.numel(), device=device), anchors] = False
        pairs = near.nonzero()
        if pairs.shape[0] == 0:
            return float("nan")
        if pairs.shape[0] > _DIAG_PAIRS_MAX:
            keep = torch.randperm(pairs.shape[0], generator=gen, device=device)
            pairs = pairs[keep[:_DIAG_PAIRS_MAX]]
        xy = self._qpos[:size, :2]
        radius_m = self._radius * float(xy.std(dim=0, unbiased=False).mean())
        path = self._path_distance(
            self.ctx.task, xy[anchors[pairs[:, 0]]].cpu(), xy[pairs[:, 1]].cpu()
        )
        return float((path > 2.0 * radius_m).float().mean())

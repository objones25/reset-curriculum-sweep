"""S7: scores backed by a small learned side model.

Every side model owns its network and Adam optimiser and trains a fixed `steps_per_update`
gradient steps per update, entirely on device tensors (no per-step host sync). The wall
time of the last update is reported by `extra_metrics()["wall_ms_per_update"]`. The caller
must pass features that are already scaled to a sensible range (for example standardised),
and every model builds its networks on the first call that shows the feature width, using
torch's global RNG, so seed torch before constructing for reproducible runs.

`SuccessClassifier` and `RND` read `RolloutStats.start_features`, shape `(capacity, f)`, row
`i` the features of IC `i`'s start state, valid for every IC. `StepDistance` is not a
per-IC score: it exposes `embed` for `visited_count`'s near / not-near test and is trained
by `observe_transitions`.
"""

import time

import torch
from torch import Tensor, nn

from resetsweep.scores.base import SCORES, BaseScore, ICTable, median_fill
from resetsweep.types import Outcomes, RolloutStats


def band_lp(p: Tensor, band: tuple[float, float]) -> Tensor:
    """1 where `p` is inside `band`, falling linearly to 0 over a distance of
    `max(lo, 1 - hi)` outside it (0.3 for the default band, so p = 0 or 1 scores 0).

    The band (0.3, 0.7) is the success-probability range of the self-supervised curriculum
    paper (papers/text/2311.09195 Table I L257, and L353: allowed initial states have
    success probability above 0.3 and below 0.7). The linear falloff is this project's choice.
    """
    lo, hi = band
    dist = (lo - p).clamp(min=0.0) + (p - hi).clamp(min=0.0)
    return (1.0 - dist / max(lo, 1.0 - hi)).clamp(min=0.0)


def _mlp(width_in: int, hidden: int, width_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width_in, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, width_out),
    )


class _SideModel(BaseScore):
    """Shared plumbing: feature checks, per-update timing, rollout id handling."""

    def __init__(
        self,
        *,
        capacity: int,
        device: str,
        hidden: int,
        lr: float,
        steps_per_update: int,
        beta: float,
    ) -> None:
        if capacity <= 0 or hidden <= 0 or steps_per_update <= 0:
            raise ValueError(
                "capacity, hidden and steps_per_update must be positive, got "
                f"{capacity}, {hidden}, {steps_per_update}"
            )
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        self._set_beta(beta)
        self._capacity = capacity
        self._device = device
        self._hidden = hidden
        self._lr = lr
        self._steps = steps_per_update
        self._wall_ms = 0.0
        self._width: int | None = None

    def extra_metrics(self) -> dict[str, float]:
        return {"wall_ms_per_update": self._wall_ms}

    def _sync(self) -> None:
        if torch.device(self._device).type == "cuda":
            torch.cuda.synchronize(self._device)

    def _start_timer(self) -> float:
        self._sync()
        return time.perf_counter()

    def _stop_timer(self, start: float) -> None:
        # Sync only here, at the measurement point, so training itself never blocks the host.
        self._sync()
        self._wall_ms = (time.perf_counter() - start) * 1000.0

    def _check_width(self, features: Tensor) -> None:
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D (n, f), got shape {tuple(features.shape)}"
            )
        if self._width is not None and features.shape[1] != self._width:
            raise ValueError(
                f"feature width changed from {self._width} to {features.shape[1]}"
            )

    def _rollout_features(self, stats: RolloutStats) -> Tensor:
        features = stats.start_features
        self._check_width(features)
        if features.shape[0] != self._capacity:
            raise ValueError(
                f"start_features needs one row per IC ({self._capacity}), "
                f"got {features.shape[0]}"
            )
        return features.to(self._device, torch.float32)

    def _present(self, ic_ids: Tensor) -> Tensor:
        """`(capacity,)` bool: which ICs appear in the rollout. `-1` is ignored."""
        flat = ic_ids.reshape(-1).to(self._device)
        if flat.numel() and bool(((flat < -1) | (flat >= self._capacity)).any()):
            raise ValueError(f"ic_ids must be -1 or in [0, {self._capacity})")
        present = torch.zeros(self._capacity + 1, dtype=torch.bool, device=self._device)
        present[torch.where(flat >= 0, flat, torch.full_like(flat, self._capacity))] = (
            True
        )
        return present[: self._capacity]

    def _check_ids(self, ic_ids: Tensor) -> None:
        if ic_ids.numel() and bool(((ic_ids < 0) | (ic_ids >= self._capacity)).any()):
            raise ValueError(f"ic_ids must be in [0, {self._capacity})")


@SCORES.register("success_classifier")
class SuccessClassifier(_SideModel):
    """MLP on start features predicting an IC's success rate; lp is high where the predicted
    probability sits inside `band` (see `band_lp`), the self-supervised curriculum paper's
    "intermediate success probability" criterion (papers/text/2311.09195 L353).

    Training labels are each IC's windowed success rate from `record`, with a weighted BCE
    over the ICs that have at least one recorded outcome. `observe_rollout` stores the
    features and runs `steps_per_update` full-batch Adam steps. Until any success has been
    recorded every IC is untrusted (so lp is flat 0); afterwards every IC is trusted, since
    the classifier can score any feature row.
    """

    def __init__(
        self,
        *,
        capacity: int,
        device: str,
        band: tuple[float, float] = (0.3, 0.7),
        hidden: int = 64,
        lr: float = 1e-3,
        steps_per_update: int = 50,
        history: int = 100,
        beta: float = 0.1,
    ) -> None:
        super().__init__(
            capacity=capacity,
            device=device,
            hidden=hidden,
            lr=lr,
            steps_per_update=steps_per_update,
            beta=beta,
        )
        if not 0.0 <= band[0] < band[1] <= 1.0:
            raise ValueError(f"band must satisfy 0 <= lo < hi <= 1, got {band}")
        self._band = band
        self._table = ICTable(capacity, device, history=history)
        self._all_ids = torch.arange(capacity, device=device)
        self._net: nn.Sequential | None = None
        self._opt: torch.optim.Adam | None = None
        self._features = torch.zeros(capacity, 0, device=device)

    def record(self, outcomes: Outcomes) -> None:
        self._table.record(outcomes)

    def forget(self, ic_ids: Tensor) -> None:
        self._table.forget(ic_ids)

    def observe_rollout(self, stats: RolloutStats) -> None:
        features = self._rollout_features(stats)
        self._present(stats.ic_ids)  # bounds check only; labels come from the table
        if self._net is None:
            self._width = features.shape[1]
            self._net = _mlp(self._width, self._hidden, 1).to(self._device)
            self._opt = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        assert self._opt is not None
        self._features = features
        start = self._start_timer()
        labels = self._table.success_rate(self._all_ids)
        weight = (self._table.count(self._all_ids) > 0).to(torch.float32)
        norm = weight.sum().clamp(min=1.0)
        for _ in range(self._steps):
            logits = self._net(features).squeeze(-1)
            per_ic = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            loss = (per_ic * weight).sum() / norm
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()
        self._stop_timer(start)

    def success_prob(self, ic_ids: Tensor) -> Tensor:
        """Predicted success probability from the features stored at the last rollout."""
        if self._net is None:
            raise RuntimeError("success_prob needs one observe_rollout call first")
        self._check_ids(ic_ids)
        with torch.no_grad():
            return torch.sigmoid(self._net(self._features[ic_ids]).squeeze(-1))

    def values(self, ic_ids: Tensor) -> tuple[Tensor, Tensor]:
        self._check_ids(ic_ids)
        if self._net is None:
            zeros = torch.zeros(ic_ids.shape, device=self._device)
            return zeros, torch.zeros_like(zeros, dtype=torch.bool)
        any_success = self._table.ever_succeeded(self._all_ids).any()
        trusted = any_success.expand(ic_ids.shape)
        lp = band_lp(self.success_prob(ic_ids), self._band)
        return median_fill(lp, trusted), trusted


@SCORES.register("rnd")
class RND(_SideModel):
    """Random network distillation, following Burda et al. 2018 (not verified: no local
    copy of the paper). A fixed random target net and a trained predictor map start features
    to `out` dims; lp is the predictor's mean squared error on an IC's features.

    `observe_rollout` trains the predictor on the ICs that appear in the rollout, so ICs
    never started from keep a high error (novelty). All ICs are trusted after the first
    rollout. `forget` does nothing: novelty comes from the features, which the next
    `observe_rollout` refreshes for a replaced IC.
    """

    def __init__(
        self,
        *,
        capacity: int,
        device: str,
        hidden: int = 64,
        out: int = 32,
        lr: float = 1e-3,
        steps_per_update: int = 50,
        beta: float = 0.1,
    ) -> None:
        super().__init__(
            capacity=capacity,
            device=device,
            hidden=hidden,
            lr=lr,
            steps_per_update=steps_per_update,
            beta=beta,
        )
        if out <= 0:
            raise ValueError(f"out must be positive, got {out}")
        self._out = out
        self._target: nn.Sequential | None = None
        self._predictor: nn.Sequential | None = None
        self._opt: torch.optim.Adam | None = None
        self._features = torch.zeros(capacity, 0, device=device)

    def record(self, outcomes: Outcomes) -> None:
        del outcomes

    def forget(self, ic_ids: Tensor) -> None:
        self._check_ids(ic_ids)

    def _error(
        self, features: Tensor, target: nn.Sequential, pred: nn.Sequential
    ) -> Tensor:
        with torch.no_grad():
            tgt = target(features)
        return ((pred(features) - tgt) ** 2).mean(dim=-1)

    def observe_rollout(self, stats: RolloutStats) -> None:
        features = self._rollout_features(stats)
        present = self._present(stats.ic_ids)
        if self._target is None or self._predictor is None:
            self._width = features.shape[1]
            self._target = _mlp(self._width, self._hidden, self._out).to(self._device)
            for param in self._target.parameters():
                param.requires_grad_(False)
            self._predictor = _mlp(self._width, self._hidden, self._out).to(
                self._device
            )
            self._opt = torch.optim.Adam(self._predictor.parameters(), lr=self._lr)
        assert self._opt is not None
        self._features = features
        start = self._start_timer()
        weight = present.to(torch.float32)
        norm = weight.sum().clamp(min=1.0)
        for _ in range(self._steps):
            loss = (
                self._error(features, self._target, self._predictor) * weight
            ).sum() / norm
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()
        self._stop_timer(start)

    def values(self, ic_ids: Tensor) -> tuple[Tensor, Tensor]:
        self._check_ids(ic_ids)
        if self._target is None or self._predictor is None:
            zeros = torch.zeros(ic_ids.shape, device=self._device)
            return zeros, torch.zeros_like(zeros, dtype=torch.bool)
        with torch.no_grad():
            lp = self._error(self._features[ic_ids], self._target, self._predictor)
        return lp, torch.ones_like(lp, dtype=torch.bool)


@SCORES.register("step_distance")
class StepDistance(_SideModel):
    """Encoder whose embedding distance tracks how many steps apart two states are, used by
    `visited_count` only for near / not-near decisions (shortlist section 4.7), never as a
    per-IC score: `values` is flat 0 and untrusted.

    Reference: TLDR (papers/text/2407.08464 L194-203) learns phi so that the distance
    between consecutive states is at most 1 while other pairs are pushed apart (a QRL
    constrained objective solved by dual gradient descent). This is the simpler margin
    form of the same idea, not the dual-gradient one: for a batch of consecutive pairs it
    minimises `mean ||phi(s) - phi(s')||^2` (the transition constraint) plus
    `mean relu(margin - ||phi(s) - phi(g)||)^2` over `g` drawn by shuffling the batch
    (the push-apart term). Unlike TLDR it does not learn a graded distance beyond `margin`.

    `observe_transitions(features_t, features_t1)` takes `(n, f)` tensors of consecutive
    visited states, is called by train.py (Task 16), and runs `steps_per_update` Adam steps
    on that batch. `embed(features)` returns `(n, emb)`; before any training the encoder is
    randomly initialised.
    """

    def __init__(
        self,
        *,
        capacity: int,
        device: str,
        emb: int = 16,
        margin: float = 1.0,
        hidden: int = 64,
        lr: float = 1e-3,
        steps_per_update: int = 50,
        beta: float = 0.1,
    ) -> None:
        super().__init__(
            capacity=capacity,
            device=device,
            hidden=hidden,
            lr=lr,
            steps_per_update=steps_per_update,
            beta=beta,
        )
        if emb <= 0 or margin <= 0.0:
            raise ValueError(f"emb and margin must be positive, got {emb}, {margin}")
        self._emb = emb
        self._margin = margin
        self._net: nn.Sequential | None = None
        self._opt: torch.optim.Adam | None = None

    def record(self, outcomes: Outcomes) -> None:
        del outcomes

    def forget(self, ic_ids: Tensor) -> None:
        self._check_ids(ic_ids)

    def values(self, ic_ids: Tensor) -> tuple[Tensor, Tensor]:
        self._check_ids(ic_ids)
        zeros = torch.zeros(ic_ids.shape, device=self._device)
        return zeros, torch.zeros_like(zeros, dtype=torch.bool)

    def _ensure_net(self, width: int) -> nn.Sequential:
        if self._net is None:
            self._width = width
            self._net = _mlp(width, self._hidden, self._emb).to(self._device)
            self._opt = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        return self._net

    def embed(self, features: Tensor) -> Tensor:
        self._check_width(features)
        net = self._ensure_net(features.shape[1])
        with torch.no_grad():
            return net(features.to(self._device, torch.float32))

    def observe_transitions(self, features_t: Tensor, features_t1: Tensor) -> None:
        if features_t.shape != features_t1.shape:
            raise ValueError(
                "features_t and features_t1 must have the same shape, got "
                f"{tuple(features_t.shape)} and {tuple(features_t1.shape)}"
            )
        self._check_width(features_t)
        net = self._ensure_net(features_t.shape[1])
        assert self._opt is not None
        a = features_t.to(self._device, torch.float32)
        b = features_t1.to(self._device, torch.float32)
        start = self._start_timer()
        for _ in range(self._steps):
            za, zb = net(a), net(b)
            near = ((za - zb) ** 2).sum(dim=-1).mean()
            other = za[torch.randperm(za.shape[0], device=self._device)]
            far_dist = (((za - other) ** 2).sum(dim=-1) + 1e-8).sqrt()
            push = (self._margin - far_dist).clamp(min=0.0).pow(2).mean()
            loss = near + push
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()
        self._stop_timer(start)

"""Statistics for the sweep ranking (spec section 6): IQM, bootstrap intervals, normalisation
and the rung-0 tie-break key.

Conventions, with sources:

- IQM is the 25% trimmed mean: drop the bottom and top 25% of runs and average the rest
  (2108.13264 lines 536-538, "Also called 25% trimmed mean"). The paper's parenthetical count
  ``floor(NM/2)`` retained runs (line 538) does not match its own "25% trimmed" wording for
  odd-ish n (n=5 would keep 2, trimming 1.5 per side). We follow the trimmed-mean definition:
  ``floor(0.25 n)`` dropped from each end, as ``scipy.stats.trim_mean(x, 0.25)`` and the oracle
  ``robot-learning-expert/scripts/run_stats.py`` do. So 3 runs keep all 3 (IQM == mean), 5 keep 3.
- The stratified bootstrap resamples runs with replacement independently inside each task,
  recomputes the statistic on the pooled sample, and takes percentile limits
  (2108.13264 lines 431-433).
- IQM can hide a catastrophic seed and is most useful across collections of tasks
  (2304.01315 lines 753-758); the sweep reports seeds that end at zero next to it.

Failed runs: `rung0_key` filters on ``status == "ok"`` itself. `iqm` and the bootstrap helpers
take plain score arrays, so callers pass scores from ok runs only. Every function raises
`ValueError` on empty input rather than returning a default.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from resetsweep.train import RunResult

_TRIM = 0.25


def _as_scores(x: object, *, what: str = "scores") -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"empty {what}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"non-finite value in {what}")
    return arr


def iqm(x: np.ndarray) -> float:
    """Mean of the middle 50% of `x` (25% trimmed mean); see the module docstring."""
    arr = np.sort(_as_scores(x))
    n = arr.size
    lo, hi = math.floor(_TRIM * n), math.ceil((1 - _TRIM) * n)
    return float(arr[lo:hi].mean())


def stratified_bootstrap_ci(
    scores_by_task: Mapping[str, np.ndarray],
    *,
    stat: Callable[[np.ndarray], float] = iqm,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI of `stat` over the pooled runs, resampling seeds within each task.

    Deterministic given `seed` (``numpy.random.default_rng``).
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0.0 < alpha < 1.0:  # also False for NaN
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not scores_by_task:
        raise ValueError("empty scores_by_task")
    arrays = [
        _as_scores(v, what=f"scores for task {k!r}") for k, v in scores_by_task.items()
    ]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        resampled = [a[rng.integers(a.size, size=a.size)] for a in arrays]
        boots[b] = stat(np.concatenate(resampled))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_ci(
    x: np.ndarray,
    *,
    stat: Callable[[np.ndarray], float] = iqm,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat` over one sample (a single stratum)."""
    return stratified_bootstrap_ci(
        {"all": x}, stat=stat, n_boot=n_boot, alpha=alpha, seed=seed
    )


def normalise(score: float, *, low: float, high: float) -> float:
    """Map `low` (the `uniform` arm) to 0 and `high` (the best arm) to 1.

    When ``high == low`` there is no scale: returns 0.0 for ``score == low`` and raises
    `ValueError` for any other score, so a degenerate task cannot silently produce NaN or inf.
    Scores outside ``[low, high]`` extrapolate linearly (an arm worse than uniform is negative).
    """
    if not all(math.isfinite(v) for v in (score, low, high)):
        raise ValueError(f"non-finite input: score={score}, low={low}, high={high}")
    if high == low:
        if score == low:
            return 0.0
        raise ValueError(f"cannot normalise score {score} when high == low == {low}")
    return (score - low) / (high - low)


def rung0_key(results: Sequence[RunResult]) -> tuple[float, float, float, float]:
    """Rung-0 ranking key (spec section 6, ties at zero), compared lexicographically.

    ``(iqm auc, share reached_reward, -median steps_to_first_reward, share ever_succeeded)``

    - Only ``status == "ok"`` results count; raises `ValueError` if none remain.
    - A run with ``steps_to_first_reward is None`` counts as infinite steps, so a median of
      infinity gives ``-inf`` in the key (worse than any finite step count).
    - ``ever_succeeded_share`` is NaN for methods without per-IC outcomes. NaN entries are
      dropped from the mean; if none are finite the component is ``-inf``: "no information"
      ranks below every finite share, and the key never contains NaN, so it stays a total order.
    """
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        raise ValueError("no ok runs to rank")
    auc_iqm = iqm(np.array([r.auc for r in ok]))
    reached = sum(r.reached_reward for r in ok) / len(ok)
    steps = [
        math.inf if r.steps_to_first_reward is None else float(r.steps_to_first_reward)
        for r in ok
    ]
    ever = [
        r.ever_succeeded_share for r in ok if not math.isnan(r.ever_succeeded_share)
    ]
    ever_mean = sum(ever) / len(ever) if ever else -math.inf
    return (auc_iqm, reached, -float(np.median(steps)), ever_mean)

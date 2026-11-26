"""Statistics for the sweep ranking (spec section 6).

Expected values for `iqm` and the bootstrap were frozen only after `resetsweep.sweep.stats`
and the oracle `~/.claude/skills/robot-learning-expert/scripts/run_stats.py` agreed on the same
inputs (same seed, same draw order, so the intervals match exactly, not just approximately).
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from resetsweep.sweep.stats import (
    bootstrap_ci,
    iqm,
    normalise,
    rung0_key,
    stratified_bootstrap_ci,
)
from resetsweep.train import RunResult

NAN = float("nan")
FINITE = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


def _run(
    *,
    auc: float = 0.0,
    reached: bool = False,
    steps: int | None = None,
    ever: float = NAN,
    status: str = "ok",
) -> RunResult:
    return RunResult(
        arm_id="a",
        seed=0,
        status=status,  # type: ignore[arg-type]
        error=None if status == "ok" else "boom",
        auc=auc,
        final_success=0.0,
        reached_reward=reached,
        steps_to_first_reward=steps,
        ever_succeeded_share=ever,
        env_steps_train=0,
        env_steps_extra={},
        wall_s=0.0,
    )


# --- iqm ---------------------------------------------------------------------------------------


def test_iqm_of_known_array() -> None:
    assert iqm(np.arange(1.0, 9.0)) == pytest.approx(4.5)


def test_iqm_with_five_runs_keeps_the_middle_three() -> None:
    # floor(0.25 * 5) = 1 dropped from each end, as scipy.stats.trim_mean(0.25) and the oracle do
    assert iqm(np.array([0.0, 1.0, 2.0, 3.0, 100.0])) == pytest.approx(2.0)


def test_iqm_with_three_runs_equals_the_mean() -> None:
    assert iqm(np.array([1.0, 2.0, 6.0])) == pytest.approx(3.0)


def test_iqm_robust_to_outlier() -> None:
    clean = np.arange(1.0, 9.0)
    dirty = np.append(clean[:-1], 1000.0)
    assert iqm(dirty) == pytest.approx(4.5)
    assert dirty.mean() > 100.0


def test_iqm_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        iqm(np.array([]))


def test_iqm_rejects_nan() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        iqm(np.array([1.0, NAN, 3.0]))


@given(st.lists(FINITE, min_size=1, max_size=40), st.randoms(use_true_random=False))
def test_iqm_is_permutation_invariant(values: list[float], rnd) -> None:
    shuffled = list(values)
    rnd.shuffle(shuffled)
    assert iqm(np.array(values)) == pytest.approx(iqm(np.array(shuffled)))


@given(st.lists(FINITE, min_size=1, max_size=40))
def test_iqm_lies_between_min_and_max(values: list[float]) -> None:
    result = iqm(np.array(values))
    assert min(values) - 1e-6 <= result <= max(values) + 1e-6


# --- bootstrap ---------------------------------------------------------------------------------


def test_bootstrap_ci_matches_oracle_frozen_values() -> None:
    lo, hi = bootstrap_ci(np.arange(1.0, 9.0), seed=0)
    assert (lo, hi) == pytest.approx((2.5, 6.5))


def test_bootstrap_ci_is_deterministic_given_seed() -> None:
    x = np.arange(1.0, 9.0)
    assert bootstrap_ci(x, seed=3) == bootstrap_ci(x, seed=3)


def test_bootstrap_ci_differs_across_seeds() -> None:
    x = np.random.default_rng(1).normal(size=12)
    assert bootstrap_ci(x, seed=1) != bootstrap_ci(x, seed=2)


def test_bootstrap_ci_contains_point_estimate_and_narrows_with_n() -> None:
    big = np.random.default_rng(123).normal(size=80)
    small = big[:8]
    lo_s, hi_s = bootstrap_ci(small, seed=0)
    lo_b, hi_b = bootstrap_ci(big, seed=0)
    assert lo_s <= iqm(small) <= hi_s
    assert lo_b <= iqm(big) <= hi_b
    # oracle: small (-0.50212515, 0.82578446), big (-0.19728122, 0.26243466)
    assert (lo_s, hi_s) == pytest.approx((-0.5021251489194318, 0.8257844583347885))
    assert (lo_b, hi_b) == pytest.approx((-0.19728122025102907, 0.2624346820208628))
    assert hi_b - lo_b < hi_s - lo_s


def test_bootstrap_ci_accepts_a_custom_statistic() -> None:
    lo, hi = bootstrap_ci(np.full(6, 2.0), stat=np.mean, seed=0)
    assert (lo, hi) == (2.0, 2.0)


def test_stratified_resamples_within_task() -> None:
    # Task a is all 0s and task b all 10s. Resampling inside each task always yields 4 zeros and
    # 4 tens, so the pooled mean is exactly 5 in every replicate. Pooled resampling would not.
    by_task = {"a": np.zeros(4), "b": np.full(4, 10.0)}
    assert stratified_bootstrap_ci(by_task, stat=np.mean, seed=0) == (5.0, 5.0)


def test_stratified_bootstrap_matches_oracle_frozen_values() -> None:
    by_task = {
        "a": np.array([0.0, 1.0, 2.0, 3.0]),
        "b": np.array([10.0, 11.0, 12.0, 13.0]),
    }
    lo, hi = stratified_bootstrap_ci(by_task, seed=7)
    assert (lo, hi) == pytest.approx((5.74375, 7.25))


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, NAN])
def test_bootstrap_ci_rejects_bad_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(np.arange(5.0), alpha=alpha)


def test_bootstrap_ci_rejects_nonpositive_n_boot() -> None:
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci(np.arange(5.0), n_boot=0)


def test_bootstrap_ci_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci(np.array([]))


def test_stratified_rejects_empty_task_and_empty_dict() -> None:
    with pytest.raises(ValueError, match="empty"):
        stratified_bootstrap_ci({"a": np.array([1.0]), "b": np.array([])})
    with pytest.raises(ValueError, match="empty"):
        stratified_bootstrap_ci({})


# --- normalise ---------------------------------------------------------------------------------


def test_normalise_maps_uniform_to_zero_best_to_one() -> None:
    assert normalise(0.2, low=0.2, high=0.8) == 0.0
    assert normalise(0.8, low=0.2, high=0.8) == 1.0
    assert normalise(0.5, low=0.2, high=0.8) == pytest.approx(0.5)


def test_normalise_allows_scores_below_uniform_to_go_negative() -> None:
    assert normalise(0.1, low=0.2, high=0.6) == pytest.approx(-0.25)


def test_normalise_degenerate_range_returns_zero_at_low() -> None:
    assert normalise(0.3, low=0.3, high=0.3) == 0.0


@pytest.mark.parametrize("score", [0.0, 0.31])
def test_normalise_degenerate_range_raises_for_other_scores(score: float) -> None:
    with pytest.raises(ValueError, match="high == low"):
        normalise(score, low=0.3, high=0.3)


def test_normalise_rejects_nan_inputs() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        normalise(NAN, low=0.0, high=1.0)


# --- rung0_key ---------------------------------------------------------------------------------


def test_rung0_key_field_order_and_values() -> None:
    runs = [
        _run(auc=0.2, reached=True, steps=100, ever=0.5),
        _run(auc=0.4, reached=False, steps=None, ever=0.25),
        _run(auc=0.6, reached=True, steps=300, ever=0.75),
    ]
    # iqm of 3 runs = mean 0.4; share reached 2/3; median steps of (100, inf, 300) = 300; ever mean 0.5
    assert rung0_key(runs) == pytest.approx((0.4, 2 / 3, -300.0, 0.5))


def test_rung0_key_breaks_zero_auc_ties_by_reached_reward_then_steps() -> None:
    none = [_run(), _run(), _run()]
    some = [_run(reached=True, steps=900), _run(), _run()]
    fast = [_run(reached=True, steps=100), _run(reached=True, steps=100), _run()]
    slow = [_run(reached=True, steps=900), _run(reached=True, steps=900), _run()]
    assert rung0_key(none) < rung0_key(some) < rung0_key(slow) < rung0_key(fast)


def test_rung0_key_breaks_remaining_tie_by_ever_succeeded_share() -> None:
    low = [_run(ever=0.1), _run(ever=0.1)]
    high = [_run(ever=0.4), _run(ever=0.4)]
    assert rung0_key(low) < rung0_key(high)


def test_rung0_key_auc_dominates_later_components() -> None:
    better_auc = [_run(auc=0.2)]
    better_tiebreaks = [_run(auc=0.1, reached=True, steps=1, ever=1.0)]
    assert rung0_key(better_tiebreaks) < rung0_key(better_auc)


def test_rung0_key_never_reached_means_infinite_steps() -> None:
    key = rung0_key([_run(steps=None), _run(steps=None)])
    assert key[2] == -math.inf


def test_rung0_key_nan_ever_succeeded_ranks_below_any_finite_value() -> None:
    no_info = rung0_key([_run(ever=NAN), _run(ever=NAN)])
    zero = rung0_key([_run(ever=0.0), _run(ever=0.0)])
    assert no_info[3] == -math.inf
    assert no_info < zero
    assert not any(math.isnan(c) for c in no_info)


def test_rung0_key_ever_succeeded_averages_only_finite_entries() -> None:
    assert rung0_key([_run(ever=NAN), _run(ever=0.5)])[3] == pytest.approx(0.5)


def test_rung0_key_is_a_total_order_over_mixed_nan_inputs() -> None:
    keys = [
        rung0_key([_run(ever=NAN)]),
        rung0_key([_run(ever=0.0)]),
        rung0_key([_run(ever=0.3, reached=True, steps=5)]),
        rung0_key([_run(steps=None)]),
    ]
    assert sorted(keys) == sorted(keys, reverse=True)[::-1]
    for a in keys:
        for b in keys:
            assert a < b or a == b or a > b


def test_rung0_key_ignores_failed_runs() -> None:
    ok = [_run(auc=0.5)]
    with_failed = ok + [_run(auc=99.0, status="failed")]
    assert rung0_key(with_failed) == rung0_key(ok)


def test_rung0_key_raises_when_no_ok_runs() -> None:
    with pytest.raises(ValueError, match="no ok runs"):
        rung0_key([_run(status="failed")])
    with pytest.raises(ValueError, match="no ok runs"):
        rung0_key([])


def test_rung0_key_rejects_nan_auc() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        rung0_key([_run(auc=NAN)])


# --- group keys: NaN ever_succeeded_share is neutral (final-review I4) ---------------------------


def test_group_keys_drop_ever_succeeded_when_any_arm_has_no_ics() -> None:
    from resetsweep.sweep.stats import rung0_group_keys

    keys = rung0_group_keys(
        {
            "uniform": [_run(ever=NAN)],
            "walk_low": [_run(ever=0.1)],
            "walk_high": [_run(ever=0.9)],
        }
    )

    # Same AUC, reward share and steps: without the dropped component all three tie.
    assert (
        keys["uniform"]
        == keys["walk_low"]
        == keys["walk_high"]
        == (0.0, 0.0, -math.inf)
    )


def test_group_keys_keep_ever_succeeded_when_every_arm_has_it() -> None:
    from resetsweep.sweep.stats import rung0_group_keys

    keys = rung0_group_keys({"low": [_run(ever=0.1)], "high": [_run(ever=0.9)]})

    assert keys["low"] < keys["high"]
    assert keys["high"] == pytest.approx((0.0, 0.0, -math.inf, 0.9))


def test_group_keys_still_order_by_earlier_components_with_nan_present() -> None:
    from resetsweep.sweep.stats import rung0_group_keys

    keys = rung0_group_keys(
        {"none": [_run(ever=NAN)], "reached": [_run(ever=0.1, reached=True, steps=5)]}
    )

    assert keys["none"] < keys["reached"]
    assert not any(math.isnan(c) for k in keys.values() for c in k)

from collections.abc import Callable

import pytest
import torch

from resetsweep.scores import SCORES
from resetsweep.scores.slope import Slope
from resetsweep.types import Outcomes

MakeOutcomes = Callable[..., Outcomes]


def _ids(*ids: int) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int64)


def _rise_and_fall(make_outcomes: MakeOutcomes) -> Outcomes:
    """IC 0 returns 1..5 (rising), IC 1 returns 5..1 (falling), interleaved in one call."""
    rows: list[tuple[int, bool, float]] = []
    for k in range(5):
        rows.append((0, False, float(k + 1)))
        rows.append((1, False, float(5 - k)))
    return make_outcomes(rows)


def test_slope_positive_for_rising_returns_and_abs_counts_falling(
    make_outcomes: MakeOutcomes,
) -> None:
    signed = Slope(capacity=2, device="cpu", form="signed", window=5)
    absolute = Slope(capacity=2, device="cpu", form="abs", window=5)
    for score in (signed, absolute):
        score.record(_rise_and_fall(make_outcomes))
    # Least-squares slope of 1..5 against 0..4 is exactly +1; of 5..1 it is -1.
    lp_signed, _ = signed.values(_ids(0, 1))
    lp_abs, _ = absolute.values(_ids(0, 1))
    assert lp_signed.tolist() == pytest.approx([1.0, -1.0])
    assert lp_abs.tolist() == pytest.approx([1.0, 1.0])


def test_slope_uses_only_the_last_window_outcomes(make_outcomes: MakeOutcomes) -> None:
    score = Slope(capacity=1, device="cpu", window=3, history=10)
    # Flat at 10 then rising 0, 2, 4: only the last three count -> slope 2.
    rows = [(0, False, 10.0)] * 4 + [(0, False, r) for r in (0.0, 2.0, 4.0)]
    score.record(make_outcomes(rows))
    lp, _ = score.values(_ids(0))
    assert lp.tolist() == pytest.approx([2.0])


def test_slope_untrusted_below_two_outcomes_gets_median(
    make_outcomes: MakeOutcomes,
) -> None:
    score = Slope(capacity=3, device="cpu", window=5)
    rows = [(0, False, float(k)) for k in range(1, 4)]  # slope 1 over 3 points
    rows += [(1, False, 7.0)]  # a single point has no slope
    score.record(make_outcomes(rows))
    lp, trusted = score.values(_ids(0, 1, 2))
    assert trusted.tolist() == [True, False, False]
    assert lp.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_fast_minus_slow_positive_after_jump(make_outcomes: MakeOutcomes) -> None:
    score = Slope(
        capacity=1, device="cpu", form="fast_minus_slow", history=4, fast=0.5, slow=0.1
    )
    score.record(make_outcomes([(0, s, 0.0) for s in (False, False, True, True)]))
    lp, _ = score.values(_ids(0))
    # Ages (oldest..newest) 3,2,1,0.  fast weights 0.5**age = .125,.25,.5,1 (sum 1.875)
    # -> avg 1.5/1.875 = 0.8.  slow weights 0.9**age = .729,.81,.9,1 (sum 3.439)
    # -> avg 1.9/3.439.  Difference = 0.8 - 1.9/3.439.
    assert lp.tolist() == pytest.approx([0.8 - 1.9 / 3.439])
    assert lp.item() > 0.0


def test_fast_minus_slow_is_zero_for_constant_success(
    make_outcomes: MakeOutcomes,
) -> None:
    score = Slope(capacity=1, device="cpu", form="fast_minus_slow", history=5)
    score.record(make_outcomes([(0, True, 0.0)] * 5))
    lp, _ = score.values(_ids(0))
    assert lp.tolist() == pytest.approx([0.0], abs=1e-6)


def test_forget_resets_history(make_outcomes: MakeOutcomes) -> None:
    score = Slope(capacity=2, device="cpu", window=5)
    score.record(_rise_and_fall(make_outcomes))
    score.forget(_ids(0))
    _, trusted = score.values(_ids(0, 1))
    assert trusted.tolist() == [False, True]


def test_registry_constructs_slope() -> None:
    score = SCORES.construct("slope", capacity=4, device="cpu", form="abs")
    assert isinstance(score, Slope)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"form": "nope"}, "form"),
        ({"window": 1}, "window"),
        ({"window": 20, "history": 10}, "window"),
        ({"fast": 0.0}, "fast"),
        ({"slow": 1.5}, "slow"),
        ({"fast": 0.01, "slow": 0.1}, "fast"),
    ],
)
def test_rejects_invalid_settings(settings: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Slope(capacity=2, device="cpu", **settings)  # type: ignore[arg-type]  # deliberate bad input

import math
from collections.abc import Callable

import pytest
import torch

from resetsweep.scores import SCORES
from resetsweep.scores.success_band import SuccessBand
from resetsweep.types import Outcomes

MakeOutcomes = Callable[..., Outcomes]


def _ids(*ids: int) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int64)


def _rates(make_outcomes: MakeOutcomes, successes: list[int], total: int) -> Outcomes:
    """IC i gets `total` outcomes, the first `successes[i]` of them successes."""
    rows = [(ic, k < s, 0.0) for ic, s in enumerate(successes) for k in range(total)]
    return make_outcomes(rows)


def _sgs_logit(p: float, *, t: float, kappa: float, eps: float) -> float:
    w = (p + eps) ** (kappa * t) * (1 - p + eps) ** (kappa * (1 - t))
    return math.log(w) + eps


def test_pq_score_peaks_at_half_success(make_outcomes: MakeOutcomes) -> None:
    score = SuccessBand(capacity=3, device="cpu", form="pq")
    # 4 outcomes each; rates 0, 0.5, 1 -> p(1-p) = 0, 0.25, 0.
    score.record(_rates(make_outcomes, [0, 2, 4], total=4))
    lp, trusted = score.values(_ids(0, 1, 2))
    assert lp.tolist() == [0.0, 0.25, 0.0]
    assert trusted.tolist() == [True, True, True]


def test_sgs_weight_peaks_at_target(make_outcomes: MakeOutcomes) -> None:
    # rates 0.2, 0.5, 0.8 from 10 outcomes each; manipulation defaults t=0.5, kappa=1.
    score = SuccessBand(capacity=3, device="cpu", form="sgs")
    score.record(_rates(make_outcomes, [2, 5, 8], total=10))
    lp, _ = score.values(_ids(0, 1, 2))
    # l(0.5) = ln(0.5001) + 1e-4 = -0.693247...; l(0.2) = 0.5 ln(0.2001*0.8001) + 1e-4.
    expected = [_sgs_logit(p, t=0.5, kappa=1.0, eps=1e-4) for p in (0.2, 0.5, 0.8)]
    assert lp.tolist() == pytest.approx(expected, abs=1e-5)
    assert lp[1] > lp[0]
    assert lp[1] > lp[2]


def test_sgs_peak_moves_with_target(make_outcomes: MakeOutcomes) -> None:
    score = SuccessBand(capacity=3, device="cpu", form="sgs", target=0.8, kappa=5.0)
    score.record(_rates(make_outcomes, [2, 5, 8], total=10))
    lp, _ = score.values(_ids(0, 1, 2))
    assert int(torch.argmax(lp)) == 2


def test_sgs_probs_are_softmax_of_logits_over_temperature(
    make_outcomes: MakeOutcomes,
) -> None:
    score = SuccessBand(capacity=3, device="cpu", form="sgs", temperature=2.0)
    score.record(_rates(make_outcomes, [2, 5, 8], total=10))
    probs = score.probs(_ids(0, 1, 2))
    logits = [_sgs_logit(p, t=0.5, kappa=1.0, eps=1e-4) / 2.0 for p in (0.2, 0.5, 0.8)]
    z = sum(math.exp(x) for x in logits)
    assert probs.tolist() == pytest.approx([math.exp(x) / z for x in logits], abs=1e-5)
    assert probs.sum().item() == pytest.approx(1.0)


def test_sgs_never_assigns_zero_probability_to_extreme_rates(
    make_outcomes: MakeOutcomes,
) -> None:
    score = SuccessBand(capacity=3, device="cpu", form="sgs")
    score.record(_rates(make_outcomes, [0, 5, 10], total=10))
    probs = score.probs(_ids(0, 1, 2))
    assert bool((probs > 0).all())


def test_untrusted_ics_get_median(make_outcomes: MakeOutcomes) -> None:
    score = SuccessBand(capacity=5, device="cpu", form="pq", min_count=3)
    # Rates 0, 0.25, 0.5 from 4 outcomes -> lp 0, 0.1875, 0.25 (median 0.1875).
    rows = _rates(make_outcomes, [0, 1, 2], total=4)
    extra = make_outcomes([(3, True, 0.0)])  # IC 3: 1 outcome < min_count; IC 4 unseen
    score.record(Outcomes.cat([rows, extra]))
    lp, trusted = score.values(_ids(0, 1, 2, 3, 4))
    assert trusted.tolist() == [True, True, True, False, False]
    assert lp.tolist() == [0.0, 0.1875, 0.25, 0.1875, 0.1875]


def test_no_trusted_ics_gives_zero_lp_and_all_untrusted() -> None:
    score = SuccessBand(capacity=3, device="cpu", form="pq", min_count=2)
    lp, trusted = score.values(_ids(0, 1, 2))
    assert lp.tolist() == [0.0, 0.0, 0.0]
    assert trusted.tolist() == [False, False, False]


def test_forget_resets_history(make_outcomes: MakeOutcomes) -> None:
    score = SuccessBand(capacity=2, device="cpu", form="pq", min_count=1)
    score.record(_rates(make_outcomes, [2, 2], total=4))
    score.forget(_ids(0))
    lp, trusted = score.values(_ids(0, 1))
    assert trusted.tolist() == [False, True]
    assert lp.tolist() == [0.25, 0.25]  # IC 0 untrusted -> median of {0.25}


def test_pq_probs_fall_back_to_rank_proportional(make_outcomes: MakeOutcomes) -> None:
    score = SuccessBand(capacity=3, device="cpu", form="pq")
    score.record(_rates(make_outcomes, [0, 2, 1], total=4))  # lp 0, 0.25, 0.1875
    # Ranks (1 = highest lp): IC1 -> 1, IC2 -> 2, IC0 -> 3; p ~ 1/rank, norm 11/6.
    probs = score.probs(_ids(0, 1, 2))
    assert probs.tolist() == pytest.approx([(1 / 3) * 6 / 11, 6 / 11, (1 / 2) * 6 / 11])


def test_registry_constructs_success_band() -> None:
    score = SCORES.construct(
        "success_band", capacity=4, device="cpu", history=50, target=0.3
    )
    assert isinstance(score, SuccessBand)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"form": "nope"}, "form"),
        ({"target": 1.5}, "target"),
        ({"history": 0}, "history"),
        ({"min_count": 0}, "min_count"),
        ({"temperature": 0.0}, "temperature"),
        ({"eps": 0.0}, "eps"),
        ({"kappa": -1.0}, "kappa"),
    ],
)
def test_rejects_invalid_settings(settings: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SuccessBand(capacity=2, device="cpu", **settings)  # type: ignore[arg-type]  # deliberate bad input

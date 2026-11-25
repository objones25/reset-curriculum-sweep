from collections.abc import Callable

import pytest
import torch

from resetsweep.scores import SCORES
from resetsweep.scores.controls import Inverted, Random
from resetsweep.types import Outcomes

MakeOutcomes = Callable[..., Outcomes]


def _ids(*ids: int) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int64)


def test_random_score_fixed_until_forget(make_outcomes: MakeOutcomes) -> None:
    score = Random(capacity=8, device="cpu", seed=3)
    ids = torch.arange(8)
    first, trusted = score.values(ids)
    score.record(
        make_outcomes([(0, True, 1.0), (5, False, 0.0)])
    )  # outcomes do not matter
    again, _ = score.values(ids)
    assert torch.equal(first, again)
    assert trusted.tolist() == [True] * 8
    assert bool(((first >= 0) & (first < 1)).all())

    score.forget(_ids(2, 5))
    after, _ = score.values(ids)
    kept = [0, 1, 3, 4, 6, 7]
    assert torch.equal(after[kept], first[kept])
    assert bool((after[[2, 5]] != first[[2, 5]]).all())


def test_random_score_is_reproducible_per_seed() -> None:
    ids = torch.arange(6)
    a, _ = Random(capacity=6, device="cpu", seed=1).values(ids)
    b, _ = Random(capacity=6, device="cpu", seed=1).values(ids)
    c, _ = Random(capacity=6, device="cpu", seed=2).values(ids)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_inverted_reverses_ranking_of_inner(make_outcomes: MakeOutcomes) -> None:
    kwargs = {"capacity": 3, "device": "cpu"}
    inner = SCORES.construct("success_band", form="pq", **kwargs)
    inverted = SCORES.construct(
        "inverted", inner="success_band", inner_kw={"form": "pq"}, **kwargs
    )
    assert isinstance(inverted, Inverted)
    # Rates 0, 0.5, 0.25 from 4 outcomes each -> inner lp 0, 0.25, 0.1875.
    rows = [(ic, k < s, 0.0) for ic, s in enumerate([0, 2, 1]) for k in range(4)]
    for score in (inner, inverted):
        score.record(make_outcomes(rows))  # type: ignore[attr-defined]  # object from registry
    ids = _ids(0, 1, 2)
    inner_lp, _ = inner.values(ids)  # type: ignore[attr-defined]  # object from registry
    inv_lp, trusted = inverted.values(ids)
    # max(lp) - lp = 0.25 - [0, 0.25, 0.1875]
    assert inv_lp.tolist() == pytest.approx([0.25, 0.0, 0.0625])
    assert (
        torch.argsort(inv_lp).tolist()
        == torch.argsort(inner_lp, descending=True).tolist()
    )
    assert trusted.tolist() == [True, True, True]


def test_inverted_forget_reaches_inner(make_outcomes: MakeOutcomes) -> None:
    inverted = Inverted(
        capacity=2, device="cpu", inner="success_band", inner_kw={"form": "pq"}
    )
    inverted.record(make_outcomes([(0, True, 1.0), (1, True, 1.0)]))
    inverted.forget(_ids(0))
    _, trusted = inverted.values(_ids(0, 1))
    assert trusted.tolist() == [False, True]


def test_inverted_rejects_unknown_inner() -> None:
    with pytest.raises(KeyError, match="unknown score"):
        Inverted(capacity=2, device="cpu", inner="does_not_exist")


def test_rank_probs_favour_highest_lp_first() -> None:
    score = Random(capacity=4, device="cpu", seed=0)
    ids = torch.arange(4)
    lp, _ = score.values(ids)
    probs = score.probs(ids)
    # PLR (2010.03934 L204-214, beta = 0.1 at L299): p_i ~ (1/rank_i)^(1/beta) = rank^-10.
    # Z = 1 + 2^-10 + 3^-10 + 4^-10 = 1 + 9.765625e-4 + 1.6935e-5 + 9.537e-7 = 1.00099445.
    # Rank 1 -> 1/Z = 0.99900662; rank 4 -> 9.537e-7/Z = 9.527e-7.
    # (The earlier expectation 12/25 and 3/25 encoded beta = 1, i.e. no temperature.)
    assert probs.sum().item() == pytest.approx(1.0)
    best = int(torch.argmax(lp))
    assert probs[best].item() == pytest.approx(0.99900662, abs=1e-7)
    worst = int(torch.argmin(lp))
    assert probs[worst].item() == pytest.approx(9.527e-7, abs=1e-9)


def test_rank_probs_beta_one_is_plain_inverse_rank() -> None:
    score = Random(capacity=4, device="cpu", seed=0, beta=1.0)
    ids = torch.arange(4)
    lp, _ = score.values(ids)
    # beta = 1: p ~ 1/rank, normaliser 1 + 1/2 + 1/3 + 1/4 = 25/12.
    assert score.probs(ids)[int(torch.argmax(lp))].item() == pytest.approx(12 / 25)


def test_rejects_non_positive_beta() -> None:
    with pytest.raises(ValueError, match="beta"):
        Random(capacity=2, device="cpu", beta=0.0)


def test_random_rejects_ids_outside_capacity() -> None:
    score = Random(capacity=3, device="cpu")
    for bad in (_ids(3), _ids(-1)):
        with pytest.raises(ValueError, match="ic_ids"):
            score.values(bad)
        with pytest.raises(ValueError, match="ic_ids"):
            score.forget(bad)


def test_inverted_with_untrusted_ics(make_outcomes: MakeOutcomes) -> None:
    inverted = Inverted(
        capacity=4,
        device="cpu",
        inner="success_band",
        inner_kw={"form": "pq", "min_count": 3},
    )
    # ICs 0..2 get 4 outcomes at rates 0, 0.25, 0.5 -> inner lp 0, 0.1875, 0.25; IC 3 is unseen,
    # so it gets the median of the trusted (0.1875) before reversal.
    rows = [(ic, k < s, 0.0) for ic, s in enumerate([0, 1, 2]) for k in range(4)]
    inverted.record(make_outcomes(rows))
    lp, trusted = inverted.values(_ids(0, 1, 2, 3))
    # max(lp) - lp = 0.25 - [0, 0.1875, 0.25, 0.1875]
    assert lp.tolist() == pytest.approx([0.25, 0.0625, 0.0, 0.0625])
    assert trusted.tolist() == [True, True, True, False]

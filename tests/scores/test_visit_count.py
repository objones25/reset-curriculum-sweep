from collections.abc import Callable

import pytest
import torch

from resetsweep.scores import SCORES
from resetsweep.scores.visit_count import VisitCount
from resetsweep.types import Outcomes

MakeOutcomes = Callable[..., Outcomes]


def _ids(*ids: int) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int64)


def test_visit_count_score_decreases_with_count() -> None:
    score = VisitCount(capacity=3, device="cpu")
    score.set_counts(_ids(0, 1, 2), _ids(0, 3, 8))
    lp, trusted = score.values(_ids(0, 1, 2))
    # 1/sqrt(0+1) = 1, 1/sqrt(3+1) = 0.5, 1/sqrt(8+1) = 1/3.
    assert lp.tolist() == pytest.approx([1.0, 0.5, 1.0 / 3.0])
    assert trusted.tolist() == [True, True, True]


def test_unset_counts_default_to_zero() -> None:
    score = VisitCount(capacity=2, device="cpu")
    lp, _ = score.values(_ids(0, 1))
    assert lp.tolist() == [1.0, 1.0]


def test_count_x_learnability_zero_before_reward_is_replaced_by_count(
    make_outcomes: MakeOutcomes,
) -> None:
    score = VisitCount(capacity=3, device="cpu", post_reward="count_x_learnability")
    score.set_counts(_ids(0, 1, 2), _ids(0, 3, 8))
    # 4 outcomes per IC: successes 2, 0, 4 -> p = 0.5, 0, 1 -> p(1-p) = 0.25, 0, 0.
    rows = [(ic, k < s, 0.0) for ic, s in enumerate([2, 0, 4]) for k in range(4)]
    score.record(make_outcomes(rows))

    before, _ = score.values(_ids(0, 1, 2))
    assert before.tolist() == pytest.approx([1.0, 0.5, 1.0 / 3.0])

    score.notify_first_reward()
    after, _ = score.values(_ids(0, 1, 2))
    # count term * (p(1-p) + eps), eps = 0.05: 1*0.30, 0.5*0.05, (1/3)*0.05.
    assert after.tolist() == pytest.approx([0.3, 0.025, 0.05 / 3.0])


def test_count_form_ignores_learnability_after_reward(
    make_outcomes: MakeOutcomes,
) -> None:
    score = VisitCount(capacity=1, device="cpu", post_reward="count")
    score.set_counts(_ids(0), _ids(3))
    score.record(make_outcomes([(0, True, 1.0), (0, False, 0.0)]))
    score.notify_first_reward()
    lp, _ = score.values(_ids(0))
    assert lp.tolist() == pytest.approx([0.5])


def test_forget_resets_count() -> None:
    score = VisitCount(capacity=2, device="cpu")
    score.set_counts(_ids(0, 1), _ids(3, 8))
    score.forget(_ids(1))
    lp, _ = score.values(_ids(0, 1))
    assert lp.tolist() == pytest.approx([0.5, 1.0])


def test_probs_rank_fewer_visits_first() -> None:
    score = VisitCount(capacity=3, device="cpu")
    score.set_counts(_ids(0, 1, 2), _ids(8, 0, 3))
    probs = score.probs(_ids(0, 1, 2))
    assert int(torch.argmax(probs)) == 1
    assert probs[2] > probs[0]
    assert float(probs.sum()) == pytest.approx(1.0)


def test_set_counts_rejects_bad_input() -> None:
    score = VisitCount(capacity=3, device="cpu")
    with pytest.raises(ValueError, match="ic_ids"):
        score.set_counts(_ids(3), _ids(1))
    with pytest.raises(ValueError, match="non-negative"):
        score.set_counts(_ids(0), _ids(-1))
    with pytest.raises(ValueError, match="same shape"):
        score.set_counts(_ids(0, 1), _ids(1))


@pytest.mark.parametrize(
    ("settings", "message"),
    [({"post_reward": "nope"}, "post_reward"), ({"eps": 0.0}, "eps")],
)
def test_rejects_invalid_settings(settings: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VisitCount(capacity=2, device="cpu", **settings)  # type: ignore[arg-type]  # deliberate bad input


def test_registry_constructs_visit_count() -> None:
    score = SCORES.construct("visit_count", capacity=4, device="cpu")
    assert isinstance(score, VisitCount)

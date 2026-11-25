import pytest
import torch

from resetsweep.scores import SCORES
from resetsweep.scores.steps_to_reward import StepsToReward


def _ids(*ids: int) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int64)


def test_steps_to_reward_prefers_fewer_steps() -> None:
    score = StepsToReward(capacity=4, device="cpu")
    score.set_steps(_ids(0, 1, 2), torch.tensor([5, 20, 10]))
    lp, trusted = score.values(_ids(0, 1, 2, 3))
    # lp = -steps for the three ICs with metadata; IC 3 has none, so it gets the median
    # of the trusted (sorted -20, -10, -5 -> -10).
    assert lp.tolist() == [-5.0, -20.0, -10.0, -10.0]
    assert trusted.tolist() == [True, True, True, False]
    assert int(torch.argmax(score.probs(_ids(0, 1, 2)))) == 0


def test_forget_drops_steps() -> None:
    score = StepsToReward(capacity=2, device="cpu")
    score.set_steps(_ids(0, 1), torch.tensor([5, 9]))
    score.forget(_ids(0))
    _, trusted = score.values(_ids(0, 1))
    assert trusted.tolist() == [False, True]


def test_set_steps_overwrites() -> None:
    score = StepsToReward(capacity=1, device="cpu")
    score.set_steps(_ids(0), torch.tensor([9]))
    score.set_steps(_ids(0), torch.tensor([4]))
    lp, _ = score.values(_ids(0))
    assert lp.tolist() == [-4.0]


def test_set_steps_rejects_bad_input() -> None:
    score = StepsToReward(capacity=2, device="cpu")
    with pytest.raises(ValueError, match="ic_ids"):
        score.set_steps(_ids(2), torch.tensor([1]))
    with pytest.raises(ValueError, match="non-negative"):
        score.set_steps(_ids(0), torch.tensor([-1]))
    with pytest.raises(ValueError, match="same shape"):
        score.set_steps(_ids(0, 1), torch.tensor([1]))


def test_registry_constructs_steps_to_reward() -> None:
    score = SCORES.construct("steps_to_reward", capacity=4, device="cpu")
    assert isinstance(score, StepsToReward)

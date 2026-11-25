import pytest
import torch

from resetsweep.scores.base import SCORES

_KWARGS: dict[str, dict[str, object]] = {
    "critic_loss": {"form": "abs_gae"},
    "critic_value": {"vmax": 1.0},
    "inverted": {"inner": "success_band"},
}


@pytest.mark.parametrize("name", SCORES.names())
def test_every_score_reports_its_capacity(name: str) -> None:
    torch.manual_seed(0)

    score = SCORES.construct(name, capacity=7, device="cpu", **_KWARGS.get(name, {}))

    assert score.capacity == 7  # pyright: ignore[reportAttributeAccessIssue] # registry returns object

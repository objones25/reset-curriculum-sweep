import pytest
import torch

from resetsweep.types import Box, StartStates


def _states(n: int, offset: float = 0.0) -> StartStates:
    return StartStates(
        qpos=torch.arange(n, dtype=torch.float32).unsqueeze(1) + offset,
        qvel=torch.zeros(n, 2),
        extra=torch.zeros(n, 0),
    )


def test_start_states_n_and_index() -> None:
    s = _states(4)
    sub = s.index(torch.tensor([3, 1]))
    assert s.n == 4
    assert sub.n == 2
    assert sub.qpos.flatten().tolist() == [3.0, 1.0]


def test_start_states_cat_concatenates_rows() -> None:
    out = StartStates.cat([_states(2), _states(3, offset=10.0)])
    assert out.n == 5
    assert out.qpos.flatten().tolist() == [0.0, 1.0, 10.0, 11.0, 12.0]


def test_start_states_rejects_mismatched_rows() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        StartStates(
            qpos=torch.zeros(2, 1), qvel=torch.zeros(3, 1), extra=torch.zeros(2, 0)
        )


def test_box_holds_bounds() -> None:
    box = Box(low=torch.zeros(3), high=torch.ones(3))
    assert box.high.sum().item() == 3.0

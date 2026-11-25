from collections.abc import Callable

import pytest
import torch

from resetsweep.scores.base import ICTable
from resetsweep.types import Outcomes

MakeOutcomes = Callable[..., Outcomes]


def _ids(*ids: int) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int64)


def test_counts_and_success_rate_per_ic(make_outcomes: MakeOutcomes) -> None:
    table = ICTable(capacity=4, device="cpu", history=10)
    # IC 0: 1 success of 4 -> 0.25.  IC 1: 2 of 2 -> 1.0.  IC 2: unseen.
    table.record(
        make_outcomes(
            [
                (0, True, 1.0),
                (1, True, 1.0),
                (0, False, 0.0),
                (0, False, 0.0),
                (1, True, 1.0),
                (0, False, 0.0),
            ]
        )
    )
    ids = _ids(0, 1, 2)
    assert table.count(ids).tolist() == [4, 2, 0]
    assert table.success_rate(ids).tolist() == [0.25, 1.0, 0.0]
    assert table.ever_succeeded(ids).tolist() == [True, True, False]


def test_ignores_task_start_rows_and_partial_episodes(
    make_outcomes: MakeOutcomes,
) -> None:
    table = ICTable(capacity=3, device="cpu", history=10)
    table.record(
        make_outcomes(
            [(-1, True, 1.0), (0, True, 1.0), (1, True, 1.0)],
            full=[True, False, True],
        )
    )
    # Row 0 has ic_id -1, row 1 has full=False: only IC 1 is recorded.
    assert table.count(_ids(0, 1, 2)).tolist() == [0, 1, 0]


def test_circular_buffer_keeps_only_last_history_outcomes(
    make_outcomes: MakeOutcomes,
) -> None:
    table = ICTable(capacity=2, device="cpu", history=4)
    # 2 failures then 4 successes on IC 0: the window holds the last 4 -> rate 1.0.
    rows = [(0, False, 0.0)] * 2 + [(0, True, 1.0)] * 4
    table.record(make_outcomes(rows))
    assert table.count(_ids(0)).tolist() == [4]
    assert table.success_rate(_ids(0)).tolist() == [1.0]


def test_circular_buffer_wraps_across_separate_calls(
    make_outcomes: MakeOutcomes,
) -> None:
    table = ICTable(capacity=1, device="cpu", history=3)
    table.record(make_outcomes([(0, True, 1.0), (0, True, 1.0), (0, True, 1.0)]))
    table.record(make_outcomes([(0, False, 0.0)]))
    # Window is now [T, T, F] after overwriting the oldest slot.
    assert table.success_rate(_ids(0)).tolist() == pytest.approx([2 / 3])


def test_recent_returns_are_chronological_with_validity_mask(
    make_outcomes: MakeOutcomes,
) -> None:
    table = ICTable(capacity=2, device="cpu", history=4)
    table.record(make_outcomes([(0, False, float(r)) for r in (1, 2, 3, 4, 5, 6)]))
    table.record(make_outcomes([(1, False, 7.0)]))
    ret, _, valid = table.recent(_ids(0, 1), 3)
    # IC 0 has seen 1..6 with history 4; the last 3 in time order are 4, 5, 6.
    assert ret[0].tolist() == [4.0, 5.0, 6.0]
    # IC 1 has seen one outcome: only the final position is valid.
    assert valid[1].tolist() == [False, False, True]
    assert ret[1, 2].item() == 7.0


def test_forget_resets_history(make_outcomes: MakeOutcomes) -> None:
    table = ICTable(capacity=2, device="cpu", history=4)
    table.record(make_outcomes([(0, True, 1.0), (0, True, 1.0), (1, True, 1.0)]))
    table.forget(_ids(0))
    ids = _ids(0, 1)
    assert table.count(ids).tolist() == [0, 1]
    assert table.success_rate(ids).tolist() == [0.0, 1.0]
    assert table.ever_succeeded(ids).tolist() == [False, True]


def test_record_rejects_ic_id_beyond_capacity(make_outcomes: MakeOutcomes) -> None:
    table = ICTable(capacity=2, device="cpu", history=4)
    with pytest.raises(ValueError, match="capacity"):
        table.record(make_outcomes([(2, True, 1.0)]))


@pytest.mark.parametrize(
    "kwargs", [{"capacity": 0, "history": 4}, {"capacity": 2, "history": 0}]
)
def test_rejects_non_positive_sizes(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        ICTable(device="cpu", **kwargs)


def test_empty_outcomes_are_a_no_op(make_outcomes: MakeOutcomes) -> None:
    table = ICTable(capacity=2, device="cpu", history=4)
    table.record(make_outcomes([]))
    assert table.count(_ids(0, 1)).tolist() == [0, 0]

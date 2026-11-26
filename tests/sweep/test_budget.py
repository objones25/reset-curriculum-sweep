"""Budget guard (spec section 8, Review Focus 2: the ledger survives a pod restart).

Prices are chosen so that 1 USD = 1 s of pod time: gpu 3600 USD/h, volume 0, so hand-computed
values stay readable.
"""

import json
from pathlib import Path

import pytest

from resetsweep.sweep.budget import (
    HEARTBEAT_S,
    LEDGER_GAP_S,
    MARGIN_USD,
    BudgetGuard,
    LedgerError,
    volume_usd_per_h,
)


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _guard(
    tmp_path: Path, clock: FakeClock, *, cap: float = 100.0, margin: float = 5.0
) -> BudgetGuard:
    return BudgetGuard(
        cap_usd=cap,
        margin_usd=margin,
        gpu_usd_per_h=3600.0,
        volume_usd_per_h=0.0,
        ledger_path=tmp_path / "ledger.jsonl",
        clock=clock,
    )


def _ledger(tmp_path: Path) -> list[dict[str, float]]:
    """Ledger rows without their wall-clock `utc` stamp (tested on its own below)."""
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    return [{k: v for k, v in row.items() if k != "utc"} for row in rows]


def test_volume_cost_per_hour() -> None:
    # 20 GB x 0.07 USD/GB-month / 730 h/month = 0.0019178... USD/h
    assert volume_usd_per_h(20) == pytest.approx(20 * 0.07 / 730)


def test_volume_cost_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="gb must be >= 0"):
        volume_usd_per_h(-1)


def test_spent_counts_wall_clock_times_gpu_and_volume_price(tmp_path: Path) -> None:
    clock = FakeClock()
    guard = BudgetGuard(
        cap_usd=100.0,
        margin_usd=MARGIN_USD,
        gpu_usd_per_h=2.0,
        volume_usd_per_h=0.5,
        ledger_path=tmp_path / "ledger.jsonl",
        clock=clock,
    )
    clock.t += 1800.0  # half an hour at 2.5 USD/h

    assert guard.spent() == pytest.approx(1.25)


def test_spent_includes_prior_ledger_entries(tmp_path: Path) -> None:
    (tmp_path / "ledger.jsonl").write_text(
        json.dumps({"t": 0.0, "spent": 1.0})
        + "\n"
        + json.dumps({"t": 30.0, "spent": 7.5})
        + "\n"
    )
    clock = FakeClock()
    guard = _guard(tmp_path, clock)
    clock.t += 10.0

    # last ledger value 7.5, plus the restart allowance (time lost since the last heartbeat,
    # at most LEDGER_GAP_S = 60 s = 60 USD here), plus 10 s of this session.
    assert guard.spent() == pytest.approx(7.5 + LEDGER_GAP_S + 10.0)


def test_fresh_ledger_starts_at_zero(tmp_path: Path) -> None:
    guard = _guard(tmp_path, FakeClock())

    assert guard.spent() == 0.0


def test_restart_writes_session_start_line_with_prior_spend(tmp_path: Path) -> None:
    (tmp_path / "ledger.jsonl").write_text(json.dumps({"t": 5.0, "spent": 3.0}) + "\n")

    _guard(tmp_path, FakeClock())

    assert _ledger(tmp_path)[-1] == {
        "t": 0.0,
        "spent": pytest.approx(3.0 + LEDGER_GAP_S),
    }


def test_ledger_torn_last_line_is_truncated(tmp_path: Path) -> None:
    good = json.dumps({"t": 5.0, "spent": 3.0}) + "\n"
    (tmp_path / "ledger.jsonl").write_text(good + '{"t": 9.0, "sp')

    guard = _guard(tmp_path, FakeClock())

    assert guard.spent() == pytest.approx(3.0 + LEDGER_GAP_S)
    assert [row["spent"] for row in _ledger(tmp_path)] == [3.0, pytest.approx(63.0)]


def test_ledger_bad_line_before_the_last_raises(tmp_path: Path) -> None:
    good = json.dumps({"t": 5.0, "spent": 3.0}) + "\n"
    (tmp_path / "ledger.jsonl").write_text("not json\n" + good)

    with pytest.raises(LedgerError, match="line 1"):
        _guard(tmp_path, FakeClock())


def test_ledger_rejects_decreasing_spend(tmp_path: Path) -> None:
    rows = [{"t": 0.0, "spent": 5.0}, {"t": 1.0, "spent": 4.0}]
    (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    with pytest.raises(LedgerError, match="decreases"):
        _guard(tmp_path, FakeClock())


def test_can_start_true_when_projected_within_cap_minus_margin(tmp_path: Path) -> None:
    clock = FakeClock()
    guard = _guard(tmp_path, clock, cap=100.0, margin=5.0)
    clock.t += 50.0  # spent 50

    # 50 + 30 + 15 = 95 = cap - margin exactly: allowed.
    assert guard.can_start(30.0, 15.0) is True


def test_can_start_false_when_projected_exceeds_cap_minus_margin(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    guard = _guard(tmp_path, clock, cap=100.0, margin=5.0)
    clock.t += 50.0  # spent 50

    # 50 + 30 + 15.5 = 95.5 > 95
    assert guard.can_start(30.0, 15.5) is False


def test_can_start_counts_prior_sessions(tmp_path: Path) -> None:
    (tmp_path / "ledger.jsonl").write_text(json.dumps({"t": 0.0, "spent": 30.0}) + "\n")
    guard = _guard(tmp_path, FakeClock(), cap=100.0, margin=5.0)

    # prior 30 + gap 60 = 90; 90 + 6 = 96 > 95
    assert guard.can_start(6.0, 0.0) is False


def test_can_start_rejects_negative_estimates(tmp_path: Path) -> None:
    guard = _guard(tmp_path, FakeClock())

    with pytest.raises(ValueError, match="in_flight_s must be finite and >= 0"):
        guard.can_start(1.0, -1.0)


def test_heartbeat_appends_after_interval(tmp_path: Path) -> None:
    clock = FakeClock()
    guard = _guard(tmp_path, clock)
    clock.t += HEARTBEAT_S - 1.0
    guard.heartbeat()  # too early: nothing written
    clock.t += 1.0
    guard.heartbeat()

    assert _ledger(tmp_path) == [
        {"t": 0.0, "spent": 0.0},
        {"t": HEARTBEAT_S, "spent": pytest.approx(HEARTBEAT_S)},
    ]


def test_heartbeat_interval_is_at_most_60_s() -> None:
    # The contract: a ledger line at least every 60 s while the loop polls (poll <= 5 s).
    assert HEARTBEAT_S <= 55.0
    assert LEDGER_GAP_S == 60.0


def test_flush_always_writes(tmp_path: Path) -> None:
    clock = FakeClock()
    guard = _guard(tmp_path, clock)
    clock.t += 1.0
    guard.flush()

    assert _ledger(tmp_path)[-1] == {"t": 1.0, "spent": pytest.approx(1.0)}


def test_new_session_resumes_from_previous_session_ledger(tmp_path: Path) -> None:
    clock = FakeClock()
    first = _guard(tmp_path, clock)
    clock.t += 20.0
    first.flush()  # 20 USD in session 1

    second = _guard(tmp_path, FakeClock(t=0.0))

    assert second.spent() == pytest.approx(20.0 + LEDGER_GAP_S)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cap_usd", 0.0),
        ("margin_usd", -1.0),
        ("gpu_usd_per_h", 0.0),
        ("volume_usd_per_h", -0.1),
    ],
)
def test_constructor_rejects_bad_prices(
    tmp_path: Path, field: str, value: float
) -> None:
    kw: dict[str, float] = {
        "cap_usd": 100.0,
        "margin_usd": 5.0,
        "gpu_usd_per_h": 1.0,
        "volume_usd_per_h": 0.0,
    }
    kw[field] = value

    with pytest.raises(ValueError, match=field):
        BudgetGuard(**kw, ledger_path=tmp_path / "ledger.jsonl", clock=FakeClock())


def test_every_ledger_line_carries_a_utc_timestamp(tmp_path: Path) -> None:
    clock = FakeClock()
    guard = BudgetGuard(
        cap_usd=100.0,
        margin_usd=5.0,
        gpu_usd_per_h=3600.0,
        volume_usd_per_h=0.0,
        ledger_path=tmp_path / "ledger.jsonl",
        clock=clock,
        utc_now=lambda: "2026-11-25T12:00:00+00:00",
    )
    clock.t += HEARTBEAT_S
    guard.heartbeat()

    rows = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
    ]
    assert [row["utc"] for row in rows] == ["2026-11-25T12:00:00+00:00"] * 2


def test_heartbeat_interval_is_15_s() -> None:
    assert HEARTBEAT_S == 15.0


# --- work deadline (final-review I2) -------------------------------------------------------------


def test_stop_at_lowers_the_limit_to_the_spend_at_that_time(tmp_path: Path) -> None:
    # 1 USD per second; the stop is 30 s after the guard starts -> limit 30 USD, below 95.
    clock = FakeClock(1000.0)
    guard = BudgetGuard(
        cap_usd=100.0,
        margin_usd=5.0,
        gpu_usd_per_h=3600.0,
        volume_usd_per_h=0.0,
        ledger_path=tmp_path / "ledger.jsonl",
        clock=clock,
        stop_at=1030.0,
    )

    assert guard.limit_usd() == pytest.approx(30.0)
    assert not guard.deadline_reached()
    assert guard.can_start(29.0, 0.0)
    assert not guard.can_start(31.0, 0.0)
    clock.t = 1030.0
    assert guard.deadline_reached()
    assert guard.remaining() == pytest.approx(0.0)


def test_stop_at_after_the_cap_leaves_the_cap_limit(tmp_path: Path) -> None:
    clock = FakeClock(1000.0)
    guard = BudgetGuard(
        cap_usd=100.0,
        margin_usd=5.0,
        gpu_usd_per_h=3600.0,
        volume_usd_per_h=0.0,
        ledger_path=tmp_path / "ledger.jsonl",
        clock=clock,
        stop_at=10_000.0,
    )

    assert guard.limit_usd() == pytest.approx(95.0)
    assert not guard.deadline_reached()

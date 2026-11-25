"""Spend floor in `ledger.settle` (final-review I3), hand-computed.

A launch's spend is at least ``gpu_usd_per_h * min(hours since created_at, max_hours)`` for
its pod, plus the volume's storage since its first use, until the billing API reports at
least that much. Billing that lags, or reads 0 for a pod that deleted itself, cannot lower it.
"""

from datetime import UTC, datetime, timedelta

import pytest

from resetsweep.launch.ledger import Launch, settle
from resetsweep.sweep.budget import NETWORK_VOLUME_GB, volume_usd_per_h

CREATED = datetime(2026, 11, 24, 9, 15, tzinfo=UTC)
VOLUME_USD_PER_H = volume_usd_per_h(NETWORK_VOLUME_GB)  # 20 * 0.07 / 730


class FakeBilling:
    def __init__(self, pod: float = 0.0, volume: float = 0.0) -> None:
        self.pod = pod
        self.volume = volume

    def pod_spend_usd(self, pod_id: str, *, since: str) -> float:
        del pod_id, since
        return self.pod

    def volume_spend_usd(self, volume_id: str, *, since: str) -> float:
        del volume_id, since
        return self.volume

    def find_pod_by_name(self, name: str) -> dict[str, object] | None:
        del name
        return None


def _launch(**kw: object) -> Launch:
    fields: dict[str, object] = {
        "pod_id": "pod-1",
        "volume_id": "vol-1",
        "gpu_id": "NVIDIA GeForce RTX 4090",
        "gpu_usd_per_h": 0.9,
        "created_at": CREATED.isoformat(),
        "spent_usd": 0.0,
        "smoke": False,
        "max_hours": 10.0,
    }
    return Launch(**(fields | kw))  # type: ignore[arg-type]


def _settle(billing: FakeBilling, launches: list[Launch], hours: float) -> list[Launch]:
    return settle(billing, launches, now=CREATED + timedelta(hours=hours))  # type: ignore[arg-type]


def test_billing_lag_is_floored_at_price_times_hours() -> None:
    # 2 h after create, billing shows 0.10: floor = 0.9 * 2 + volume 2 h.
    (settled,) = _settle(FakeBilling(pod=0.1), [_launch()], hours=2.0)

    assert settled.spent_usd == pytest.approx(0.9 * 2 + VOLUME_USD_PER_H * 2)


def test_billing_zero_for_a_deleted_pod_keeps_the_floor_capped_at_max_hours() -> None:
    # 30 h later the pod is long gone and billing reads 0: the pod part stops at max_hours 10.
    (settled,) = _settle(FakeBilling(pod=0.0, volume=0.0), [_launch()], hours=30.0)

    assert settled.spent_usd == pytest.approx(0.9 * 10 + VOLUME_USD_PER_H * 30)


def test_billing_above_the_floor_wins() -> None:
    (settled,) = _settle(FakeBilling(pod=12.0, volume=0.5), [_launch()], hours=2.0)

    assert settled.spent_usd == pytest.approx(12.5)


def test_floor_never_lowers_an_earlier_settled_spend() -> None:
    (settled,) = _settle(FakeBilling(), [_launch(spent_usd=50.0)], hours=1.0)

    assert settled.spent_usd == pytest.approx(50.0)


def test_orphaned_volume_is_floored_by_storage_only() -> None:
    (settled,) = _settle(
        FakeBilling(), [_launch(pod_id=None, gpu_id=None, gpu_usd_per_h=None, max_hours=None)], hours=5.0
    )

    assert settled.spent_usd == pytest.approx(VOLUME_USD_PER_H * 5)


def test_reused_volume_storage_is_floored_once() -> None:
    first = _launch()
    second = _launch(pod_id="pod-2", created_at=(CREATED + timedelta(hours=1)).isoformat())

    a, b = _settle(FakeBilling(), [first, second], hours=3.0)

    assert a.spent_usd == pytest.approx(0.9 * 3 + VOLUME_USD_PER_H * 3)
    assert b.spent_usd == pytest.approx(0.9 * 2)


def test_ledger_entry_without_max_hours_still_loads() -> None:
    assert _launch(max_hours=None).max_hours is None

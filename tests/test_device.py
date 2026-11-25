import pytest
import warp as wp

from resetsweep.device import pick_device


def test_pick_device_returns_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wp, "is_cuda_available", lambda: True)

    assert pick_device() == "cuda:0"


def test_pick_device_returns_cpu_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wp, "is_cuda_available", lambda: False)

    assert pick_device() == "cpu"

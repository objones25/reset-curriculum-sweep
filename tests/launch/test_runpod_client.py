"""RunPod REST v2 client (Review Focus 3): bounded retries, typed errors, the key never leaks.

Endpoints and fields: docs.runpod.io/api-reference-v2 (catalog, billing, pods, network volumes),
read 2026-11-25.
"""

import random
from collections.abc import Callable

import httpx
import pytest
from runpod_fakes import DC, FAKE_RUNPOD_KEY, PRICES, FakeRunPod

from resetsweep.launch.runpod_client import DataCenter, RunPodClient, RunPodError


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _scripted(
    responses: list[httpx.Response], seen: list[httpx.Request]
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)

    return handler


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    sleep: RecordingSleep,
    *,
    max_retries: int = 4,
) -> RunPodClient:
    return RunPodClient(
        FAKE_RUNPOD_KEY,
        transport=httpx.MockTransport(handler),
        max_retries=max_retries,
        sleep=sleep,
        rng=random.Random(0),
    )


def test_retries_on_429_honouring_retry_after_then_succeeds() -> None:
    seen: list[httpx.Request] = []
    sleep = RecordingSleep()
    responses = [
        httpx.Response(429, headers={"Retry-After": "7"}, json={"detail": "slow down"}),
        httpx.Response(429, headers={"Retry-After": "3"}, json={"detail": "slow down"}),
        httpx.Response(200, json={"balanceAmount": 42.5, "billingMode": "PREPAID"}),
    ]
    client = _client(_scripted(responses, seen), sleep)

    balance = client.balance_usd()

    assert balance == pytest.approx(42.5)
    assert sleep.calls == pytest.approx([7.0, 3.0])
    assert len(seen) == 3


def test_gives_up_after_max_retries_with_runpod_error_without_key_in_message() -> None:
    seen: list[httpx.Request] = []
    sleep = RecordingSleep()
    echo = {"detail": f"upstream failed for token {FAKE_RUNPOD_KEY}"}
    responses = [httpx.Response(503, json=echo) for _ in range(3)]
    client = _client(_scripted(responses, seen), sleep, max_retries=2)

    with pytest.raises(RunPodError, match="503") as info:
        client.datacenters()

    assert info.value.status == 503
    assert FAKE_RUNPOD_KEY not in str(info.value)
    assert FAKE_RUNPOD_KEY not in info.value.detail
    assert "***" in info.value.detail
    assert len(seen) == 3
    assert len(sleep.calls) == 2


def test_backoff_without_retry_after_uses_full_jitter_within_exponential_cap() -> None:
    seen: list[httpx.Request] = []
    sleep = RecordingSleep()
    responses = [httpx.Response(502) for _ in range(4)] + [
        httpx.Response(200, json={"dataCenters": []})
    ]
    client = _client(_scripted(responses, seen), sleep, max_retries=4)

    client.datacenters()

    caps = [1.0, 2.0, 4.0, 8.0]
    assert len(sleep.calls) == 4
    assert all(0.0 <= s <= cap for s, cap in zip(sleep.calls, caps, strict=True))
    assert len(set(sleep.calls)) == 4


def test_post_is_not_replayed_on_5xx_so_no_second_volume() -> None:
    seen: list[httpx.Request] = []
    sleep = RecordingSleep()
    responses = [
        httpx.Response(503, json={"detail": "maybe created"}),
        httpx.Response(201, json={"id": "v2"}),
    ]
    client = _client(_scripted(responses, seen), sleep)

    with pytest.raises(RunPodError, match="503"):
        client.create_volume("vol", 20, DC)

    assert len(seen) == 1
    assert sleep.calls == []


def test_post_is_retried_on_429() -> None:
    seen: list[httpx.Request] = []
    sleep = RecordingSleep()
    responses = [
        httpx.Response(429, headers={"Retry-After": "1"}),
        httpx.Response(
            201, json={"id": "vol-9", "name": "vol", "size": 20, "dataCenter": DC}
        ),
    ]
    client = _client(_scripted(responses, seen), sleep)

    volume_id = client.create_volume("vol", 20, DC)

    assert volume_id == "vol-9"
    assert len(seen) == 2


def test_create_volume_sends_name_size_and_datacenter() -> None:
    seen: list[httpx.Request] = []
    responses = [httpx.Response(201, json={"id": "vol-9"})]
    client = _client(_scripted(responses, seen), RecordingSleep())

    client.create_volume("resetsweep-vol", 20, DC)

    assert seen[0].url == "https://api.runpod.io/v2/network-volumes"
    assert (
        seen[0].read() == b'{"name":"resetsweep-vol","size":20,"dataCenter":"EU-RO-1"}'
    )


def test_retry_after_is_capped() -> None:
    seen: list[httpx.Request] = []
    sleep = RecordingSleep()
    responses = [
        httpx.Response(429, headers={"Retry-After": "86400"}),
        httpx.Response(200, json={"dataCenters": []}),
    ]
    client = _client(_scripted(responses, seen), sleep)

    client.datacenters()

    assert sleep.calls == pytest.approx([60.0])


def test_transport_error_on_get_is_retried_then_raises_runpod_error() -> None:
    sleep = RecordingSleep()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler, sleep, max_retries=3)

    with pytest.raises(RunPodError, match="ConnectError") as info:
        client.balance_usd()

    assert info.value.status == 0
    assert len(calls) == 4


def test_read_timeout_on_post_is_not_replayed() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    client = _client(handler, RecordingSleep())

    with pytest.raises(RunPodError, match="ReadTimeout"):
        client.create_pod({"name": "x"})

    assert len(calls) == 1


def test_delete_pod_204_is_success() -> None:
    seen: list[httpx.Request] = []
    client = _client(_scripted([httpx.Response(204)], seen), RecordingSleep())

    result = client.delete_pod("pod-7")

    assert result is None
    assert seen[0].method == "DELETE"
    assert seen[0].url == "https://api.runpod.io/v2/pods/pod-7"


def test_delete_pod_404_raises() -> None:
    seen: list[httpx.Request] = []
    client = _client(
        _scripted([httpx.Response(404, json={"detail": "gone"})], seen),
        RecordingSleep(),
    )

    with pytest.raises(RunPodError, match="404"):
        client.delete_pod("pod-7")


def test_requests_carry_bearer_token(fake_runpod: FakeRunPod) -> None:
    client = RunPodClient(FAKE_RUNPOD_KEY, transport=fake_runpod.transport())

    client.balance_usd()

    assert (
        fake_runpod.requests[0].headers["Authorization"] == f"Bearer {FAKE_RUNPOD_KEY}"
    )


def test_runpod_client_repr_hides_key() -> None:
    client = RunPodClient(
        FAKE_RUNPOD_KEY, transport=httpx.MockTransport(lambda r: httpx.Response(200))
    )

    assert FAKE_RUNPOD_KEY not in repr(client)


def test_gpu_types_queries_secure_pod_availability_and_parses(
    fake_runpod: FakeRunPod,
) -> None:
    client = RunPodClient(FAKE_RUNPOD_KEY, transport=fake_runpod.transport())

    gpus = client.gpu_types()

    params = dict(fake_runpod.requests[0].url.params)
    assert params == {"include": "AVAILABILITY", "product": "POD", "cloud": "SECURE"}
    by_id = {g.id: g for g in gpus}
    assert by_id["NVIDIA A40"].secure_usd_per_h == pytest.approx(PRICES["NVIDIA A40"])
    assert by_id["NVIDIA A40"].available_datacenters == ("US-NOVOL-1", DC)


def test_gpu_types_skips_datacenters_with_no_stock(fake_runpod: FakeRunPod) -> None:
    fake_runpod.gpus = [
        {
            "id": "NVIDIA A40",
            "price": {"secure": 0.4, "community": 0.2},
            "dataCenters": [
                {"id": "A", "availability": "NONE"},
                {"id": "B", "availability": "LOW"},
            ],
        }
    ]
    client = RunPodClient(FAKE_RUNPOD_KEY, transport=fake_runpod.transport())

    assert client.gpu_types()[0].available_datacenters == ("B",)


def test_datacenters_parses_network_volume_types_and_gpu_stock(
    fake_runpod: FakeRunPod,
) -> None:
    client = RunPodClient(FAKE_RUNPOD_KEY, transport=fake_runpod.transport())

    dcs = client.datacenters()

    assert dict(fake_runpod.requests[0].url.params) == {"include": "GPU_AVAILABILITY"}
    assert dcs[1] == DataCenter(
        id=DC, network_volume_types=("STANDARD",), available_gpus=tuple(PRICES)
    )


def test_malformed_catalog_raises_runpod_error() -> None:
    seen: list[httpx.Request] = []
    client = _client(
        _scripted([httpx.Response(200, json={"gpus": [{"id": "x"}]})], seen),
        RecordingSleep(),
    )

    with pytest.raises(RunPodError, match="price"):
        client.gpu_types()


def test_pod_spend_reads_billing_totals_for_one_pod(fake_runpod: FakeRunPod) -> None:
    fake_runpod.pod_spend_usd["pod-old"] = 12.25
    client = RunPodClient(FAKE_RUNPOD_KEY, transport=fake_runpod.transport())

    spend = client.pod_spend_usd("pod-old", since="2026-11-24T10:00:00Z")

    params = dict(fake_runpod.requests[0].url.params)
    assert spend == pytest.approx(12.25)
    assert params == {
        "podId": "pod-old",
        "startTime": "2026-11-24T10:00:00Z",
        "bucketSize": "hour",
    }


def test_volume_spend_reads_billing_totals_for_one_volume(
    fake_runpod: FakeRunPod,
) -> None:
    fake_runpod.volume_spend_usd["vol-old"] = 0.5
    client = RunPodClient(FAKE_RUNPOD_KEY, transport=fake_runpod.transport())

    spend = client.volume_spend_usd("vol-old", since="2026-11-24T10:00:00Z")

    assert spend == pytest.approx(0.5)
    assert fake_runpod.requests[0].url.path == "/v2/billing/network-volumes"
    assert fake_runpod.requests[0].url.params["networkVolumeId"] == "vol-old"

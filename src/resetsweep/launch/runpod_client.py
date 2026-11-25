"""A small synchronous client for the RunPod REST v2 API (spec section 9, Review Focus 3).

Endpoints, from docs.runpod.io/api-reference-v2 (read 2026-11-25 via the RunPod docs search):
- `GET  /catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE`: `gpus[]` with `id`,
  `price.secure` (USD/h for one GPU) and `dataCenters[]` {`id`, `availability`
  NONE|LOW|MEDIUM|HIGH}.
- `GET  /catalog/datacenters?include=GPU_AVAILABILITY`: `dataCenters[]` with `id`,
  `networkVolumeTypes` (STANDARD|HIGH_PERFORMANCE) and `gpuAvailability[]`.
- `POST /network-volumes` {name, size, dataCenter} -> 201 with `id`.
- `GET  /network-volumes/{id}` -> `id`, `size` (GB), `dataCenter`, `type`.
- `POST /pods` -> 201 with `id` and `cost` (USD/h). 400 = no capacity or a rule violation,
  422 = validation failed.
- `DELETE /pods/{id}` -> 204.
- `GET  /billing/balance` -> `balanceAmount` (USD), `billingMode`.
- `GET  /billing/pods?podId=&startTime=&bucketSize=hour` and
  `GET  /billing/network-volumes?networkVolumeId=&startTime=&bucketSize=hour` ->
  `metadata.totals.totalAmount` (USD over the window, the in-progress bucket included).

Retries (bounded by `max_retries`): exponential backoff with full jitter, base 1 s, each delay
capped at 60 s; a `Retry-After` header wins (also capped at 60 s). GET and DELETE retry on
408/425/429/5xx and on any transport error. POST is not idempotent here (a replayed create can
make a second volume or pod), so it retries only on 408/425/429, which mean the request was not
acted on, and on connect failures, where nothing reached the server. Every failure raises
`RunPodError`; the API key is replaced by `***` in any detail before it is stored.

httpx 0.28.1 (installed source): `httpx.Client(base_url=, transport=, timeout=, headers=)`,
`httpx.MockTransport(handler)` for tests, `httpx.TransportError` as the base of connect, read
and write failures.
"""

import email.utils
import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import httpx

BASE_URL: Final = "https://api.runpod.io/v2"
BACKOFF_BASE_S: Final = 1.0
BACKOFF_MAX_S: Final = 60.0
MAX_RETRIES_LIMIT: Final = 10
DETAIL_MAX_CHARS: Final = 500
IDEMPOTENT_RETRY_STATUSES: Final = frozenset({408, 425, 429, 500, 502, 503, 504})
POST_RETRY_STATUSES: Final = frozenset({408, 425, 429})


class RunPodError(Exception):
    """A RunPod API call failed. `status` is the HTTP status, or 0 for a transport failure."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"RunPod API {status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class GpuType:
    id: str
    secure_usd_per_h: float
    available_datacenters: tuple[
        str, ...
    ]  # data centers with stock (availability != NONE)


@dataclass(frozen=True)
class DataCenter:
    id: str
    network_volume_types: tuple[str, ...]
    available_gpus: tuple[str, ...]  # GPU type ids with stock (availability != NONE)


@dataclass(frozen=True)
class NetworkVolume:
    id: str
    datacenter: str
    size_gb: int
    type: str


class RunPodClient:
    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        base_url: str = BASE_URL,
        timeout_s: float = 30.0,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key is blank")
        if not 0 <= max_retries <= MAX_RETRIES_LIMIT:
            raise ValueError(
                f"max_retries must be in [0, {MAX_RETRIES_LIMIT}], got {max_retries}"
            )
        if not timeout_s > 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        self._api_key = api_key
        self._max_retries = max_retries
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._http = httpx.Client(
            base_url=base_url,
            transport=transport,
            timeout=httpx.Timeout(timeout_s),
            headers={"Authorization": f"Bearer {api_key}"},
            follow_redirects=False,
        )

    def __repr__(self) -> str:
        return f"RunPodClient(base_url={str(self._http.base_url)!r}, api_key=***)"

    def close(self) -> None:
        self._http.close()

    # -- catalog -------------------------------------------------------------------------

    def gpu_types(self) -> list[GpuType]:
        params = {"include": "AVAILABILITY", "product": "POD", "cloud": "SECURE"}
        data = self._json("GET", "/catalog/gpus", params=params)
        return [_parse_gpu(g) for g in _list(data, "gpus")]

    def datacenters(self) -> list[DataCenter]:
        data = self._json(
            "GET", "/catalog/datacenters", params={"include": "GPU_AVAILABILITY"}
        )
        return [_parse_datacenter(d) for d in _list(data, "dataCenters")]

    # -- resources -----------------------------------------------------------------------

    def create_volume(self, name: str, size_gb: int, datacenter: str) -> str:
        body = {"name": name, "size": size_gb, "dataCenter": datacenter}
        return _str(self._json("POST", "/network-volumes", body=body), "id")

    def get_volume(self, volume_id: str) -> NetworkVolume:
        data = self._json("GET", f"/network-volumes/{volume_id}")
        size = data.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RunPodError(
                200, "malformed response: volume `size` is not a positive integer"
            )
        volume = NetworkVolume(
            id=_str(data, "id"),
            datacenter=_str(data, "dataCenter"),
            size_gb=size,
            type=_str(data, "type"),
        )
        if volume.id != volume_id:
            raise RunPodError(200, f"asked for volume {volume_id}, got {volume.id}")
        return volume

    def create_pod(self, body: dict[str, object]) -> dict[str, object]:
        pod = self._json("POST", "/pods", body=body)
        _str(pod, "id")
        return pod

    def delete_pod(self, pod_id: str) -> None:
        response = self._request("DELETE", f"/pods/{pod_id}")
        if response.status_code != 204:
            raise RunPodError(
                response.status_code, f"DELETE /pods/{pod_id}: expected 204"
            )

    # -- billing -------------------------------------------------------------------------

    def balance_usd(self) -> float:
        return _number(self._json("GET", "/billing/balance"), "balanceAmount")

    def pod_spend_usd(self, pod_id: str, *, since: str) -> float:
        params = {"podId": pod_id, "startTime": since, "bucketSize": "hour"}
        return _billing_total(self._json("GET", "/billing/pods", params=params))

    def volume_spend_usd(self, volume_id: str, *, since: str) -> float:
        params = {
            "networkVolumeId": volume_id,
            "startTime": since,
            "bucketSize": "hour",
        }
        return _billing_total(
            self._json("GET", "/billing/network-volumes", params=params)
        )

    # -- transport -----------------------------------------------------------------------

    def _json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        response = self._request(method, path, params=params, body=body)
        try:
            data = response.json()
        except ValueError as e:
            raise RunPodError(
                response.status_code, f"{method} {path}: body is not JSON"
            ) from e
        if not isinstance(data, dict):
            raise RunPodError(
                response.status_code, f"{method} {path}: body is not a JSON object"
            )
        return data

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: Mapping[str, object] | None = None,
    ) -> httpx.Response:
        idempotent = method in ("GET", "DELETE")
        retry_statuses = (
            IDEMPOTENT_RETRY_STATUSES if idempotent else POST_RETRY_STATUSES
        )
        for attempt in range(self._max_retries + 1):
            last = attempt == self._max_retries
            try:
                response = self._http.request(method, path, params=params, json=body)
            except httpx.TransportError as e:
                replay_safe = idempotent or isinstance(
                    e, httpx.ConnectError | httpx.ConnectTimeout
                )
                if replay_safe and not last:
                    self._sleep(self._backoff_s(attempt, None))
                    continue
                raise RunPodError(0, f"{method} {path}: {type(e).__name__}") from e
            if response.status_code in retry_statuses and not last:
                self._sleep(
                    self._backoff_s(attempt, response.headers.get("Retry-After"))
                )
                continue
            if response.status_code >= 400:
                raise RunPodError(
                    response.status_code, f"{method} {path}: {self._detail(response)}"
                )
            return response
        raise AssertionError("unreachable: the last attempt always returns or raises")

    def _backoff_s(self, attempt: int, retry_after: str | None) -> float:
        hinted = _parse_retry_after(retry_after)
        if hinted is not None:
            return min(hinted, BACKOFF_MAX_S)
        delay = self._rng.uniform(0.0, min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2**attempt))
        if not 0.0 <= delay <= BACKOFF_MAX_S:
            raise AssertionError(f"backoff {delay} outside [0, {BACKOFF_MAX_S}]")
        return delay

    def _detail(self, response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("detail"), str):
            text = data["detail"]
        else:
            text = response.text
        return self._redact(text)[:DETAIL_MAX_CHARS] or "(empty body)"

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***")


def _parse_retry_after(value: str | None) -> float | None:
    """Seconds from a Retry-After header (delta-seconds or HTTP-date), or None if unusable."""
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _list(data: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    items = data.get(key)
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise RunPodError(200, f"malformed response: `{key}` is not a list of objects")
    return items


def _str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise RunPodError(
            200, f"malformed response: `{key}` is missing or not a string"
        )
    return value


def _number(data: Mapping[str, Any], key: str) -> float:
    value = data.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise RunPodError(
            200, f"malformed response: `{key}` is missing or not a finite number"
        )
    return float(value)


def _available_ids(entries: object) -> tuple[str, ...]:
    if entries is None:
        return ()
    if not isinstance(entries, list):
        raise RunPodError(200, "malformed response: availability is not a list")
    return tuple(
        _str(e, "id")
        for e in entries
        if isinstance(e, dict) and e.get("availability") != "NONE"
    )


def _parse_gpu(data: Mapping[str, Any]) -> GpuType:
    price = data.get("price")
    if not isinstance(price, dict):
        raise RunPodError(
            200, f"malformed response: GPU {data.get('id')!r} has no `price`"
        )
    usd_per_h = _number(price, "secure")
    if usd_per_h < 0:
        raise RunPodError(
            200, f"malformed response: negative price for {data.get('id')!r}"
        )
    return GpuType(
        id=_str(data, "id"),
        secure_usd_per_h=usd_per_h,
        available_datacenters=_available_ids(data.get("dataCenters")),
    )


def _parse_datacenter(data: Mapping[str, Any]) -> DataCenter:
    types = data.get("networkVolumeTypes")
    if not isinstance(types, list) or not all(isinstance(t, str) for t in types):
        raise RunPodError(
            200,
            f"malformed response: data center {data.get('id')!r} has no `networkVolumeTypes`",
        )
    return DataCenter(
        id=_str(data, "id"),
        network_volume_types=tuple(types),
        available_gpus=_available_ids(data.get("gpuAvailability")),
    )


def _billing_total(data: Mapping[str, Any]) -> float:
    metadata = data.get("metadata")
    totals = metadata.get("totals") if isinstance(metadata, dict) else None
    if not isinstance(totals, dict):
        raise RunPodError(200, "malformed billing response: no `metadata.totals`")
    total = _number(totals, "totalAmount")
    if total < 0:
        raise RunPodError(200, f"malformed billing response: negative total {total}")
    return total

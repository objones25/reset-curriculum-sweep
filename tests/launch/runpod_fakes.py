"""A fake RunPod REST v2 API behind `httpx.MockTransport`, plus fake secrets (fixtures live in
conftest.py; this module is imported by name because pytest runs in prepend import mode).

Response shapes follow docs.runpod.io/api-reference-v2 (read 2026-11-25): `gpus[]` with
`price.secure` and `dataCenters[].availability`, `dataCenters[]` with `networkVolumeTypes` and
`gpuAvailability[]`, `balanceAmount`, and billing `metadata.totals.totalAmount`. Every number is
ours. The fake never talks to the network.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

FAKE_RUNPOD_KEY = "rk-FAKE-0123456789abcdef"
FAKE_WANDB_KEY = "wk-FAKE-fedcba9876543210"
DC = "EU-RO-1"
DC_NO_VOLUMES = "US-NOVOL-1"
PRICES = {
    "NVIDIA GeForce RTX 4090": 0.69,
    "NVIDIA A40": 0.40,
    "NVIDIA L40S": 0.86,
    "NVIDIA RTX A6000": 0.49,
}


def env_line(name: str, value: str) -> str:
    # Built from parts so the secret-pattern scan (tests/test_no_secrets.py) sees no assignment.
    return name + "=" + value + "\n"


def _gpu(gpu_id: str, dcs: list[str]) -> dict[str, object]:
    return {
        "id": gpu_id,
        "name": gpu_id,
        "secure": True,
        "community": True,
        "price": {"secure": PRICES[gpu_id], "community": PRICES[gpu_id] / 2},
        "availability": "HIGH",
        "dataCenters": [{"id": dc, "name": dc, "availability": "HIGH"} for dc in dcs],
    }


def _datacenter(dc: str, volume_types: list[str]) -> dict[str, object]:
    return {
        "id": dc,
        "name": dc,
        "networkVolumeTypes": volume_types,
        "gpuAvailability": [
            {"id": g, "name": g, "availability": "HIGH"} for g in PRICES
        ],
    }


@dataclass
class FakeRunPod:
    """Routes the endpoints the launcher uses; records every request."""

    gpus: list[dict[str, object]] = field(
        default_factory=lambda: [_gpu(g, [DC_NO_VOLUMES, DC]) for g in PRICES]
    )
    datacenters: list[dict[str, object]] = field(
        default_factory=lambda: [
            _datacenter(DC_NO_VOLUMES, []),
            _datacenter(DC, ["STANDARD"]),
        ]
    )
    balance_usd: float = 500.0
    pod_spend_usd: dict[str, float] = field(default_factory=dict)
    volume_spend_usd: dict[str, float] = field(default_factory=dict)
    pod_create_statuses: list[int] = field(default_factory=list)
    volume_create_status: int = 201
    volumes: dict[str, dict[str, object]] = field(default_factory=dict)
    listed_pods: list[dict[str, object]] = field(default_factory=list)  # GET /v2/pods
    created_despite_error: bool = (
        False  # a failed create still made the pod (shows in list)
    )
    echo_body_on_error: bool = (
        False  # error detail repeats the request body, as some APIs do
    )
    requests: list[httpx.Request] = field(default_factory=list)
    _pods_created: int = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        route = (request.method, request.url.path)
        routes: dict[tuple[str, str], Callable[[httpx.Request], httpx.Response]] = {
            ("GET", "/v2/catalog/gpus"): lambda _: httpx.Response(
                200, json={"gpus": self.gpus}
            ),
            ("GET", "/v2/catalog/datacenters"): lambda _: httpx.Response(
                200, json={"dataCenters": self.datacenters}
            ),
            ("GET", "/v2/billing/balance"): lambda _: httpx.Response(
                200, json={"balanceAmount": self.balance_usd, "billingMode": "PREPAID"}
            ),
            ("GET", "/v2/billing/pods"): lambda r: self._billing(
                r, "podId", self.pod_spend_usd
            ),
            ("GET", "/v2/billing/network-volumes"): lambda r: self._billing(
                r, "networkVolumeId", self.volume_spend_usd
            ),
            ("POST", "/v2/network-volumes"): self._create_volume,
            ("POST", "/v2/pods"): self._create_pod,
            ("GET", "/v2/pods"): lambda _: httpx.Response(
                200,
                json={
                    "pods": self.listed_pods,
                    "pagination": {"nextCursor": None, "hasNextPage": False},
                },
            ),
        }
        if request.method == "GET" and request.url.path.startswith(
            "/v2/network-volumes/"
        ):
            volume = self.volumes.get(request.url.path.rsplit("/", 1)[1])
            if volume is None:
                return httpx.Response(404, json={"detail": "volume not found"})
            return httpx.Response(200, json=volume)
        if route not in routes:
            return httpx.Response(404, json={"detail": f"no fake route {route}"})
        return routes[route](request)

    def _billing(
        self, request: httpx.Request, id_param: str, spend: dict[str, float]
    ) -> httpx.Response:
        resource_id = request.url.params[id_param]
        total = spend.get(resource_id, 0.0)
        return httpx.Response(
            200,
            json={
                "records": [],
                "metadata": {"totals": {"totalAmount": total}, "recordCount": 0},
            },
        )

    def _create_volume(self, request: httpx.Request) -> httpx.Response:
        if self.volume_create_status != 201:
            return httpx.Response(self.volume_create_status, json={"detail": "fake"})
        body = json.loads(request.content)
        return httpx.Response(201, json={"id": "vol-1", **body, "type": "STANDARD"})

    def _create_pod(self, request: httpx.Request) -> httpx.Response:
        status = self.pod_create_statuses.pop(0) if self.pod_create_statuses else 201
        body = json.loads(request.content)
        if status != 201:
            if self.created_despite_error:
                self._pods_created += 1
                self.listed_pods.append(
                    {
                        "id": f"pod-{self._pods_created}",
                        "name": body["name"],
                        "cost": 0.69,
                    }
                )
            detail = f"fake {status}" + (
                f": {request.content.decode()}" if self.echo_body_on_error else ""
            )
            return httpx.Response(status, json={"detail": detail})
        self._pods_created += 1
        [price] = [
            g["price"]["secure"] for g in self.gpus if g["id"] == body["gpu"]["id"]
        ]  # type: ignore[index]
        return httpx.Response(
            201,
            json={
                "id": f"pod-{self._pods_created}",
                "cost": price,
                "status": "PROVISIONING",
                "createdAt": "2026-11-25T10:00:00Z",
            },
        )

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def sent(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]

    def bodies(self, method: str, path: str) -> list[dict[str, object]]:
        return [json.loads(r.content) for r in self.sent(method, path)]

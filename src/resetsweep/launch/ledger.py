"""The launcher-side, account-wide spend ledger (`~/.resetsweep/ledger.json`, spec section 8).

The in-pod ledger misses pod time spent outside `sweep-run` (uv sync, the GPU tests, the time
after the report), and the network volume keeps billing after the pod is gone. So every launch is
recorded here (pod id or None for an orphaned volume, volume id, hourly price, creation time),
and before each new launch `settle` replaces every entry's `spent_usd` with what the RunPod
billing API reports for that pod and that volume since the hour it was created.

File format: `{"version": 1, "launches": [{pod_id, volume_id, gpu_id, gpu_usd_per_h,
created_at (ISO 8601 with offset), spent_usd, smoke, name, unresolved, max_hours, repo_sha},
...]}`. `name`, `unresolved`, `max_hours` and `repo_sha` default to null/false so older files
still load. `gpu_usd_per_h` is the pod's price per hour (the price the floor uses).

Spend floor (final-review I3): billing runs behind, and reads 0 for a pod that has deleted
itself, so `settle` never lets a launch's spend fall below
``gpu_usd_per_h * min(hours since created_at, max_hours)`` for its pod (when both are known)
plus ``volume_usd_per_h(volume GB) * hours since created_at`` for the launch that first used its
volume. Billing replaces the floor only once it reports at least as much. The floor is
deliberately pessimistic: a pod that ended early still counts up to `max_hours` of pod time.

An `unresolved` launch is one whose pod create failed ambiguously (a 5xx or a read timeout
after sending):
a pod with that `name` may exist and bill. `settle` looks it up by name each time; until it is
found, the launcher refuses new launches (a pod that already ran and deleted itself can no
longer be found by name, so that case needs a hand edit after checking the console). A file that does not parse raises
`LedgerError`: the launcher then refuses to launch rather than guess the spend.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from resetsweep.launch.runpod_client import RunPodClient
from resetsweep.sweep.budget import NETWORK_VOLUME_GB, volume_usd_per_h

LEDGER_VERSION: Final = 1
LEDGER_PATH_DEFAULT: Final = Path.home() / ".resetsweep" / "ledger.json"
LAUNCHES_MAX: Final = 1000  # far above any real use; a bigger file is corrupt, not busy


class LedgerError(Exception):
    """The ledger file exists but cannot be trusted."""


@dataclass(frozen=True)
class Launch:
    pod_id: str | None  # None: the volume was created but no pod was
    volume_id: str
    gpu_id: str | None
    gpu_usd_per_h: float | None
    created_at: str  # ISO 8601 with a UTC offset
    spent_usd: float
    smoke: bool
    name: str | None = None
    unresolved: bool = False
    max_hours: float | None = None  # the SWEEP_MAX_HOURS the pod got
    repo_sha: str | None = None  # the commit the pod checked out

    def created(self) -> datetime:
        return datetime.fromisoformat(self.created_at)


def load(path: Path) -> list[Launch]:
    """Launches recorded in `path`; [] if the file does not exist yet."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise LedgerError(f"ledger {path} is not valid JSON ({e})") from e
    if not isinstance(data, dict) or data.get("version") != LEDGER_VERSION:
        raise LedgerError(
            f"ledger {path} is not a version-{LEDGER_VERSION} ledger object"
        )
    raw = data.get("launches")
    if not isinstance(raw, list) or len(raw) > LAUNCHES_MAX:
        raise LedgerError(
            f"ledger {path}: `launches` is not a list of at most {LAUNCHES_MAX}"
        )
    return [_parse(path, i, item) for i, item in enumerate(raw)]


def save(path: Path, launches: list[Launch]) -> None:
    """Write atomically (temp file + rename) so a crash never leaves half a ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {"version": LEDGER_VERSION, "launches": [asdict(x) for x in launches]}
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    if len(load(path)) != len(launches):
        raise AssertionError(f"ledger {path} did not read back what was written")


def _floor_usd(launch: Launch, *, now: datetime, first_on_volume: bool) -> float:
    """The least this launch can have cost by `now` (see the module docstring)."""
    hours = max(0.0, (now - launch.created()).total_seconds() / 3600.0)
    floor = 0.0
    pod_may_exist = launch.pod_id is not None or launch.unresolved
    if pod_may_exist and launch.gpu_usd_per_h is not None and launch.max_hours is not None:
        floor += launch.gpu_usd_per_h * min(hours, launch.max_hours)
    if first_on_volume:
        floor += volume_usd_per_h(NETWORK_VOLUME_GB) * hours
    return floor


def settle(client: RunPodClient, launches: list[Launch], *, now: datetime) -> list[Launch]:
    """Each launch with `spent_usd` set to its pod + volume spend from the billing API, but
    never below the spend floor at `now` (see the module docstring).

    A volume reused by later launches (`--volume-id`) is billed once, to the earliest launch
    that used it, from that launch's creation hour. Never lowered: billing runs every few
    minutes, so a fresh read can lag an earlier one."""
    if now.utcoffset() is None:
        raise ValueError("now must carry a UTC offset")
    first_use: dict[str, int] = {}
    for i, launch in enumerate(launches):
        j = first_use.get(launch.volume_id)
        if j is None or launch.created() < launches[j].created():
            first_use[launch.volume_id] = i
    settled = []
    for i, launch in enumerate(launches):
        if launch.unresolved:
            launch = _resolve(client, launch)
        since = _billing_start(launch.created())
        spent = 0.0
        if first_use[launch.volume_id] == i:
            spent += client.volume_spend_usd(launch.volume_id, since=since)
        if launch.pod_id is not None:
            spent += client.pod_spend_usd(launch.pod_id, since=since)
        floor = _floor_usd(launch, now=now, first_on_volume=first_use[launch.volume_id] == i)
        settled.append(replace(launch, spent_usd=max(spent, floor, launch.spent_usd)))
    if len(settled) != len(launches):
        raise AssertionError("settle must keep every launch")
    return settled


def _resolve(client: RunPodClient, launch: Launch) -> Launch:
    if launch.name is None:
        raise AssertionError("an unresolved launch always has a name (checked on load)")
    pod = client.find_pod_by_name(launch.name)
    if pod is None:
        return launch
    return replace(launch, pod_id=str(pod["id"]), unresolved=False)


def unresolved(launches: list[Launch]) -> list[Launch]:
    return [x for x in launches if x.unresolved]


def total_spent_usd(launches: list[Launch]) -> float:
    total = math.fsum(x.spent_usd for x in launches)
    if not (math.isfinite(total) and total >= 0):
        raise AssertionError(
            f"ledger total {total} is not a finite non-negative amount"
        )
    return total


def _billing_start(created: datetime) -> str:
    """The start of the UTC hour `created` falls in, RFC 3339 (billing snaps to hour buckets)."""
    if created.utcoffset() is None:
        raise AssertionError(
            "ledger timestamps always carry a UTC offset (checked on load)"
        )
    hour = created.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    return hour.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(path: Path, index: int, item: Any) -> Launch:
    where = f"ledger {path}, launch {index}"
    if not isinstance(item, dict):
        raise LedgerError(f"{where} is not an object")
    try:
        launch = Launch(**item)
    except TypeError as e:
        raise LedgerError(f"{where}: {e}") from e
    if not isinstance(launch.volume_id, str) or not launch.volume_id:
        raise LedgerError(f"{where}: volume_id must be a non-empty string")
    if launch.pod_id is not None and (
        not isinstance(launch.pod_id, str) or not launch.pod_id
    ):
        raise LedgerError(f"{where}: pod_id must be null or a non-empty string")
    spent = launch.spent_usd
    if isinstance(spent, bool) or not isinstance(spent, int | float) or not spent >= 0:
        raise LedgerError(f"{where}: spent_usd must be a non-negative number")
    try:
        created = launch.created()
    except (TypeError, ValueError) as e:
        raise LedgerError(f"{where}: created_at is not ISO 8601 ({e})") from e
    if created.utcoffset() is None:
        raise LedgerError(f"{where}: created_at has no UTC offset")
    if launch.name is not None and (
        not isinstance(launch.name, str) or not launch.name
    ):
        raise LedgerError(f"{where}: name must be null or a non-empty string")
    if not isinstance(launch.unresolved, bool):
        raise LedgerError(f"{where}: unresolved must be true or false")
    if launch.unresolved and (launch.name is None or launch.pod_id is not None):
        raise LedgerError(f"{where}: an unresolved launch needs a name and no pod_id")
    for field, value in (("gpu_usd_per_h", launch.gpu_usd_per_h), ("max_hours", launch.max_hours)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int | float) or not value >= 0
        ):
            raise LedgerError(f"{where}: {field} must be null or a non-negative number")
    if launch.repo_sha is not None and (
        not isinstance(launch.repo_sha, str) or not launch.repo_sha
    ):
        raise LedgerError(f"{where}: repo_sha must be null or a non-empty string")
    return launch

"""`sweep-launch`: start the sweep pod on RunPod within the account-wide budget (spec 8, 9).

Steps:
1. Load `.env` secrets (`load_secrets`); a missing or blank key exits 1 naming it, before any
   HTTP call (Review Focus 4). Then load the local ledger (a corrupt ledger exits 1, no HTTP).
2. Settle the ledger from the billing API: every earlier launch's pod and volume spend.
3. `remaining = cap - spent`; refuse (exit 3) unless `remaining - MARGIN_USD > 0` and the
   account balance is at least `remaining`. Ruling (coordinator, 2026-11-25): spec 8 says
   "below the cap", but its intent is that the balance cannot run out mid-sweep, which
   `balance >= remaining` (cap minus what is already spent) satisfies.
4. Pick a data center with STANDARD network volumes where `--gpu` (or else a fallback) has
   stock. With `--volume-id`, the data center is that volume's. Candidates there are tried in
   order: `--gpu` first, then the fallbacks by ascending live price; GPUs without stock are
   skipped. Automatic picks (the default GPU and the fallbacks) must cost at most
   `--max-usd-per-h` (default 1.20); a GPU named with `--gpu` ignores that ceiling. If nothing
   at or under the ceiling has stock, exit 2 listing the in-stock GPUs above it, their prices
   and the command that picks each.
5. `max_hours = (remaining - margin) / (gpu_usd_per_h + volume_usd_per_h)`, printed.
6. `--dry-run`: print the request bodies with secrets as "***" and stop (GETs only).
   Otherwise create one 20 GB STANDARD volume (or reuse `--volume-id`), then the pod. A
   pod-create 400 (no capacity) moves to the next candidate; another 4xx, or 400 on every
   candidate, exits 2, and a volume created by this launch is reported as orphaned with its
   delete command and recorded in the ledger (Review Focus 3). A 5xx or a lost connection may
   have created the pod: it is looked up by its unique name; if found the launch goes on, if
   not it is recorded as `unresolved`, "a pod may exist" is printed, and later launches are
   refused until a settle finds it. A second volume or pod is never created.
7. Record the launch in the ledger; print the pod id, the hourly cost, the W&B project and
   how to resume.

Pod cap: `SWEEP_CAP_USD` is `cap - spent` for a new volume. For a reused volume it is
`cap - (spend on other volumes) - SETUP_ALLOWANCE_USD x (earlier launches on it)`, because the
in-pod guard already subtracts that volume's own ledger (see `_pod_cap_usd`).

Resume: the pod deletes itself on any exit, SIGTERM included (budget safety over in-place
resume). The sweep state lives on the network volume, so `sweep-launch --volume-id <id>`
(with `--smoke` if the first launch had it) starts a new pod on the same volume, and
`sweep-run` picks up where it stopped.

Pod secrets: our RunPod key goes in as `SWEEP_RUNPOD_API_KEY`, never `RUNPOD_API_KEY`,
because RunPod injects its own pod-scoped key under that name
(docs.runpod.io/pods/templates/environment-variables). The pod command installs an EXIT trap
that deletes the pod before it clones, so a failed clone cannot leave a pod billing; the
bootstrap then replaces the shell (`exec`) and installs its own.

Smoke and full runs get different `SWEEP_STATE_DIR`s, because `sweep-run` refuses (exit 4) a
state dir made by the other mode.

Exit codes: 0 launched (or dry run done), 1 bad config, ledger, volume or RunPod API error,
2 no pod could be created (or no data center fits), 3 refused by the budget check; argparse
usage errors also exit 2.
"""

import argparse
import copy
import json
import math
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import httpx

from resetsweep.launch import ledger
from resetsweep.launch.env_file import MissingSecret, Secrets, load_secrets
from resetsweep.launch.runpod_client import (
    DataCenter,
    GpuType,
    RunPodClient,
    RunPodError,
)
from resetsweep.sweep.budget import (
    CAP_USD_DEFAULT,
    MARGIN_USD,
    NETWORK_VOLUME_GB,
    volume_usd_per_h,
)

IMAGE: Final = "runpod/pytorch:2.2.0-cu1290-torch291-ubuntu2404"
GPU_DEFAULT: Final = "NVIDIA GeForce RTX 4090"
# Automatic picks (the default GPU and every fallback) must cost at most this per hour; a GPU
# above it runs only when named with --gpu, so the user, not the launcher, accepts the price.
MAX_USD_PER_H_DEFAULT: Final = 1.20
# NVIDIA, on CUDA >= 13.2 hosts (mujoco-warp). Exact ids from GET /v2/catalog/gpus,
# 2026-11-25; listed by that day's Secure price, and re-sorted by the live price at launch.
GPU_FALLBACKS: Final = (
    "NVIDIA RTX A5000",
    "NVIDIA A40",
    "NVIDIA L4",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA A100-SXM4-80GB",
)
MIN_CUDA_VERSION: Final = "13.2"
CONTAINER_DISK_GB: Final = 40
VOLUME_MOUNT: Final = "/workspace"
VOLUME_TYPE: Final = "STANDARD"
STATE_DIR_FULL: Final = "/workspace/sweep"  # == sweep.main.STATE_DIR_DEFAULT (tested)
STATE_DIR_SMOKE: Final = "/workspace/sweep-smoke"
WANDB_PROJECT: Final = (
    "reset-curriculum-sweep"  # == sweep.main.PROJECT_DEFAULT (tested)
)
POD_KEY_ENV: Final = "SWEEP_RUNPOD_API_KEY"
REDACTED: Final = "***"
SECRET_ENV: Final = (POD_KEY_ENV, "WANDB_API_KEY")
API_V2: Final = "https://api.runpod.io/v2"
GIT_TIMEOUT_S: Final = 30
SETUP_ALLOWANCE_USD: Final = (
    2.0  # per earlier launch on a reused volume; see _pod_cap_usd
)
_SELF_DELETE: Final = (
    'self_delete() { if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${SWEEP_RUNPOD_API_KEY:-}" ]; '
    "then printf 'Authorization: Bearer %s\\n' \"$SWEEP_RUNPOD_API_KEY\" | curl -fsS -X DELETE "
    "--max-time 30 --retry 5 --retry-all-errors -H @- "
    '"https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" || true; fi; }; trap self_delete EXIT; '
)
# The bootstrap's EXIT trap leaves DONE-<pod id> on the volume; a restarted container of the
# same pod sees it and deletes the pod instead of redoing paid work. A new pod (resume) has a new
# id, so it is not blocked.
_DONE_CHECK: Final = 'if [ -e "${SWEEP_STATE_DIR:-/workspace/sweep}/DONE-${RUNPOD_POD_ID:-}" ]; then exit 0; fi; '
# Fetch and checkout are separate commands (not `&&`) so `set -e` stops on a failed fetch, and
# `test -f` runs before `exec` because a failed `exec` exits without running the EXIT trap.
CLONE_ONE_LINER: Final = (
    "set -euo pipefail; "
    + _SELF_DELETE
    + _DONE_CHECK
    + "cd /workspace; if [ -d repo ]; then git -C repo fetch --depth 1 origin "
    '"$REPO_REF"; git -C repo checkout FETCH_HEAD; else git clone --depth 1 --branch '
    '"$REPO_REF" "$REPO_URL" repo; fi; test -f repo/scripts/pod_bootstrap.sh; '
    "exec bash repo/scripts/pod_bootstrap.sh"
)


class GitError(Exception):
    """A git default (`--ref`, `--repo-url`) could not be read."""


class LaunchRefused(Exception):
    """The budget or balance check says no; nothing was created."""


class VolumeUnusable(Exception):
    """`--volume-id` names a volume this launcher cannot use or price."""


@dataclass(frozen=True)
class Plan:
    datacenter: str
    candidates: tuple[GpuType, ...]  # in try order, all with stock in `datacenter`


@dataclass(frozen=True)
class _Volume:
    id: str | None  # None until created
    datacenter: str | None  # None: pick one
    size_gb: int
    reused: bool


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    env_path: Path = Path(".env"),
    ledger_path: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] | None = None,
) -> int:
    """`sweep-launch`. See the module docstring for the steps and exit codes."""
    args = _parser().parse_args(argv)
    clock = now or (lambda: datetime.now(UTC))
    try:
        secrets = load_secrets(env_path)
        launches = ledger.load(ledger_path or ledger.LEDGER_PATH_DEFAULT)
        repo_url = args.repo_url or _git("remote", "get-url", "origin")
        ref = args.ref or _git("rev-parse", "--abbrev-ref", "HEAD")
    except (MissingSecret, FileNotFoundError, ledger.LedgerError, GitError) as e:
        _err(f"sweep-launch: {e}")
        return 1
    client = RunPodClient(
        secrets.runpod_api_key,
        transport=transport,
        sleep=sleep,
        redact=(secrets.wandb_api_key,),
    )
    try:
        return _launch(
            client,
            args,
            argv=list(sys.argv[1:] if argv is None else argv),
            secrets=secrets,
            launches=launches,
            ledger_path=ledger_path or ledger.LEDGER_PATH_DEFAULT,
            repo_url=repo_url,
            ref=ref,
            created=clock(),
        )
    except LaunchRefused as e:
        _err(f"sweep-launch: refused: {e}")
        return 3
    except (RunPodError, VolumeUnusable) as e:
        _err(f"sweep-launch: {e}")
        return 1
    finally:
        client.close()


def _launch(
    client: RunPodClient,
    args: argparse.Namespace,
    *,
    argv: list[str],
    secrets: Secrets,
    launches: list[ledger.Launch],
    ledger_path: Path,
    repo_url: str,
    ref: str,
    created: datetime,
) -> int:
    launches = ledger.settle(client, launches)
    pending = ledger.unresolved(launches)
    if pending:
        names = ", ".join(str(x.name) for x in pending)
        raise LaunchRefused(
            f"an earlier pod create failed ambiguously and no pod named {names} was found; a pod "
            "may exist and bill: check the RunPod console, then set its ledger entry's pod_id "
            f"(or unresolved: false if none exists) in {ledger_path}"
        )
    spent = ledger.total_spent_usd(launches)
    remaining = args.cap - spent
    print(
        f"ledger: {spent:.2f} USD spent by {len(launches)} earlier launch(es); cap {args.cap:.2f}"
    )
    if remaining - MARGIN_USD <= 0:
        raise LaunchRefused(
            f"{spent:.2f} USD already spent of the {args.cap:.2f} USD cap; "
            f"{remaining:.2f} left is not above the {MARGIN_USD:.2f} USD margin"
        )
    balance = client.balance_usd()
    if balance < remaining:
        raise LaunchRefused(
            f"account balance {balance:.2f} USD is below the {remaining:.2f} USD this launch may spend"
        )
    volume = _volume(client, args.volume_id)
    pod_cap = _pod_cap_usd(args.cap, launches, volume.id)
    if pod_cap - MARGIN_USD <= 0:
        raise LaunchRefused(
            f"the pod's cap would be {pod_cap:.2f} USD, not above the margin"
        )
    gpu_id = args.gpu or GPU_DEFAULT
    gpus, datacenters = client.gpu_types(), client.datacenters()
    if args.gpu is not None and args.gpu not in {g.id for g in gpus}:
        _err(
            f"sweep-launch: --gpu {args.gpu!r} is not a GPU type id in the RunPod catalog"
        )
        return 2
    plan = _plan(
        gpu_id,
        gpus,
        datacenters,
        only_dc=volume.datacenter,
        max_usd_per_h=args.max_usd_per_h,
        explicit=args.gpu is not None,
    )
    if plan is None:
        where = (
            f"data center {volume.datacenter} (volume {volume.id})"
            if volume.reused
            else f"no data center with {VOLUME_TYPE} network volumes"
        )
        _err(
            f"sweep-launch: {where} has stock of {', '.join(_candidate_ids(gpu_id))} "
            f"at or under {args.max_usd_per_h:.2f} USD/h"
        )
        _print_options_over_ceiling(
            gpu_id,
            gpus,
            datacenters,
            only_dc=volume.datacenter,
            ceiling=args.max_usd_per_h,
            argv=argv,
        )
        return 2
    mode = "smoke" if args.smoke else "sweep"
    name = f"resetsweep-{mode}-{created:%Y%m%d-%H%M%S}"
    pod_args = _PodArgs(
        name=name,
        secrets=secrets,
        repo_url=repo_url,
        ref=ref,
        remaining=remaining,
        pod_cap_usd=pod_cap,
        smoke=args.smoke,
        volume_gb=volume.size_gb,
    )
    first = plan.candidates[0]
    print(
        f"data center {plan.datacenter}; GPUs to try: {', '.join(g.id for g in plan.candidates)}"
    )
    print(
        f"remaining={remaining:.2f} USD pod_cap={pod_cap:.2f} USD balance={balance:.2f} USD"
    )
    _print_cost(first, volume.size_gb)
    _print_max_hours(remaining, first, volume.size_gb)
    if args.dry_run:
        if volume.reused:
            print(f"dry run: reusing network volume {volume.id} in {plan.datacenter}")
        else:
            volume_body = {
                "name": name,
                "size": volume.size_gb,
                "dataCenter": plan.datacenter,
                "type": VOLUME_TYPE,
            }
            print("dry run: POST /v2/network-volumes")
            print(json.dumps(volume_body, indent=2))
        print("dry run: POST /v2/pods (first GPU)")
        body = _pod_body(
            first, plan.datacenter, volume.id or "<new-volume-id>", pod_args
        )
        print(_redacted_json(body, secrets))
        print("dry run: nothing created")
        return 0

    if volume.id is None:
        try:
            volume_id = client.create_volume(name, volume.size_gb, plan.datacenter)
        except RunPodError as e:
            # A 5xx or a lost connection after sending does not say whether the volume exists.
            _err(
                f"sweep-launch: volume create failed ({e}); a volume named {name} may still exist: "
                f"check GET {API_V2}/network-volumes before launching again"
            )
            return 1
        print(f"created network volume {volume_id} in {plan.datacenter}")
    else:
        volume_id = volume.id
        print(f"reusing network volume {volume_id} in {plan.datacenter}")
    record = ledger.Launch(
        pod_id=None,
        volume_id=volume_id,
        gpu_id=None,
        gpu_usd_per_h=None,
        created_at=created.isoformat(),
        spent_usd=0.0,
        smoke=args.smoke,
        name=name,
    )
    outcome, pod, gpu = _create_pod(client, plan, volume_id, pod_args)
    if outcome == "ambiguous":
        pending_launch = replace(
            record, gpu_id=gpu.id, gpu_usd_per_h=gpu.secure_usd_per_h, unresolved=True
        )
        saved = _save_ledger(
            ledger_path,
            [*launches, pending_launch],
            note=f"pod name {name}, volume {volume_id}, unresolved",
        )
        _err(
            f"sweep-launch: a pod may exist — check the console for a pod named {name} "
            f"(volume {volume_id}); later launches are refused until it is resolved"
        )
        return 2 if saved else 1
    if pod is None:
        if volume.reused:
            _err(
                f"sweep-launch: no pod was created; your volume {volume_id} is unchanged"
            )
            return 2
        saved = _save_ledger(
            ledger_path, [*launches, record], note=f"orphaned volume {volume_id}"
        )
        _err(
            f"sweep-launch: no pod was created; network volume {volume_id} is orphaned and bills until deleted:"
        )
        _err(f"  {_delete_volume_command(volume_id)}")
        return 2 if saved else 1

    pod_id = str(pod["id"])
    cost = pod.get("cost")
    usd_per_h = (
        float(cost)
        if isinstance(cost, int | float) and not isinstance(cost, bool)
        else gpu.secure_usd_per_h
    )
    if not math.isclose(usd_per_h, gpu.secure_usd_per_h, abs_tol=1e-6):
        _err(
            f"sweep-launch: warning: pod cost {usd_per_h} USD/h differs from the catalog's {gpu.secure_usd_per_h}"
        )
    launch = replace(record, pod_id=pod_id, gpu_id=gpu.id, gpu_usd_per_h=usd_per_h)
    if not _save_ledger(
        ledger_path,
        [*launches, launch],
        note=f"pod {pod_id} ({name}) on volume {volume_id}, {usd_per_h} USD/h, created {created.isoformat()}",
    ):
        return 1
    print(f"launched pod {pod_id} on {gpu.id} in {plan.datacenter}")
    _print_cost(gpu, volume.size_gb)
    _print_max_hours(remaining, gpu, volume.size_gb)
    print(
        f"W&B project: https://wandb.ai/<your-entity>/{WANDB_PROJECT} (group = sweep id)"
    )
    resume = f"uv run sweep-launch --volume-id {volume_id}" + (
        " --smoke" if args.smoke else ""
    )
    print(
        f"the pod deletes itself when it exits; to resume this sweep on the same volume: {resume}"
    )
    print(
        f"network volume {volume_id} is kept after the pod ends; delete it when done:"
    )
    print(f"  {_delete_volume_command(volume_id)}")
    return 0


CreateOutcome = Literal["created", "failed", "ambiguous"]


def _create_pod(
    client: RunPodClient, plan: Plan, volume_id: str, pod_args: "_PodArgs"
) -> tuple[CreateOutcome, dict[str, object] | None, GpuType]:
    """Try the candidates in order. 400 moves on; another 4xx stops. A 5xx or a transport
    failure may have created the pod, so it is looked up by its unique name and never retried
    on another GPU (that could make a second pod)."""
    gpu = plan.candidates[0]
    for gpu in plan.candidates:
        try:
            return (
                "created",
                client.create_pod(_pod_body(gpu, plan.datacenter, volume_id, pod_args)),
                gpu,
            )
        except RunPodError as e:
            _err(f"sweep-launch: pod create on {gpu.id} failed: {e}")
            if e.status == 0 or e.status >= 500:
                return _lookup_after_ambiguous_create(client, pod_args.name, gpu)
            if e.status != 400:
                break
    return "failed", None, gpu


def _lookup_after_ambiguous_create(
    client: RunPodClient, name: str, gpu: GpuType
) -> tuple[CreateOutcome, dict[str, object] | None, GpuType]:
    try:
        found = client.find_pod_by_name(name)
    except RunPodError as e:
        _err(f"sweep-launch: could not look the pod up by name: {e}")
        return "ambiguous", None, gpu
    if found is None:
        return "ambiguous", None, gpu
    print(f"pod create reported an error, but pod {found['id']} named {name} exists")
    return "created", found, gpu


def _save_ledger(path: Path, launches: list[ledger.Launch], *, note: str) -> bool:
    """Write the ledger; on a disk error say exactly what to record by hand."""
    try:
        ledger.save(path, launches)
    except OSError as e:
        _err(f"sweep-launch: could not write the ledger {path} ({e})")
        _err(
            f"  record this launch by hand, or the next launch will undercount spend: {note}"
        )
        return False
    return True


def _pod_cap_usd(
    cap: float, launches: list[ledger.Launch], volume_id: str | None
) -> float:
    """`SWEEP_CAP_USD` for the pod.

    A new volume: `cap - spent` (all billed spend). A reused volume (`--volume-id`): the
    in-pod guard reloads that volume's own ledger and subtracts its sweep-run spend itself,
    so subtracting this volume's billed spend here too would count it twice. Instead:
    `cap - (billed spend of launches on other volumes) - SETUP_ALLOWANCE_USD x (earlier
    launches on this volume)`. The allowance stands in for pod time on those launches outside
    `sweep-run` (uv sync, GPU tests, the report), which the in-pod ledger never saw. Reading the
    in-pod ledger total directly would need the network-volume S3 API, which takes separate S3
    API keys (docs.runpod.io/storage/s3-api), so a fixed allowance it is."""
    if volume_id is None:
        return cap - ledger.total_spent_usd(launches)
    other = ledger.total_spent_usd([x for x in launches if x.volume_id != volume_id])
    same = sum(1 for x in launches if x.volume_id == volume_id)
    return cap - other - SETUP_ALLOWANCE_USD * same


def _volume(client: RunPodClient, volume_id: str | None) -> _Volume:
    if volume_id is None:
        return _Volume(
            id=None, datacenter=None, size_gb=NETWORK_VOLUME_GB, reused=False
        )
    found = client.get_volume(volume_id)
    if found.type != VOLUME_TYPE:
        raise VolumeUnusable(
            f"volume {volume_id} is {found.type}; only {VOLUME_TYPE} volumes are priced here"
        )
    return _Volume(
        id=found.id, datacenter=found.datacenter, size_gb=found.size_gb, reused=True
    )


@dataclass(frozen=True)
class _PodArgs:
    name: str
    secrets: Secrets
    repo_url: str
    ref: str
    remaining: float  # new money this launch may spend: sets max_hours
    pod_cap_usd: float  # SWEEP_CAP_USD: see _pod_cap_usd
    smoke: bool
    volume_gb: int

    def __repr__(self) -> str:
        return f"_PodArgs(name={self.name!r}, secrets=***)"


def _pod_body(
    gpu: GpuType, datacenter: str, volume_id: str, a: _PodArgs
) -> dict[str, object]:
    env = {
        POD_KEY_ENV: a.secrets.runpod_api_key,
        "WANDB_API_KEY": a.secrets.wandb_api_key,
        "REPO_URL": a.repo_url,
        "REPO_REF": a.ref,
        "SWEEP_CAP_USD": _floor2(a.pod_cap_usd),
        "SWEEP_GPU_USD_PER_H": repr(gpu.secure_usd_per_h),
        "SWEEP_MAX_HOURS": _floor2(_max_hours(a.remaining, gpu, a.volume_gb)),
        "SWEEP_STATE_DIR": STATE_DIR_SMOKE if a.smoke else STATE_DIR_FULL,
        "WANDB_PROJECT": WANDB_PROJECT,
    }
    if a.smoke:
        env["SWEEP_SMOKE"] = "1"
    if "RUNPOD_API_KEY" in env:
        raise AssertionError(
            "RUNPOD_API_KEY is RunPod's own pod variable; send ours as " + POD_KEY_ENV
        )
    return {
        "name": a.name,
        "image": IMAGE,
        "cloud": "SECURE",
        "gpu": {"id": gpu.id, "count": 1, "minCudaVersion": MIN_CUDA_VERSION},
        "disk": CONTAINER_DISK_GB,
        "mounts": {"network": [{"volumeId": volume_id, "path": VOLUME_MOUNT}]},
        "dataCenterIds": [datacenter],
        "env": env,
        # entrypoint overrides the image's own ENTRYPOINT; cmd is its argument list
        # (docs.runpod.io/api-reference-v2/pods/create-a-pod), so the image cannot wrap it.
        "entrypoint": ["bash", "-c"],
        "cmd": [CLONE_ONE_LINER],
    }


def _plan(
    gpu_id: str,
    gpus: list[GpuType],
    datacenters: list[DataCenter],
    *,
    only_dc: str | None,
    max_usd_per_h: float,
    explicit: bool,
) -> Plan | None:
    """Where to launch and which GPUs to try there, in order; None if nothing fits.

    Candidates are `gpu_id` first, then the fallbacks by ascending live price (ties keep
    `GPU_FALLBACKS` order); only GPUs with stock in the data center are kept, and only those at
    or under `max_usd_per_h` unless it is `gpu_id` named explicitly. With `only_dc`
    (a reused volume's data center) that data center is the only choice. Otherwise the
    STANDARD-volume data center whose best candidate ranks earliest wins, ties broken by how
    many candidates it has in stock (more fallbacks)."""
    ranked = [
        g
        for g in _ranked(gpu_id, gpus)
        if g.secure_usd_per_h <= max_usd_per_h or (explicit and g.id == gpu_id)
    ]
    if only_dc is not None:
        here = tuple(g for g in ranked if only_dc in g.available_datacenters)
        return Plan(datacenter=only_dc, candidates=here) if here else None
    best: tuple[tuple[int, int], Plan] | None = None
    for dc_id in _eligible_dcs(datacenters):
        here = tuple(g for g in ranked if dc_id in g.available_datacenters)
        if not here:
            continue
        key = (ranked.index(here[0]), -len(here))
        if best is None or key < best[0]:
            best = (key, Plan(datacenter=dc_id, candidates=here))
    return None if best is None else best[1]


def _ranked(gpu_id: str, gpus: list[GpuType]) -> list[GpuType]:
    """`gpu_id` first, then the fallbacks by ascending live price (stable: ties keep list order)."""
    by_id = {g.id: g for g in gpus}
    fallbacks = sorted(
        (by_id[c] for c in _candidate_ids(gpu_id)[1:] if c in by_id),
        key=lambda g: g.secure_usd_per_h,
    )
    return ([by_id[gpu_id]] if gpu_id in by_id else []) + fallbacks


def _eligible_dcs(datacenters: list[DataCenter]) -> list[str]:
    return [dc.id for dc in datacenters if VOLUME_TYPE in dc.network_volume_types]


def _print_options_over_ceiling(
    gpu_id: str,
    gpus: list[GpuType],
    datacenters: list[DataCenter],
    *,
    only_dc: str | None,
    ceiling: float,
    argv: list[str],
) -> None:
    """The in-stock candidates above the ceiling, cheapest first, each with the command that
    picks it explicitly (the user's other flags kept)."""
    dcs = [only_dc] if only_dc is not None else _eligible_dcs(datacenters)
    over = sorted(
        (g for g in _ranked(gpu_id, gpus) if g.secure_usd_per_h > ceiling),
        key=lambda g: g.secure_usd_per_h,
    )
    rest = _without_gpu_flag(argv)
    shown = 0
    for g in over:
        where = [dc for dc in dcs if dc in g.available_datacenters]
        if not where:
            continue
        shown += 1
        command = shlex.join(["uv", "run", "sweep-launch", "--gpu", g.id, *rest])
        _err(f"  {g.id}: {g.secure_usd_per_h} USD/h in {', '.join(where)}")
        _err(f"    {command}")
    if shown:
        _err(
            f"(above the {ceiling:.2f} USD/h ceiling: naming one with --gpu accepts its price)"
        )
    else:
        _err("  no candidate GPU has stock at any price")


def _without_gpu_flag(argv: list[str]) -> list[str]:
    rest: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
        elif arg == "--gpu":
            skip = True
        elif not arg.startswith("--gpu="):
            rest.append(arg)
    return rest


def _candidate_ids(gpu_id: str) -> tuple[str, ...]:
    return (gpu_id, *(g for g in GPU_FALLBACKS if g != gpu_id))


def _max_hours(remaining: float, gpu: GpuType, volume_gb: int) -> float:
    usd_per_h = gpu.secure_usd_per_h + volume_usd_per_h(volume_gb)
    if not usd_per_h > 0:
        raise RunPodError(
            200, f"catalog price for {gpu.id} is not positive: {gpu.secure_usd_per_h}"
        )
    hours = (remaining - MARGIN_USD) / usd_per_h
    if not hours > 0:
        raise AssertionError(
            f"max_hours {hours} must be positive once the budget check passed"
        )
    return hours


def _print_cost(gpu: GpuType, volume_gb: int) -> None:
    volume = volume_usd_per_h(volume_gb)
    total = gpu.secure_usd_per_h + volume
    print(
        f"{gpu.id}: {total:.3f} USD/h (gpu {gpu.secure_usd_per_h} + {volume_gb} GB volume {volume:.4f})"
    )


def _print_max_hours(remaining: float, gpu: GpuType, volume_gb: int) -> None:
    print(f"max_hours={_max_hours(remaining, gpu, volume_gb):.2f} on {gpu.id}")


def _floor2(x: float) -> str:
    """Two decimals, rounded down, so a printed cap or duration never exceeds the real one."""
    return f"{math.floor(x * 100) / 100:.2f}"


def _redacted_json(body: dict[str, object], secrets: Secrets) -> str:
    shown = copy.deepcopy(body)
    env = shown["env"]
    if not isinstance(env, dict):
        raise TypeError("pod body env must be a dict")
    for name in SECRET_ENV:
        env[name] = REDACTED
    text = json.dumps(shown, indent=2)
    if secrets.runpod_api_key in text or secrets.wandb_api_key in text:
        raise AssertionError("a secret survived redaction")
    return text


def _delete_volume_command(volume_id: str) -> str:
    # The key is referenced as a shell variable, never expanded here. RunPod refuses (409) a
    # delete within five minutes of the volume's creation.
    return (
        'curl -fsS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" '
        f"{API_V2}/network-volumes/{volume_id}"
    )


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_S,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise GitError(
            f"git {' '.join(args)} failed ({e}); pass the value explicitly"
        ) from e
    if not out:
        raise GitError(
            f"git {' '.join(args)} printed nothing; pass the value explicitly"
        )
    return out


def _cap(text: str) -> float:
    try:
        value = float(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from e
    if not 0 < value <= CAP_USD_DEFAULT:  # also rejects nan
        raise argparse.ArgumentTypeError(
            f"must be in (0, {CAP_USD_DEFAULT}], got {text}"
        )
    return value


def _price_ceiling(text: str) -> float:
    try:
        value = float(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from e
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"must be a positive finite USD/h, got {text}")
    return value


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sweep-launch",
        description="Launch the reset-curriculum sweep pod on RunPod within the USD cap.",
    )
    p.add_argument(
        "--gpu",
        default=None,
        help=f"GPU type id, used even above --max-usd-per-h (default {GPU_DEFAULT}; "
        f"fallbacks, cheapest first: {', '.join(GPU_FALLBACKS)})",
    )
    p.add_argument(
        "--max-usd-per-h",
        type=_price_ceiling,
        default=MAX_USD_PER_H_DEFAULT,
        help=f"price ceiling for automatic GPU picks (default {MAX_USD_PER_H_DEFAULT})",
    )
    p.add_argument(
        "--cap",
        type=_cap,
        default=CAP_USD_DEFAULT,
        help="account-wide USD cap, at most 100",
    )
    p.add_argument(
        "--ref",
        default=None,
        help="git ref the pod checks out (default: current branch)",
    )
    p.add_argument(
        "--repo-url",
        default=None,
        help="repo the pod clones (default: git remote origin)",
    )
    p.add_argument(
        "--volume-id",
        default=None,
        help="reuse this network volume (and its data center) to resume a sweep",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the request bodies, create nothing",
    )
    p.add_argument(
        "--smoke", action="store_true", help="short GPU smoke run (SWEEP_SMOKE=1)"
    )
    return p


def _err(message: str) -> None:
    print(message, file=sys.stderr)

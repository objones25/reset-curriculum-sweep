"""`sweep-launch` (spec section 9, Review Focus 3 and 4), against the fake RunPod API.

Budget numbers: cap 100, margin 5, a 20 GB volume at 0.07 USD/GB-month (730 h), so the
volume costs 20 * 0.07 / 730 USD/h. The 4090's fake price is 0.69 USD/h.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from runpod_fakes import (
    DC,
    FAKE_RUNPOD_KEY,
    FAKE_WANDB_KEY,
    PRICES,
    FakeRunPod,
    env_line,
)

from resetsweep.launch.launcher import (
    CLONE_ONE_LINER,
    STATE_DIR_FULL,
    STATE_DIR_SMOKE,
    GitError,
    main,
)

REPO = "https://github.com/example/reset-curriculum-sweep"
NOW = datetime(2026, 11, 25, 10, 30, tzinfo=UTC)
SHA = "0123456789abcdef0123456789abcdef01234567"  # what feat/sweep points at on the remote
HEAD_SHA = (
    "89abcdef0123456789abcdef0123456789abcdef"  # local HEAD (also pushed, on main)
)


class FakeGit:
    """Stands in for `git` (I7): `rev-parse HEAD` and `ls-remote <url>`. Records every call."""

    def __init__(
        self, refs: dict[str, str] | None = None, *, head: str = HEAD_SHA
    ) -> None:
        self.refs = (
            {"refs/heads/feat/sweep": SHA, "refs/heads/main": HEAD_SHA}
            if refs is None
            else refs
        )
        self.head = head
        self.calls: list[tuple[str, ...]] = []
        self.fail_ls_remote = False

    def __call__(self, *args: str) -> str:
        self.calls.append(args)
        if args == ("rev-parse", "HEAD"):
            return self.head
        if args[0] == "ls-remote":
            if self.fail_ls_remote:
                raise GitError("git ls-remote failed (repository not found)")
            return "\n".join(f"{sha}\t{ref}" for ref, sha in self.refs.items())
        raise AssertionError(f"unexpected git call {args}")


VOLUME_USD_PER_H = 20 * 0.07 / 730


def _no_sleep(_: float) -> None:
    raise AssertionError(
        "the launcher should not need to back off against the fake API"
    )


def _run(
    fake: FakeRunPod,
    env_path: Path,
    tmp_path: Path,
    *args: str,
    ledger: Path | None = None,
    git: FakeGit | None = None,
    ref: str | None = "feat/sweep",
) -> int:
    ref_args = ["--ref", ref] if ref is not None else []
    return main(
        ["--repo-url", REPO, *ref_args, *args],
        transport=fake.transport(),
        env_path=env_path,
        ledger_path=ledger or tmp_path / "ledger.json",
        sleep=_no_sleep,
        now=lambda: NOW,
        git=git or FakeGit(),
    )


def _write_ledger(path: Path, launches: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"version": 1, "launches": launches}), encoding="utf-8")


def _old_launch(
    pod_id: str | None = "pod-old", volume_id: str = "vol-old"
) -> dict[str, object]:
    return {
        "pod_id": pod_id,
        "volume_id": volume_id,
        "gpu_id": "NVIDIA A40",
        "gpu_usd_per_h": 0.4,
        "created_at": "2026-11-24T09:15:00+00:00",
        "spent_usd": 0.0,
        "smoke": True,
    }


@pytest.mark.parametrize("name", ["RUNPOD_API_KEY", "WANDB_API_KEY"])
@pytest.mark.parametrize("value", [None, "  "], ids=["absent", "blank"])
def test_missing_or_blank_key_raises_naming_key_before_any_http(
    fake_runpod: FakeRunPod,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    value: str | None,
) -> None:
    keys = {"RUNPOD_API_KEY": FAKE_RUNPOD_KEY, "WANDB_API_KEY": FAKE_WANDB_KEY}
    lines = [env_line(k, v) for k, v in keys.items() if k != name]
    lines += [] if value is None else [env_line(name, value)]
    env_path = tmp_path / ".env"
    env_path.write_text("".join(lines), encoding="utf-8")

    code = _run(fake_runpod, env_path, tmp_path)

    out = capsys.readouterr()
    assert code == 1
    assert fake_runpod.requests == []
    assert name in out.err
    assert FAKE_RUNPOD_KEY not in out.out + out.err
    assert FAKE_WANDB_KEY not in out.out + out.err


def test_missing_env_file_exits_1_before_any_http(
    fake_runpod: FakeRunPod, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(fake_runpod, tmp_path / "absent.env", tmp_path)

    assert code == 1
    assert fake_runpod.requests == []
    assert "absent.env" in capsys.readouterr().err


def test_create_pod_body_has_network_mount_secure_cloud_min_cuda_and_env(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    code = _run(fake_runpod, env_file, tmp_path)

    assert code == 0
    [volume] = fake_runpod.bodies("POST", "/v2/network-volumes")
    assert volume == {
        "name": "resetsweep-sweep-20261125-103000",
        "size": 20,
        "dataCenter": DC,
        "type": "STANDARD",
    }
    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    assert pod["cloud"] == "SECURE"
    assert pod["image"] == "runpod/pytorch:2.2.0-cu1290-torch291-ubuntu2404"
    assert pod["gpu"] == {
        "id": "NVIDIA GeForce RTX 4090",
        "count": 1,
        "minCudaVersion": "13.2",
    }
    assert pod["disk"] == 40
    assert pod["mounts"] == {"network": [{"volumeId": "vol-1", "path": "/workspace"}]}
    assert pod["dataCenterIds"] == [DC]
    assert pod["entrypoint"] == ["bash", "-c"]
    assert pod["cmd"] == [CLONE_ONE_LINER]
    env = pod["env"]
    assert isinstance(env, dict)
    assert env["SWEEP_RUNPOD_API_KEY"] == FAKE_RUNPOD_KEY
    assert (
        "RUNPOD_API_KEY" not in env
    )  # RunPod may inject its own pod-scoped key under that name
    assert env["WANDB_API_KEY"] == FAKE_WANDB_KEY
    assert env["REPO_URL"] == REPO
    assert env["REPO_REF"] == SHA  # I7: the branch resolved to the commit it points at
    assert env["SWEEP_CAP_USD"] == "100.00"
    assert env["SWEEP_GPU_USD_PER_H"] == "0.69"
    max_hours = (100.0 - 5.0) / (0.69 + VOLUME_USD_PER_H)
    assert float(env["SWEEP_MAX_HOURS"]) == pytest.approx(max_hours, abs=0.01)
    assert env["SWEEP_STATE_DIR"] == "/workspace/sweep"
    assert env["WANDB_PROJECT"] == "reset-curriculum-sweep"
    assert "SWEEP_SMOKE" not in env
    assert all(isinstance(v, str) for v in env.values())


def test_clone_one_liner_installs_self_delete_trap_before_the_specified_clone() -> None:
    trap = (
        'set -euo pipefail; self_delete() { if [ -n "${RUNPOD_POD_ID:-}" ] && '
        "[ -n \"${SWEEP_RUNPOD_API_KEY:-}\" ]; then printf 'Authorization: Bearer %s\\n' "
        '"$SWEEP_RUNPOD_API_KEY" | curl -fsS -X DELETE --max-time 30 --retry 5 '
        '--retry-all-errors -H @- "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" || true; fi; }; '
        "trap self_delete EXIT; "
        'if [ -e "${SWEEP_STATE_DIR:-/workspace/sweep}/DONE-${RUNPOD_POD_ID:-}" ]; then exit 0; fi; '
    )
    clone = (
        "cd /workspace; if [ -d repo ]; then git -C repo remote set-url origin "
        '"$REPO_URL"; else git init -q repo; git -C repo remote add origin "$REPO_URL"; fi; '
        'git -C repo fetch --depth 1 origin "$REPO_REF"; git -C repo checkout -q -f FETCH_HEAD; '
        "test -f repo/scripts/pod_bootstrap.sh; exec bash repo/scripts/pod_bootstrap.sh"
    )

    assert CLONE_ONE_LINER == trap + clone


def test_smoke_flag_sets_env(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    code = _run(fake_runpod, env_file, tmp_path, "--smoke")

    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    env = pod["env"]
    assert isinstance(env, dict)
    assert code == 0
    assert env["SWEEP_SMOKE"] == "1"
    assert env["SWEEP_STATE_DIR"] == "/workspace/sweep-smoke"


def test_smoke_projects_the_full_ladder_at_the_default_account_cap_not_the_smoke_cap(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    # A $5 smoke must still answer "what would the full $100 sweep do"; projecting at
    # the smoke's own --cap leaves 5 - 5 margin - 10 calibration < 0 and always says exit 5.
    code = _run(fake_runpod, env_file, tmp_path, "--cap", "5", "--smoke")

    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    env = pod["env"]
    assert isinstance(env, dict)
    assert code == 0
    assert env["SWEEP_PROJECT_CAP_USD"] == "100.00"


def test_state_dirs_differ_and_full_matches_sweep_run_default() -> None:
    from resetsweep.launch.launcher import WANDB_PROJECT
    from resetsweep.sweep.main import PROJECT_DEFAULT, STATE_DIR_DEFAULT

    assert STATE_DIR_FULL == STATE_DIR_DEFAULT
    assert STATE_DIR_SMOKE == "/workspace/sweep-smoke"
    assert WANDB_PROJECT == PROJECT_DEFAULT


def test_prints_pod_id_hourly_cost_max_hours_and_wandb_url(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(fake_runpod, env_file, tmp_path)

    out = capsys.readouterr().out
    assert "pod-1" in out
    assert "max_hours=137.30" in out
    assert "0.692 USD/h" in out
    assert "https://wandb.ai/" in out
    assert "reset-curriculum-sweep" in out
    assert "vol-1" in out


def test_dry_run_redacts_secrets_and_makes_no_post(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = tmp_path / "ledger.json"

    code = _run(fake_runpod, env_file, tmp_path, "--dry-run", ledger=ledger)

    out = capsys.readouterr()
    assert code == 0
    assert {r.method for r in fake_runpod.requests} == {"GET"}
    assert FAKE_RUNPOD_KEY not in out.out + out.err
    assert FAKE_WANDB_KEY not in out.out + out.err
    assert '"SWEEP_RUNPOD_API_KEY": "***"' in out.out
    assert '"RUNPOD_API_KEY"' not in out.out
    assert '"WANDB_API_KEY": "***"' in out.out
    assert '"minCudaVersion": "13.2"' in out.out
    assert '"size": 20' in out.out
    assert "max_hours=137.30" in out.out
    assert not ledger.exists()


def test_refuses_launch_when_balance_below_remaining_cap(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_runpod.balance_usd = 99.0

    code = _run(fake_runpod, env_file, tmp_path)

    assert code == 3
    assert fake_runpod.sent("POST", "/v2/network-volumes") == []
    assert fake_runpod.sent("POST", "/v2/pods") == []
    assert "balance" in capsys.readouterr().err


def test_refuses_launch_when_ledger_exhausted_cap(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_old_launch()])
    fake_runpod.pod_spend_usd["pod-old"] = 96.0

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    assert code == 3
    assert [r.method for r in fake_runpod.requests if r.method != "GET"] == []
    assert "96.00" in capsys.readouterr().err


def test_settles_earlier_launches_from_billing_before_computing_remaining(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_old_launch()])
    fake_runpod.pod_spend_usd["pod-old"] = 29.5
    fake_runpod.volume_spend_usd["vol-old"] = 0.5

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    [billing] = fake_runpod.sent("GET", "/v2/billing/pods")
    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    env = pod["env"]
    assert isinstance(env, dict)
    assert code == 0
    assert billing.url.params["podId"] == "pod-old"
    assert billing.url.params["startTime"] == "2026-11-24T09:00:00Z"
    assert env["SWEEP_CAP_USD"] == "70.00"
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert saved["launches"][0]["spent_usd"] == pytest.approx(30.0)


def test_orphan_volume_from_failed_launch_is_settled_too(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_old_launch(pod_id=None)])
    fake_runpod.volume_spend_usd["vol-old"] = 10.0

    _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    env = pod["env"]
    assert isinstance(env, dict)
    assert fake_runpod.sent("GET", "/v2/billing/pods") == []
    assert env["SWEEP_CAP_USD"] == "90.00"


def test_corrupt_ledger_refuses_before_any_post(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{not json", encoding="utf-8")

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    assert code == 1
    assert fake_runpod.requests == []
    assert "ledger" in capsys.readouterr().err


def test_launch_recorded_in_ledger_with_pod_id_price_and_time(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger.json"

    _run(fake_runpod, env_file, tmp_path, "--smoke", ledger=ledger)

    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert saved == {  # audit: ignore[FLOAT_EQ] 0.69 and 0.0 round-trip JSON exactly
        "version": 1,
        "launches": [
            {
                "pod_id": "pod-1",
                "volume_id": "vol-1",
                "gpu_id": "NVIDIA GeForce RTX 4090",
                "gpu_usd_per_h": 0.69,
                "created_at": "2026-11-25T10:30:00+00:00",
                "spent_usd": 0.0,
                "smoke": True,
                "name": "resetsweep-smoke-20261125-103000",
                "unresolved": False,
                # I3: the SWEEP_MAX_HOURS sent, rounded down; a smoke launch keeps the 0.50
                # smoke margin (I6): (100 - 0.5) / (0.69 + 20 * 0.07 / 730)
                "max_hours": 143.8,
                "repo_sha": SHA,  # I7: the commit the pod checks out
            }
        ],
    }


def test_picks_datacenter_that_supports_standard_network_volumes(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    _run(fake_runpod, env_file, tmp_path)

    [volume] = fake_runpod.bodies("POST", "/v2/network-volumes")
    assert volume["dataCenter"] == DC


def test_no_datacenter_with_gpu_and_volumes_refuses_before_any_post(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_runpod.datacenters = [
        dc | {"networkVolumeTypes": []} for dc in fake_runpod.datacenters
    ]

    code = _run(fake_runpod, env_file, tmp_path)

    assert code == 2
    assert fake_runpod.sent("POST", "/v2/network-volumes") == []
    assert "STANDARD" in capsys.readouterr().err


def test_falls_back_to_next_gpu_on_400(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    fake_runpod.pod_create_statuses = [400]

    code = _run(fake_runpod, env_file, tmp_path)

    bodies = fake_runpod.bodies("POST", "/v2/pods")
    gpu_ids = [b["gpu"]["id"] for b in bodies]  # type: ignore[index]
    second_env = bodies[1]["env"]
    assert isinstance(second_env, dict)
    assert code == 0
    assert gpu_ids == ["NVIDIA GeForce RTX 4090", "NVIDIA A40"]
    assert second_env["SWEEP_GPU_USD_PER_H"] == "0.4"
    assert len(fake_runpod.sent("POST", "/v2/network-volumes")) == 1


def test_falls_back_to_next_gpu_on_400_and_reports_orphan_volume_when_all_fail(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_runpod.pod_create_statuses = [400, 400, 400, 400]
    ledger = tmp_path / "ledger.json"

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    err = capsys.readouterr().err
    gpu_ids = [b["gpu"]["id"] for b in fake_runpod.bodies("POST", "/v2/pods")]  # type: ignore[index]
    assert code == 2
    assert gpu_ids == sorted(
        PRICES, key=lambda g: (g != "NVIDIA GeForce RTX 4090", PRICES[g])
    )
    assert len(fake_runpod.sent("POST", "/v2/network-volumes")) == 1
    assert "vol-1" in err
    assert "curl -fsS -X DELETE" in err
    assert "https://api.runpod.io/v2/network-volumes/vol-1" in err
    assert FAKE_RUNPOD_KEY not in err
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert [(x["pod_id"], x["volume_id"]) for x in saved["launches"]] == [
        (None, "vol-1")
    ]


def test_422_on_create_pod_does_not_try_other_gpus(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_runpod.pod_create_statuses = [422]

    code = _run(fake_runpod, env_file, tmp_path)

    assert code == 2
    assert len(fake_runpod.sent("POST", "/v2/pods")) == 1
    assert "vol-1" in capsys.readouterr().err


@pytest.mark.parametrize("cap", ["0", "-1", "100.01", "nan"])
def test_cap_outside_0_to_100_is_a_usage_error(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path, cap: str
) -> None:
    with pytest.raises(SystemExit) as info:
        _run(fake_runpod, env_file, tmp_path, "--cap", cap)

    assert info.value.code == 2
    assert fake_runpod.requests == []


def test_lower_cap_lowers_remaining(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    _run(fake_runpod, env_file, tmp_path, "--cap", "20")

    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    env = pod["env"]
    assert isinstance(env, dict)
    assert env["SWEEP_CAP_USD"] == "20.00"


def test_volume_create_5xx_warns_of_possible_orphan_and_creates_no_pod(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_runpod.volume_create_status = 503

    code = _run(fake_runpod, env_file, tmp_path)

    err = capsys.readouterr().err
    assert code == 1
    assert len(fake_runpod.sent("POST", "/v2/network-volumes")) == 1
    assert fake_runpod.sent("POST", "/v2/pods") == []
    assert "GET https://api.runpod.io/v2/network-volumes" in err


def test_settled_spend_is_never_lowered_by_a_lagging_billing_read(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_old_launch() | {"spent_usd": 40.0}])
    fake_runpod.pod_spend_usd["pod-old"] = 10.0

    _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    env = pod["env"]
    assert isinstance(env, dict)
    assert env["SWEEP_CAP_USD"] == "60.00"


def test_fallbacks_are_tried_in_ascending_live_price_and_unavailable_ones_skipped(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    prices = {
        "NVIDIA GeForce RTX 4090": 0.89,
        "NVIDIA RTX A5000": 0.27,
        "NVIDIA A40": 0.59,
        "NVIDIA L4": 0.69,
        "NVIDIA RTX A6000": 0.69,
        "NVIDIA L40S": 1.09,
        "NVIDIA RTX 6000 Ada Generation": 1.09,
        "NVIDIA A100-SXM4-80GB": 2.19,
    }
    out_of_stock = {"NVIDIA A40", "NVIDIA L40S"}
    fake_runpod.gpus = [
        {
            "id": g,
            "price": {"secure": p, "community": p},
            "dataCenters": [
                {"id": DC, "availability": "NONE" if g in out_of_stock else "LOW"}
            ],
        }
        for g, p in reversed(prices.items())
    ]
    fake_runpod.pod_create_statuses = [400] * 8

    code = _run(fake_runpod, env_file, tmp_path)

    gpu_ids = [b["gpu"]["id"] for b in fake_runpod.bodies("POST", "/v2/pods")]  # type: ignore[index]
    assert code == 2
    assert gpu_ids == [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A5000",
        "NVIDIA L4",
        "NVIDIA RTX A6000",
        "NVIDIA RTX 6000 Ada Generation",
    ]  # the A100 (2.19) is above the 1.20 USD/h automatic-pick ceiling


def test_fallback_list_holds_the_ruled_catalog_ids() -> None:
    from resetsweep.launch.launcher import GPU_FALLBACKS

    assert set(GPU_FALLBACKS) == {
        "NVIDIA A40",
        "NVIDIA L40S",
        "NVIDIA RTX A6000",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA L4",
        "NVIDIA RTX A5000",
        "NVIDIA A100-SXM4-80GB",
    }


OTHER_DC = "US-TX-3"


def _existing_volume(
    fake: FakeRunPod, *, dc: str = OTHER_DC, kind: str = "STANDARD"
) -> None:
    fake.volumes["vol-old"] = {
        "id": "vol-old",
        "name": "old",
        "size": 20,
        "dataCenter": dc,
        "type": kind,
    }
    fake.gpus = [
        g
        | {
            "dataCenters": [
                {"id": DC, "availability": "HIGH"},
                {"id": dc, "availability": "LOW"},
            ]
        }
        for g in fake.gpus
    ]


def test_volume_id_reuses_the_volume_and_pins_its_datacenter(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    _existing_volume(fake_runpod)
    ledger = tmp_path / "ledger.json"

    code = _run(
        fake_runpod, env_file, tmp_path, "--volume-id", "vol-old", ledger=ledger
    )

    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    assert code == 0
    assert fake_runpod.sent("POST", "/v2/network-volumes") == []
    assert pod["mounts"] == {"network": [{"volumeId": "vol-old", "path": "/workspace"}]}
    assert pod["dataCenterIds"] == [OTHER_DC]
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert saved["launches"][0]["volume_id"] == "vol-old"


def test_volume_id_failure_does_not_call_the_users_volume_orphaned(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _existing_volume(fake_runpod)
    fake_runpod.pod_create_statuses = [400] * 4

    code = _run(fake_runpod, env_file, tmp_path, "--volume-id", "vol-old")

    err = capsys.readouterr().err
    assert code == 2
    assert "orphaned" not in err
    assert "vol-old" in err


def test_volume_id_not_found_exits_1_before_any_post(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    code = _run(fake_runpod, env_file, tmp_path, "--volume-id", "vol-missing")

    assert code == 1
    assert [r.method for r in fake_runpod.requests if r.method != "GET"] == []


def test_volume_id_of_a_non_standard_volume_is_refused(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _existing_volume(fake_runpod, kind="HIGH_PERFORMANCE")

    code = _run(fake_runpod, env_file, tmp_path, "--volume-id", "vol-old")

    assert code == 1
    assert fake_runpod.sent("POST", "/v2/pods") == []
    assert "HIGH_PERFORMANCE" in capsys.readouterr().err


def test_volume_id_in_a_datacenter_without_stock_exits_2(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    fake_runpod.volumes["vol-old"] = {
        "id": "vol-old",
        "size": 20,
        "dataCenter": "XX-NONE-1",
        "type": "STANDARD",
    }

    code = _run(fake_runpod, env_file, tmp_path, "--volume-id", "vol-old")

    assert code == 2
    assert fake_runpod.sent("POST", "/v2/pods") == []


def test_shared_volume_spend_is_counted_once(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger.json"
    later = _old_launch(pod_id="pod-2") | {"created_at": "2026-11-24T20:00:00+00:00"}
    _write_ledger(ledger, [_old_launch(), later])
    fake_runpod.pod_spend_usd.update({"pod-old": 10.0, "pod-2": 20.0})
    fake_runpod.volume_spend_usd["vol-old"] = 5.0

    _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    env = pod["env"]
    assert isinstance(env, dict)
    assert env["SWEEP_CAP_USD"] == "65.00"


def test_prints_how_to_resume_on_the_same_volume(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(fake_runpod, env_file, tmp_path, "--smoke")

    assert "sweep-launch --volume-id vol-1 --smoke" in capsys.readouterr().out


def _priced_catalog(
    fake: FakeRunPod, prices: dict[str, float], out_of_stock: set[str]
) -> None:
    fake.gpus = [
        {
            "id": g,
            "price": {"secure": p, "community": p},
            "dataCenters": [
                {"id": DC, "availability": "NONE" if g in out_of_stock else "LOW"}
            ],
        }
        for g, p in prices.items()
    ]


def test_auto_pick_skips_gpus_above_price_ceiling(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    _priced_catalog(
        fake_runpod,
        {
            "NVIDIA GeForce RTX 4090": 0.89,
            "NVIDIA RTX A5000": 0.27,
            "NVIDIA A100-SXM4-80GB": 2.19,
        },
        out_of_stock=set(),
    )
    fake_runpod.pod_create_statuses = [400, 400, 400]

    code = _run(fake_runpod, env_file, tmp_path)

    gpu_ids = [b["gpu"]["id"] for b in fake_runpod.bodies("POST", "/v2/pods")]  # type: ignore[index]
    assert code == 2
    assert gpu_ids == ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000"]


def test_lower_price_ceiling_drops_the_default_gpu_too(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    _priced_catalog(
        fake_runpod,
        {"NVIDIA GeForce RTX 4090": 0.89, "NVIDIA RTX A5000": 0.27},
        out_of_stock=set(),
    )

    _run(fake_runpod, env_file, tmp_path, "--max-usd-per-h", "0.5")

    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    assert pod["gpu"]["id"] == "NVIDIA RTX A5000"  # type: ignore[index]


def test_explicit_gpu_overrides_price_ceiling(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    _priced_catalog(
        fake_runpod,
        {"NVIDIA A100-SXM4-80GB": 2.19, "NVIDIA RTX A5000": 0.27, "NVIDIA L40S": 1.5},
        out_of_stock=set(),
    )

    code = _run(fake_runpod, env_file, tmp_path, "--gpu", "NVIDIA A100-SXM4-80GB")

    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    env = pod["env"]
    assert isinstance(env, dict)
    assert code == 0
    assert pod["gpu"]["id"] == "NVIDIA A100-SXM4-80GB"  # type: ignore[index]
    assert env["SWEEP_GPU_USD_PER_H"] == "2.19"


def test_fallbacks_after_explicit_gpu_still_respect_the_ceiling(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    _priced_catalog(
        fake_runpod,
        {"NVIDIA A100-SXM4-80GB": 2.19, "NVIDIA RTX A5000": 0.27, "NVIDIA L40S": 1.5},
        out_of_stock=set(),
    )
    fake_runpod.pod_create_statuses = [400, 400]

    code = _run(fake_runpod, env_file, tmp_path, "--gpu", "NVIDIA A100-SXM4-80GB")

    gpu_ids = [b["gpu"]["id"] for b in fake_runpod.bodies("POST", "/v2/pods")]  # type: ignore[index]
    assert code == 2
    assert gpu_ids == ["NVIDIA A100-SXM4-80GB", "NVIDIA RTX A5000"]


def test_no_stock_under_ceiling_exits_2_with_options(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _priced_catalog(
        fake_runpod,
        {
            "NVIDIA GeForce RTX 4090": 0.89,
            "NVIDIA RTX A5000": 0.27,
            "NVIDIA A100-SXM4-80GB": 2.19,
            "NVIDIA L40S": 1.5,
        },
        out_of_stock={"NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000"},
    )

    code = _run(fake_runpod, env_file, tmp_path, "--smoke")

    err = capsys.readouterr().err
    assert code == 2
    assert [r.method for r in fake_runpod.requests if r.method != "GET"] == []
    assert "1.20 USD/h" in err
    assert f"NVIDIA L40S: 1.5 USD/h in {DC}" in err
    assert f"NVIDIA A100-SXM4-80GB: 2.19 USD/h in {DC}" in err
    assert err.index("NVIDIA L40S") < err.index("NVIDIA A100-SXM4-80GB")
    assert (
        "uv run sweep-launch --gpu 'NVIDIA L40S' --repo-url "
        f"{REPO} --ref feat/sweep --smoke"
    ) in err


@pytest.mark.parametrize("ceiling", ["0", "-1", "nan", "inf"])
def test_price_ceiling_must_be_positive_and_finite(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path, ceiling: str
) -> None:
    with pytest.raises(SystemExit) as info:
        _run(fake_runpod, env_file, tmp_path, "--max-usd-per-h", ceiling)

    assert info.value.code == 2
    assert fake_runpod.requests == []


# -- fix round 1 ------------------------------------------------------------------------


def _pod_env(fake: FakeRunPod) -> dict[str, str]:
    [pod] = fake.bodies("POST", "/v2/pods")
    env = pod["env"]
    assert isinstance(env, dict)
    return env  # type: ignore[return-value]


def test_resume_cap_does_not_double_count_this_volumes_spend(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    # $40 billed on the earlier launch of vol-old. The in-pod guard reloads vol-old's own
    # ledger (about $38 of sweep-run time) and subtracts it again, so the pod's cap must not
    # already exclude it: cap 100 - 0 (other volumes) - 2 (setup allowance, 1 earlier launch).
    # The pod may then spend about 98 - 38 - 5 margin = 55 new, 95 in total.
    _existing_volume(fake_runpod)
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_old_launch(volume_id="vol-old")])
    fake_runpod.pod_spend_usd["pod-old"] = 39.0
    fake_runpod.volume_spend_usd["vol-old"] = 1.0

    code = _run(
        fake_runpod, env_file, tmp_path, "--volume-id", "vol-old", ledger=ledger
    )

    env = _pod_env(fake_runpod)
    assert code == 0
    assert env["SWEEP_CAP_USD"] == "98.00"
    max_hours = (60.0 - 5.0) / (0.69 + VOLUME_USD_PER_H)  # new money: 100 - 40 spent
    assert float(env["SWEEP_MAX_HOURS"]) == pytest.approx(max_hours, abs=0.01)


def test_resume_cap_subtracts_other_volumes_spend_and_a_setup_allowance_per_launch(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    _existing_volume(fake_runpod)
    ledger = tmp_path / "ledger.json"
    second = _old_launch(pod_id="pod-2") | {"created_at": "2026-11-24T20:00:00+00:00"}
    other = _old_launch(pod_id="pod-x", volume_id="vol-other")
    _write_ledger(ledger, [_old_launch(), second, other])
    fake_runpod.pod_spend_usd.update({"pod-old": 20.0, "pod-2": 5.0, "pod-x": 10.0})
    fake_runpod.volume_spend_usd["vol-other"] = 0.5

    _run(fake_runpod, env_file, tmp_path, "--volume-id", "vol-old", ledger=ledger)

    assert _pod_env(fake_runpod)["SWEEP_CAP_USD"] == "85.50"  # 100 - 10.5 - 2 * 2


def test_pod_create_5xx_then_found_by_name_proceeds_as_created(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_runpod.pod_create_statuses = [503]
    fake_runpod.created_despite_error = True
    ledger = tmp_path / "ledger.json"

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    out = capsys.readouterr().out
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert code == 0
    assert len(fake_runpod.sent("POST", "/v2/pods")) == 1
    assert "launched pod pod-1" in out
    assert [(x["pod_id"], x["unresolved"]) for x in saved["launches"]] == [
        ("pod-1", False)
    ]


def test_pod_create_5xx_then_unresolved_records_and_warns(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_runpod.pod_create_statuses = [503]
    ledger = tmp_path / "ledger.json"

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    err = capsys.readouterr().err
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    [launch] = saved["launches"]
    assert code == 2
    assert len(fake_runpod.sent("POST", "/v2/pods")) == 1  # never a second pod
    assert "a pod may exist" in err
    assert "orphaned" not in err
    assert launch["pod_id"] is None
    assert launch["unresolved"] is True
    assert launch["name"] == "resetsweep-sweep-20261125-103000"


def test_transport_failure_on_pod_create_is_treated_as_ambiguous(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        is_create = (request.method, request.url.path) == ("POST", "/v2/pods")
        if is_create:  # audit: ignore[BRANCHING] fake transport routing
            fake_runpod.requests.append(request)
            raise httpx.ReadTimeout("timed out", request=request)
        return fake_runpod.handler(request)

    code = main(
        ["--repo-url", REPO, "--ref", "feat/sweep"],
        transport=httpx.MockTransport(handler),
        env_path=env_file,
        ledger_path=tmp_path / "ledger.json",
        sleep=_no_sleep,
        now=lambda: NOW,
        git=FakeGit(),
    )

    assert code == 2
    assert "a pod may exist" in capsys.readouterr().err


def _unresolved(name: str = "resetsweep-sweep-old") -> dict[str, object]:
    return _old_launch(pod_id=None) | {"name": name, "unresolved": True}


def test_unresolved_launch_blocks_later_launches_until_resolved(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_unresolved()])

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    err = capsys.readouterr().err
    assert code == 3
    assert [r.method for r in fake_runpod.requests if r.method != "GET"] == []
    assert "resetsweep-sweep-old" in err
    assert "console" in err


def test_unresolved_launch_found_by_name_is_resolved_and_settled(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_unresolved()])
    fake_runpod.listed_pods = [{"id": "pod-found", "name": "resetsweep-sweep-old"}]
    fake_runpod.pod_spend_usd["pod-found"] = 7.0

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert code == 0
    assert saved["launches"][0]["pod_id"] == "pod-found"
    assert saved["launches"][0]["unresolved"] is False
    assert _pod_env(fake_runpod)["SWEEP_CAP_USD"] == "93.00"


def test_old_ledger_entries_without_name_still_load(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_old_launch()])

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    assert code == 0


def test_wandb_key_not_in_error_output_when_api_echoes_body(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_runpod.pod_create_statuses = [422]
    fake_runpod.echo_body_on_error = True

    code = _run(fake_runpod, env_file, tmp_path)

    out = capsys.readouterr()
    assert code == 2
    assert "fake 422" in out.err
    assert FAKE_WANDB_KEY not in out.out + out.err
    assert FAKE_RUNPOD_KEY not in out.out + out.err


def test_explicit_gpu_not_in_catalog_is_a_usage_error(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = _run(fake_runpod, env_file, tmp_path, "--gpu", "NVIDIA RTX 9999")

    assert code == 2
    assert [r.method for r in fake_runpod.requests if r.method != "GET"] == []
    assert "NVIDIA RTX 9999" in capsys.readouterr().err


def test_ledger_save_failure_after_pod_create_prints_pod_id_and_exits_1(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")

    code = _run(fake_runpod, env_file, tmp_path, ledger=blocker / "ledger.json")

    err = capsys.readouterr().err
    assert code == 1
    assert "pod-1" in err
    assert "vol-1" in err
    assert "ledger" in err


def test_connect_failure_on_pod_create_is_not_created_not_unresolved(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        is_create = (request.method, request.url.path) == ("POST", "/v2/pods")
        if is_create:  # audit: ignore[BRANCHING] fake transport routing
            fake_runpod.requests.append(request)
            raise httpx.ConnectError("connection refused", request=request)
        return fake_runpod.handler(request)

    ledger = tmp_path / "ledger.json"

    code = main(
        ["--repo-url", REPO, "--ref", "feat/sweep"],
        transport=httpx.MockTransport(handler),
        env_path=env_file,
        ledger_path=ledger,
        sleep=lambda _: None,
        now=lambda: NOW,
        git=FakeGit(),
    )

    err = capsys.readouterr().err
    [launch] = json.loads(ledger.read_text(encoding="utf-8"))["launches"]
    assert code == 2
    assert (
        fake_runpod.sent("GET", "/v2/pods") == []
    )  # no lookup: nothing reached the server
    assert "a pod may exist" not in err
    assert "orphaned" in err
    assert launch["unresolved"] is False


# -- an earlier pod that still exists blocks a new launch --------------------------------


@pytest.mark.parametrize("status", ["PROVISIONING", "STARTING", "RUNNING", "ERROR"])
def test_refuses_launch_while_earlier_pod_running(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: str,
) -> None:
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_old_launch()])
    fake_runpod.pod_states["pod-old"] = status

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    err = capsys.readouterr().err
    assert code == 2
    assert [r.method for r in fake_runpod.requests if r.method != "GET"] == []
    assert f"pod-old ({status})" in err
    assert "console" in err
    assert "https://api.runpod.io/v2/pods/pod-old" in err


def test_refuses_launch_while_earlier_pod_is_stopped_but_billing_storage(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_old_launch()])
    fake_runpod.pod_states["pod-old"] = "EXITED"

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    err = capsys.readouterr().err
    assert code == 2
    assert "pod-old (EXITED)" in err
    assert "still bills" in err


@pytest.mark.parametrize("status", [None, "TERMINATED"], ids=["404", "terminated"])
def test_allows_launch_when_earlier_pods_are_gone(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path, status: str | None
) -> None:
    ledger = tmp_path / "ledger.json"
    _write_ledger(
        ledger, [_old_launch(), _old_launch(pod_id=None, volume_id="vol-orphan")]
    )
    if status is not None:  # audit: ignore[BRANCHING] None means the API answers 404
        fake_runpod.pod_states["pod-old"] = status

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    looked_up = [r.url.path for r in fake_runpod.sent("GET", "/v2/pods/pod-old")]
    assert code == 0
    assert looked_up == ["/v2/pods/pod-old"]
    assert len(fake_runpod.sent("POST", "/v2/pods")) == 1


def test_dry_run_reports_running_pod(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_old_launch()])
    fake_runpod.pod_states["pod-old"] = "RUNNING"

    code = _run(fake_runpod, env_file, tmp_path, "--dry-run", ledger=ledger)

    err = capsys.readouterr().err
    assert code == 2
    assert {r.method for r in fake_runpod.requests} == {"GET"}
    assert "pod-old (RUNNING)" in err


def test_billing_that_reads_zero_is_floored_by_the_earlier_pods_max_hours(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    # Final-review I3. The old pod (0.40 USD/h, max_hours 10) was created 25.25 h before NOW and
    # billing reads 0: its spend is floored at 0.4 * 10 = 4.00 plus 25.25 h of 20 GB storage
    # (20 * 0.07 / 730 * 25.25 = 0.048) -> SWEEP_CAP_USD = floor2(100 - 4.048) = 95.95.
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_old_launch() | {"max_hours": 10.0}])

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    env = pod["env"]
    assert isinstance(env, dict)
    assert code == 0
    assert env["SWEEP_CAP_USD"] == "95.95"
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert saved["launches"][0]["spent_usd"] == pytest.approx(
        4.0 + 20 * 0.07 / 730 * 25.25
    )


# --- smoke margin (final-review I6) ---------------------------------------------------------------


def test_readme_smoke_command_cap_5_is_accepted_with_the_smoke_margin(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The README's `sweep-launch --cap 5 --smoke`: the smoke margin is 0.50, so
    # max_hours = (5 - 0.5) / (0.69 + 20 * 0.07 / 730) = 6.50.
    code = _run(fake_runpod, env_file, tmp_path, "--cap", "5", "--smoke", "--dry-run")

    out = capsys.readouterr().out
    assert code == 0
    assert "max_hours=6.50" in out
    assert '"SWEEP_MAX_HOURS": "6.50"' in out


def test_full_run_keeps_the_5_usd_margin(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    assert _run(fake_runpod, env_file, tmp_path, "--cap", "5", "--dry-run") == 3


# --- the pod's code is pinned to a commit (final-review I7) ---------------------------------------


def test_default_ref_is_head_sha_checked_on_the_remote_and_recorded(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    git = FakeGit()
    ledger = tmp_path / "ledger.json"

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger, git=git, ref=None)

    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    env = pod["env"]
    assert isinstance(env, dict)
    assert code == 0
    assert env["REPO_REF"] == HEAD_SHA
    assert git.calls == [("rev-parse", "HEAD"), ("ls-remote", REPO)]
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert saved["launches"][0]["repo_sha"] == HEAD_SHA


def test_real_git_runs_ls_remote_with_terminal_prompts_disabled(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # smoke-patch item 4: the default runner (no FakeGit injected) runs a stub `git` on PATH,
    # which logs GIT_TERMINAL_PROMPT per call, so a private or missing repo fails instead of
    # waiting for a username on the terminal.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "git.log"
    stub = bin_dir / "git"
    stub.write_text(
        "#!/bin/sh\n"
        f'printf "%s %s\\n" "$1" "${{GIT_TERMINAL_PROMPT-unset}}" >> "{log}"\n'
        f'if [ "$1" = ls-remote ]; then printf "{SHA}\\trefs/heads/feat/sweep\\n"; fi\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.delenv("GIT_TERMINAL_PROMPT", raising=False)

    code = main(
        ["--repo-url", REPO, "--ref", "feat/sweep", "--dry-run"],
        transport=fake_runpod.transport(),
        env_path=env_file,
        ledger_path=tmp_path / "ledger.json",
        sleep=_no_sleep,
        now=lambda: NOW,
    )

    assert code == 0
    assert log.read_text(encoding="utf-8").splitlines() == ["ls-remote 0"]


def test_unpushed_sha_is_refused_with_exit_2_before_any_http(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    git = FakeGit(head="f" * 40)  # local commit the remote does not serve

    code = _run(fake_runpod, env_file, tmp_path, "--dry-run", git=git, ref=None)

    assert code == 2
    assert "push your branch first" in capsys.readouterr().err
    assert fake_runpod.requests == []


def test_missing_remote_repo_is_refused_with_exit_2(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    git = FakeGit()
    git.fail_ls_remote = True

    code = _run(fake_runpod, env_file, tmp_path, "--dry-run", git=git)

    assert code == 2
    assert "push your branch first" in capsys.readouterr().err
    assert fake_runpod.requests == []


def test_unknown_branch_is_refused_with_exit_2(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = _run(fake_runpod, env_file, tmp_path, "--dry-run", ref="feat/not-pushed")

    assert code == 2
    assert "push your branch first" in capsys.readouterr().err


@pytest.mark.parametrize(
    "url", ["git@github.com:example/repo.git", "http://github.com/example/repo"]
)
def test_non_https_repo_url_is_refused_with_exit_2(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path, url: str
) -> None:
    code = main(
        ["--repo-url", url, "--dry-run"],
        transport=fake_runpod.transport(),
        env_path=env_file,
        ledger_path=tmp_path / "ledger.json",
        sleep=_no_sleep,
        now=lambda: NOW,
        git=FakeGit(),
    )

    assert code == 2
    assert fake_runpod.requests == []


def test_volume_resume_defaults_to_the_sha_recorded_for_that_volume(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    _existing_volume(fake_runpod)
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_old_launch() | {"repo_sha": SHA}])
    git = FakeGit()

    code = _run(
        fake_runpod,
        env_file,
        tmp_path,
        "--volume-id",
        "vol-old",
        ledger=ledger,
        git=git,
        ref=None,
    )

    [pod] = fake_runpod.bodies("POST", "/v2/pods")
    env = pod["env"]
    assert isinstance(env, dict)
    assert code == 0
    assert env["REPO_REF"] == SHA
    assert ("rev-parse", "HEAD") not in git.calls


def test_volume_resume_with_another_ref_warns_about_exit_4(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _existing_volume(fake_runpod)
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, [_old_launch() | {"repo_sha": SHA}])

    code = _run(
        fake_runpod,
        env_file,
        tmp_path,
        "--volume-id",
        "vol-old",
        "--dry-run",
        ledger=ledger,
        ref="main",
    )

    err = capsys.readouterr().err
    assert code == 0
    assert SHA in err and HEAD_SHA in err
    assert "exit 4" in err
    # smoke-patch item 3: sweep-run enforces the commit itself, not only on an arm-set change
    assert "sweep_meta.json" in err
    assert "if that changes" not in err.lower()


# --- the created pod's reported price (final-review minor) ---------------------------------------


def test_higher_reported_pod_cost_rewrites_price_and_max_hours(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    # Catalog 0.69, but the create response says cost 0.80 USD/h (docs: `cost`, "The pod's
    # hourly compute rate in USD"). SWEEP_MAX_HOURS must be redone at 0.80:
    # floor2((100 - 5) / (0.80 + 20 * 0.07 / 730)) = 118.46.
    fake_runpod.create_cost = 0.80
    ledger = tmp_path / "ledger.json"

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    [patch] = fake_runpod.bodies("PATCH", "/v2/pods/pod-1")
    env = patch["env"]
    assert isinstance(env, dict)
    assert code == 0
    assert env["SWEEP_GPU_USD_PER_H"] == "0.8"
    assert env["SWEEP_MAX_HOURS"] == "118.46"
    assert env["REPO_REF"] == SHA  # the rest of the env is resent unchanged
    saved = json.loads(ledger.read_text(encoding="utf-8"))["launches"][0]
    assert (saved["gpu_usd_per_h"], saved["max_hours"]) == (0.8, 118.46)


def test_lower_or_equal_reported_cost_leaves_the_pod_alone(
    fake_runpod: FakeRunPod, env_file: Path, tmp_path: Path
) -> None:
    fake_runpod.create_cost = 0.60

    code = _run(fake_runpod, env_file, tmp_path)

    assert code == 0
    assert fake_runpod.sent("PATCH", "/v2/pods/pod-1") == []


def test_failed_price_rewrite_deletes_the_pod_and_refuses(
    fake_runpod: FakeRunPod,
    env_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_runpod.create_cost = 0.80
    fake_runpod.patch_status = 422
    ledger = tmp_path / "ledger.json"

    code = _run(fake_runpod, env_file, tmp_path, ledger=ledger)

    assert code == 3
    assert len(fake_runpod.sent("DELETE", "/v2/pods/pod-1")) == 1
    assert "0.8" in capsys.readouterr().err
    saved = json.loads(ledger.read_text(encoding="utf-8"))["launches"][0]
    assert saved["pod_id"] == "pod-1"  # recorded, so the spend floor still counts it

"""`sweep-launch` (spec section 9, Review Focus 3 and 4), against the fake RunPod API.

Budget numbers: cap 100, margin 5, a 20 GB volume at 0.07 USD/GB-month (730 h), so the
volume costs 20 * 0.07 / 730 USD/h. The 4090's fake price is 0.69 USD/h.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

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
    main,
)

REPO = "https://github.com/example/reset-curriculum-sweep"
NOW = datetime(2026, 11, 25, 10, 30, tzinfo=UTC)
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
) -> int:
    return main(
        ["--repo-url", REPO, "--ref", "feat/sweep", *args],
        transport=fake.transport(),
        env_path=env_path,
        ledger_path=ledger or tmp_path / "ledger.json",
        sleep=_no_sleep,
        now=lambda: NOW,
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
    assert pod["cmd"] == ["bash", "-c", CLONE_ONE_LINER]
    env = pod["env"]
    assert isinstance(env, dict)
    assert env["RUNPOD_API_KEY"] == FAKE_RUNPOD_KEY
    assert env["WANDB_API_KEY"] == FAKE_WANDB_KEY
    assert env["REPO_URL"] == REPO
    assert env["REPO_REF"] == "feat/sweep"
    assert env["SWEEP_CAP_USD"] == "100.00"
    assert env["SWEEP_GPU_USD_PER_H"] == "0.69"
    max_hours = (100.0 - 5.0) / (0.69 + VOLUME_USD_PER_H)
    assert float(env["SWEEP_MAX_HOURS"]) == pytest.approx(max_hours, abs=0.01)
    assert env["SWEEP_STATE_DIR"] == "/workspace/sweep"
    assert env["WANDB_PROJECT"] == "reset-curriculum-sweep"
    assert "SWEEP_SMOKE" not in env
    assert all(isinstance(v, str) for v in env.values())


def test_clone_one_liner_is_the_specified_command() -> None:
    assert CLONE_ONE_LINER == (
        "set -euo pipefail; cd /workspace; if [ -d repo ]; then git -C repo fetch --depth 1 origin "
        '"$REPO_REF" && git -C repo checkout FETCH_HEAD; else git clone --depth 1 --branch '
        '"$REPO_REF" "$REPO_URL" repo; fi; exec bash repo/scripts/pod_bootstrap.sh'
    )


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
    assert '"RUNPOD_API_KEY": "***"' in out.out
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
    assert gpu_ids == list(PRICES)
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

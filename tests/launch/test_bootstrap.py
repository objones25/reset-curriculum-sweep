"""`scripts/pod_bootstrap.sh`: static checks, then real runs with stub `python3`, `uv` and `curl`.

The stubs append their arguments to `calls.log`, so each test reads the exact command order. The
fake `uv` fails the step named in `FAIL_STEP`. Nothing touches the network.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from runpod_fakes import FAKE_RUNPOD_KEY

from resetsweep.launch.launcher import CLONE_ONE_LINER

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "pod_bootstrap.sh"
BASH = shutil.which("bash") or "/bin/bash"
DELETE_URL = "https://api.runpod.io/v2/pods/pod-xyz"
INJECTED_KEY = (
    "pod-scoped-INJECTED-by-runpod"  # RunPod's own RUNPOD_API_KEY; must not be used
)

STUB = """#!/usr/bin/env bash
name=$(basename "$0")
printf '%s %s\\n' "$name" "$*" >> "$CALLS_LOG"
if [[ $name == curl ]]; then cat > "$CALLS_LOG.curl-stdin"; fi
if [[ $name != curl && -n ${FAIL_STEP:-} && "$name $*" == *"$FAIL_STEP"* ]]; then exit 7; fi
if [[ $name == git && $1 == clone ]]; then
  for last; do :; done
  mkdir -p "$last/scripts"
  printf '%s\n' 'printf "bootstrap ran\\n" >> "$CALLS_LOG"' > "$last/scripts/pod_bootstrap.sh"
fi
exit 0
"""


def test_bootstrap_script_passes_bash_n() -> None:
    result = subprocess.run(
        [BASH, "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_bootstrap_script_passes_bash_n_and_shellcheck() -> None:
    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:  # audit: ignore[BRANCHING] optional tool
        pytest.skip("shellcheck not installed (command -v shellcheck found nothing)")
    syntax = subprocess.run(
        [BASH, "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    lint = subprocess.run(
        [shellcheck, "--severity=style", "--shell=bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    one_liner = subprocess.run(
        [shellcheck, "--severity=style", "--shell=bash", "-"],
        input=CLONE_ONE_LINER,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert syntax.returncode == 0, syntax.stderr
    assert lint.returncode == 0, lint.stdout
    assert one_liner.returncode == 0, one_liner.stdout


def _stub_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python3", "uv", "curl", "git"):
        stub = bin_dir / name
        stub.write_text(STUB, encoding="utf-8")
        stub.chmod(0o755)
    return bin_dir


def _run(
    tmp_path: Path, *, pod_id: str = "pod-xyz", fail_step: str = ""
) -> tuple[int, list[str], str]:
    bin_dir = _stub_bin(tmp_path)
    calls = tmp_path / "calls.log"
    state = tmp_path / "state"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CALLS_LOG": str(calls),
        "FAIL_STEP": fail_step,
        "RUNPOD_POD_ID": pod_id,
        "SWEEP_RUNPOD_API_KEY": FAKE_RUNPOD_KEY,
        "RUNPOD_API_KEY": INJECTED_KEY,
        "SWEEP_STATE_DIR": str(state),
    }
    result = subprocess.run(
        [BASH, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
        check=False,
    )
    lines = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
    log = (state / "bootstrap.log").read_text(encoding="utf-8")
    return result.returncode, lines, log + result.stdout + result.stderr


def test_success_runs_steps_in_order_then_terminates_pod(tmp_path: Path) -> None:
    code, calls, _ = _run(tmp_path)

    assert code == 0
    assert calls[:4] == [
        "python3 -m pip install uv==0.13.7",
        "uv sync --frozen",
        "uv run pytest -m gpu -q",
        "uv run sweep-run",
    ]
    assert calls[4].startswith("curl -fsS -X DELETE")
    assert calls[4].endswith(DELETE_URL)
    assert len(calls) == 5


def test_failed_gpu_tests_stop_before_sweep_and_still_terminate_pod(
    tmp_path: Path,
) -> None:
    code, calls, _ = _run(tmp_path, fail_step="pytest")

    assert code == 7
    assert "uv run sweep-run" not in calls
    assert calls[-1].endswith(DELETE_URL)


def test_failed_sweep_still_terminates_pod(tmp_path: Path) -> None:
    code, calls, _ = _run(tmp_path, fail_step="sweep-run")

    assert code == 7
    assert calls[-1].endswith(DELETE_URL)


def test_empty_pod_id_refuses_to_start_and_calls_no_curl(tmp_path: Path) -> None:
    code, calls, output = _run(tmp_path, pod_id="")

    assert code == 1
    assert calls == []
    assert "RUNPOD_POD_ID" in output


def test_api_key_never_appears_in_log_or_output(tmp_path: Path) -> None:
    _, _, output = _run(tmp_path, fail_step="sync")

    assert FAKE_RUNPOD_KEY not in output
    assert INJECTED_KEY not in output
    assert "uv sync --frozen" in output


def test_api_key_is_not_on_any_command_line(tmp_path: Path) -> None:
    _, calls, _ = _run(tmp_path)

    assert calls[-1].endswith(DELETE_URL)
    assert all(FAKE_RUNPOD_KEY not in line for line in calls)
    header = (tmp_path / "calls.log.curl-stdin").read_text(encoding="utf-8")
    assert header == f"Authorization: Bearer {FAKE_RUNPOD_KEY}\n"


def test_self_terminate_uses_our_key_not_runpods_injected_one(tmp_path: Path) -> None:
    _run(tmp_path)

    header = (tmp_path / "calls.log.curl-stdin").read_text(encoding="utf-8")
    assert header == f"Authorization: Bearer {FAKE_RUNPOD_KEY}\n"


def test_empty_sweep_key_refuses_to_start(tmp_path: Path) -> None:
    bin_dir = _stub_bin(tmp_path)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CALLS_LOG": str(tmp_path / "calls.log"),
        "RUNPOD_POD_ID": "pod-xyz",
        "RUNPOD_API_KEY": INJECTED_KEY,
        "SWEEP_STATE_DIR": str(tmp_path / "state"),
    }

    result = subprocess.run(
        [BASH, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 1
    assert not (tmp_path / "calls.log").exists()
    assert "SWEEP_RUNPOD_API_KEY is empty" in result.stdout + result.stderr


def _run_one_liner(
    tmp_path: Path,
    *,
    pod_id: str = "pod-xyz",
    fail_step: str = "",
    repo_exists: bool = False,
) -> tuple[int, list[str]]:
    """The launcher's pod command, with `/workspace` swapped for a temp dir (once, checked)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if repo_exists:
        (workspace / "repo" / "scripts").mkdir(parents=True)
    assert CLONE_ONE_LINER.count("cd /workspace;") == 1
    command = CLONE_ONE_LINER.replace("cd /workspace;", f"cd {workspace};")
    calls = tmp_path / "calls.log"
    env = {
        "PATH": f"{_stub_bin(tmp_path)}{os.pathsep}/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CALLS_LOG": str(calls),
        "FAIL_STEP": fail_step,
        "RUNPOD_POD_ID": pod_id,
        "SWEEP_RUNPOD_API_KEY": FAKE_RUNPOD_KEY,
        "RUNPOD_API_KEY": INJECTED_KEY,
        "REPO_URL": "https://example.invalid/repo.git",
        "REPO_REF": "feat/sweep",
    }
    result = subprocess.run(
        [BASH, "-c", command],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    lines = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
    return result.returncode, lines


def test_one_liner_failed_clone_still_terminates_pod(tmp_path: Path) -> None:
    code, calls = _run_one_liner(tmp_path, fail_step="git clone")

    header = (tmp_path / "calls.log.curl-stdin").read_text(encoding="utf-8")
    assert code == 7
    assert calls[0].startswith("git clone --depth 1 --branch feat/sweep")
    assert calls[-1].startswith("curl -fsS -X DELETE")
    assert calls[-1].endswith(DELETE_URL)
    assert header == f"Authorization: Bearer {FAKE_RUNPOD_KEY}\n"


def test_one_liner_failed_fetch_still_terminates_pod(tmp_path: Path) -> None:
    code, calls = _run_one_liner(tmp_path, fail_step="fetch", repo_exists=True)

    assert code != 0
    assert "bootstrap ran" not in calls
    assert calls[-1].endswith(DELETE_URL)


def test_one_liner_missing_bootstrap_script_still_terminates_pod(
    tmp_path: Path,
) -> None:
    code, calls = _run_one_liner(tmp_path, repo_exists=True)

    assert code != 0
    assert calls[-1].endswith(DELETE_URL)


def test_one_liner_hands_over_to_bootstrap_without_deleting(tmp_path: Path) -> None:
    code, calls = _run_one_liner(tmp_path)

    assert code == 0
    assert calls[-1] == "bootstrap ran"
    assert not any(c.startswith("curl") for c in calls)


def test_one_liner_without_pod_id_attempts_no_delete(tmp_path: Path) -> None:
    code, calls = _run_one_liner(tmp_path, pod_id="", fail_step="git clone")

    assert code == 7
    assert not any(c.startswith("curl") for c in calls)


def test_one_liner_never_puts_a_key_on_a_command_line(tmp_path: Path) -> None:
    _, calls = _run_one_liner(tmp_path, fail_step="git clone")

    assert all(FAKE_RUNPOD_KEY not in c and INJECTED_KEY not in c for c in calls)
    assert "$SWEEP_RUNPOD_API_KEY" in CLONE_ONE_LINER
    assert 'RUNPOD_API_KEY"' not in CLONE_ONE_LINER.replace("SWEEP_RUNPOD_API_KEY", "")

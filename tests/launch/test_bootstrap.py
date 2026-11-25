"""`scripts/pod_bootstrap.sh`: static checks, then real runs with stub `python3`, `uv` and `curl`.

The stubs append their arguments to `calls.log`, so each test reads the exact command order. The
fake `uv` fails the step named in `FAIL_STEP`. Nothing touches the network.
"""

import os
import re
import shutil
import subprocess
import sys
import time
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
if [[ $name == curl ]]; then
  header=$(cat)
  printf '%s\n' "$header" >> "$CALLS_LOG.curl-stdin"
  if [[ -n ${CURL_FAIL_MATCH:-} && $header == *"$CURL_FAIL_MATCH"* ]]; then exit 22; fi
fi
if [[ $name == runpodctl ]]; then exit "${RUNPODCTL_RC:-0}"; fi
if [[ $name == sleep ]]; then /bin/sleep 0.05; exit 0; fi
if [[ $name != curl && -n ${FAIL_STEP:-} && "$name $*" == *"$FAIL_STEP"* ]]; then exit 7; fi
if [[ -n ${SLOW_STEP:-} && "$name $*" == *"$SLOW_STEP"* ]]; then exec /bin/sleep 30; fi
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


def _gnu_timeout() -> str | None:
    """A GNU coreutils `timeout` on this machine, or None (then the watchdog path is used)."""
    path = shutil.which("timeout") or shutil.which("gtimeout")
    if path is None:
        return None
    version = subprocess.run(
        [path, "--version"], capture_output=True, text=True, timeout=10, check=False
    ).stdout
    return path if "coreutils" in version else None


def _stub_bin(
    tmp_path: Path, *, runpodctl: bool = False, gnu_timeout: bool = False
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    names = ["python3", "uv", "curl", "git", "sleep"] + (
        ["runpodctl"] if runpodctl else []
    )
    for name in names:
        stub = bin_dir / name
        stub.write_text(STUB, encoding="utf-8")
        stub.chmod(0o755)
    real_timeout = _gnu_timeout()
    if gnu_timeout and real_timeout is not None:
        (bin_dir / "timeout").symlink_to(real_timeout)
    return bin_dir


def _run(
    tmp_path: Path,
    *,
    pod_id: str = "pod-xyz",
    fail_step: str = "",
    runpodctl: bool = False,
    gnu_timeout: bool = False,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, list[str], str]:
    bin_dir = _stub_bin(tmp_path, runpodctl=runpodctl, gnu_timeout=gnu_timeout)
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
        "SWEEP_MAX_HOURS": "1",
        **(extra_env or {}),
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
    # `sleep` calls are the watchdog's polls and the delete backoff, not steps; the backoff is
    # checked through its "retrying in N s" log lines instead.
    lines = [c for c in lines if not c.startswith("sleep ")]
    log = (state / "bootstrap.log").read_text(encoding="utf-8")
    return result.returncode, lines, log + result.stdout + result.stderr


def test_success_runs_steps_in_order_then_terminates_pod(tmp_path: Path) -> None:
    code, calls, _ = _run(tmp_path)

    assert code == 0
    assert calls[0] == "python3 -m pip install uv==0.13.7"
    assert calls[1] == "uv sync --frozen"
    assert calls[2].startswith("uv run python -c ")  # the CUDA gate (I1)
    assert calls[3:5] == ["uv run pytest -m gpu -q", "uv run sweep-run"]
    assert calls[5].startswith("curl -fsS -X DELETE")
    assert calls[5].endswith(DELETE_URL)
    assert len(calls) == 6


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
    done_marker_for: str | None = None,
) -> tuple[int, list[str]]:
    """The launcher's pod command, with `/workspace` swapped for a temp dir (once, checked)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if done_marker_for is not None:
        (tmp_path / "state").mkdir()
        (tmp_path / "state" / f"DONE-{done_marker_for}").write_text(
            "", encoding="utf-8"
        )
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
        "SWEEP_STATE_DIR": str(tmp_path / "state"),
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


def _curl_headers(tmp_path: Path) -> list[str]:
    return (tmp_path / "calls.log.curl-stdin").read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize(
    "fail_step", ["", "pytest", "sweep-run"], ids=["success", "gpu-tests", "sweep"]
)
def test_done_marker_for_this_pod_is_written_on_any_exit(
    tmp_path: Path, fail_step: str
) -> None:
    _run(tmp_path, fail_step=fail_step)

    assert (tmp_path / "state" / "DONE-pod-xyz").exists()


def test_delete_falls_back_to_runpods_injected_key(tmp_path: Path) -> None:
    code, calls, _ = _run(tmp_path, extra_env={"CURL_FAIL_MATCH": FAKE_RUNPOD_KEY})

    curl_calls = [c for c in calls if c.startswith("curl")]
    assert code == 0
    assert _curl_headers(tmp_path) == [
        f"Authorization: Bearer {FAKE_RUNPOD_KEY}",
        f"Authorization: Bearer {INJECTED_KEY}",
    ]
    assert len(curl_calls) == 2
    assert all(FAKE_RUNPOD_KEY not in c and INJECTED_KEY not in c for c in calls)


def test_injected_key_fallback_is_skipped_when_it_equals_ours(tmp_path: Path) -> None:
    _run(
        tmp_path,
        runpodctl=True,
        extra_env={"CURL_FAIL_MATCH": "Bearer", "RUNPOD_API_KEY": FAKE_RUNPOD_KEY},
    )

    assert _curl_headers(tmp_path) == [f"Authorization: Bearer {FAKE_RUNPOD_KEY}"]


def test_delete_falls_back_to_runpodctl_when_both_keys_fail(tmp_path: Path) -> None:
    code, calls, _ = _run(
        tmp_path, runpodctl=True, extra_env={"CURL_FAIL_MATCH": "Bearer"}
    )

    assert code == 0
    assert calls[-1] == "runpodctl pod delete pod-xyz"
    assert len(_curl_headers(tmp_path)) == 2


def test_all_deletes_failing_retries_with_backoff_until_the_deadline_then_exits(
    tmp_path: Path,
) -> None:
    code, _, output = _run(
        tmp_path,
        fail_step="sweep-run",
        extra_env={
            "CURL_FAIL_MATCH": "Bearer",
            "SWEEP_MAX_HOURS": "0.001",
            "SWEEP_DEADLINE_RESERVE_S": "1",
        },
    )

    waits = [int(w) for w in re.findall(r"retrying in (\d+) s", output)]
    assert code == 7
    assert len(waits) >= 1
    assert all(
        1 <= w <= 3 for w in waits
    )  # waits never run past the 3.6 s pod deadline
    assert len(_curl_headers(tmp_path)) > 2
    assert "giving up" in output
    assert FAKE_RUNPOD_KEY not in output


def test_zero_deadline_means_a_single_round_of_attempts(tmp_path: Path) -> None:
    _, _, output = _run(
        tmp_path, extra_env={"CURL_FAIL_MATCH": "Bearer", "SWEEP_MAX_HOURS": "0"}
    )

    assert "retrying in" not in output
    assert len(_curl_headers(tmp_path)) == 2
    assert "giving up" in output


def test_one_liner_with_this_pods_done_marker_self_deletes_without_cloning(
    tmp_path: Path,
) -> None:
    code, calls = _run_one_liner(tmp_path, done_marker_for="pod-xyz")

    assert code == 0
    assert not any(c.startswith("git") for c in calls)
    assert calls[-1].endswith(DELETE_URL)


def test_one_liner_ignores_another_pods_done_marker(tmp_path: Path) -> None:
    code, calls = _run_one_liner(tmp_path, done_marker_for="pod-older")

    assert code == 0
    assert calls[0].startswith("git clone")
    assert calls[-1] == "bootstrap ran"


# -- fix round 2: SWEEP_MAX_HOURS is the pod's hard wall-clock bound ---------------------


@pytest.mark.parametrize(
    "killer",
    [
        "watchdog",
        pytest.param(
            "gnu-timeout",
            marks=pytest.mark.skipif(
                _gnu_timeout() is None,
                reason="no GNU coreutils timeout on this machine",
            ),
        ),
    ],
)
def test_bootstrap_kills_sweep_run_at_deadline_and_self_deletes(
    tmp_path: Path, killer: str
) -> None:
    started = time.monotonic()

    code, calls, output = _run(
        tmp_path,
        gnu_timeout=killer == "gnu-timeout",
        extra_env={
            "SLOW_STEP": "sweep-run",
            "SWEEP_MAX_HOURS": "0.001",  # 3.6 s for the whole pod
            "SWEEP_DEADLINE_RESERVE_S": "1",  # steps must end 2.6 s after start
        },
    )

    elapsed = time.monotonic() - started
    assert code == 124
    assert "uv run sweep-run" in calls
    assert calls[-1].endswith(DELETE_URL)
    assert elapsed < 20  # the 30 s step was cut off, not waited for
    assert "deadline" in output


def test_bootstrap_refuses_to_start_when_deadline_already_passed(
    tmp_path: Path,
) -> None:
    code, calls, output = _run(
        tmp_path,
        extra_env={"SWEEP_MAX_HOURS": "0.0001", "SWEEP_DEADLINE_RESERVE_S": "1"},
    )

    assert code == 1
    assert not any(c.startswith(("python3", "uv")) for c in calls)
    assert calls[-1].endswith(DELETE_URL)
    assert "deadline" in output


@pytest.mark.parametrize(
    "hours", ["", "abc", "-1"], ids=["missing", "text", "negative"]
)
def test_bootstrap_refuses_to_start_without_a_valid_max_hours(
    tmp_path: Path, hours: str
) -> None:
    code, calls, output = _run(tmp_path, extra_env={"SWEEP_MAX_HOURS": hours})

    assert code == 1
    assert not any(c.startswith(("python3", "uv")) for c in calls)
    assert "SWEEP_MAX_HOURS" in output


def test_deadline_survives_container_restart(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    first_start = int(time.time()) - 100  # this pod first started 100 s ago
    (state / "START-pod-xyz").write_text(f"{first_start}\n", encoding="utf-8")

    code, calls, _ = _run(
        tmp_path,
        extra_env={"SWEEP_MAX_HOURS": "0.01", "SWEEP_DEADLINE_RESERVE_S": "1"},  # 36 s
    )

    assert code == 1
    assert not any(c.startswith(("python3", "uv")) for c in calls)
    assert (state / "START-pod-xyz").read_text(encoding="utf-8") == f"{first_start}\n"


def test_first_start_records_the_start_time(tmp_path: Path) -> None:
    before = int(time.time())

    _run(tmp_path)

    recorded = int((tmp_path / "state" / "START-pod-xyz").read_text(encoding="utf-8"))
    assert before <= recorded <= int(time.time())


def test_corrupt_start_file_refuses_to_start(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "START-pod-xyz").write_text("yesterday\n", encoding="utf-8")

    code, calls, _ = _run(tmp_path)

    assert code == 1
    assert not any(c.startswith(("python3", "uv")) for c in calls)
    assert calls[-1].endswith(DELETE_URL)


def test_unwritable_state_dir_still_self_deletes(tmp_path: Path) -> None:
    blocker = tmp_path / "state-is-a-file"
    blocker.write_text("x", encoding="utf-8")
    bin_dir = _stub_bin(tmp_path)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CALLS_LOG": str(tmp_path / "calls.log"),
        "RUNPOD_POD_ID": "pod-xyz",
        "SWEEP_RUNPOD_API_KEY": FAKE_RUNPOD_KEY,
        "SWEEP_STATE_DIR": str(blocker / "sweep"),
        "SWEEP_MAX_HOURS": "1",
    }

    result = subprocess.run(
        [BASH, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    calls = (tmp_path / "calls.log").read_text(encoding="utf-8").splitlines()
    assert result.returncode != 0
    assert calls[-1].endswith(DELETE_URL)


# --- CUDA gate (final-review I1, G5) -------------------------------------------------------------


def _cuda_gate() -> str:
    """The `python -c` program the bootstrap runs before any GPU work."""
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"^readonly CUDA_GATE='([^']+)'$", text, flags=re.MULTILINE)
    assert match, "CUDA_GATE not found in pod_bootstrap.sh"
    return match.group(1)


def test_failed_cuda_gate_stops_before_gpu_tests_and_sweep_and_terminates_pod(
    tmp_path: Path,
) -> None:
    code, calls, output = _run(tmp_path, fail_step="run python -c")

    assert code == 7
    assert "uv run pytest -m gpu -q" not in calls
    assert "uv run sweep-run" not in calls
    assert calls[-1].endswith(DELETE_URL)
    assert "python -c" in output


def test_cuda_gate_program_checks_torch_and_warp() -> None:
    gate = _cuda_gate()

    assert "torch.cuda.is_available()" in gate
    assert "warp.is_cuda_available()" in gate


@pytest.mark.skipif(
    __import__("torch").cuda.is_available(), reason="checks the refusal on a machine without CUDA"
)
def test_cuda_gate_program_fails_without_cuda() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _cuda_gate()],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode != 0
    assert "CUDA" in result.stdout + result.stderr


@pytest.mark.gpu
def test_g5_cuda_gate_program_passes_on_the_pod() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _cuda_gate()],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr

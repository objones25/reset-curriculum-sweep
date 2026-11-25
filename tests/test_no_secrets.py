import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = (
    re.compile(r"(RUNPOD_API_KEY|WANDB_API_KEY)\s*=\s*\S{8,}"),
    re.compile(r"rpa_[A-Za-z0-9]{20,}"),
)


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout.decode()
    return [REPO_ROOT / name for name in out.split("\0") if name]


def _hits(text: str) -> list[str]:
    return [m.group(0) for pattern in SECRET_PATTERNS for m in pattern.finditer(text)]


def test_patterns_flag_key_assignments_and_runpod_tokens() -> None:
    fake_assignment = "RUNPOD_API_KEY" + " = " + "x" * 12
    fake_token = "rpa_" + "A" * 24

    assert _hits(fake_assignment) == [fake_assignment]
    assert _hits(fake_token) == [fake_token]
    assert _hits("WANDB_API_KEY=") == []


def test_no_tracked_file_contains_secret_like_strings() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}: pattern match"
        for path in _tracked_files()
        if path.is_file() and _hits(path.read_bytes().decode("utf-8", errors="ignore"))
    ]

    assert offenders == []


def test_env_file_is_not_tracked() -> None:
    tracked = {path.relative_to(REPO_ROOT).as_posix() for path in _tracked_files()}

    assert ".env" not in tracked

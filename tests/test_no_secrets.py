import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# `[ \t]*`, not `\s*`: an empty `KEY=` line must not match the next line of a file or a diff.
SECRET_PATTERNS = (
    re.compile(r"(RUNPOD_API_KEY|WANDB_API_KEY)[ \t]*=[ \t]*\S{8,}"),
    re.compile(r"rpa_[A-Za-z0-9]{20,}"),
    # W&B API keys: 40 hex characters assigned to a WANDB-ish name (`=` or `:`, maybe quoted).
    re.compile(r"(?i)wandb\w*[ \t]*[:=][ \t]*['\"]?[0-9a-f]{40}(?![0-9a-f])"),
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


def test_patterns_flag_a_wandb_key_assignment() -> None:
    # Final-review minor: W&B API keys are 40 hex characters.
    key = "0123456789abcdef" * 2 + "01234567"
    for line in (
        "WANDB_API_KEY" + "=" + key,
        "wandb_key" + ": '" + key + "'",
        "export WANDB_KEY" + "=" + '"' + key + '"',
    ):
        assert _hits(line), line
    assert _hits("wandb_run_hash" + ": " + key[:39]) == []  # 39 hex: not a W&B key
    assert _hits("WANDB_MODE" + "=" + "disabled") == []


def test_git_history_contains_no_secret_like_strings() -> None:
    # Final-review minor: a key committed once stays in the history of a public repo.
    history = subprocess.run(
        ["git", "log", "-p", "--all", "--no-color", "--no-ext-diff"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        timeout=120,
    ).stdout.decode("utf-8", errors="ignore")

    assert history  # the scan saw commits
    assert _hits(history) == []

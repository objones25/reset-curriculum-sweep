"""`.env` loading (Review Focus 4): a missing or blank key is named, no value is ever shown."""

from pathlib import Path

import pytest
from runpod_fakes import FAKE_RUNPOD_KEY, FAKE_WANDB_KEY, env_line

from resetsweep.launch.env_file import MissingSecret, Secrets, load_secrets


def test_load_secrets_reads_both_keys(env_file: Path) -> None:
    secrets = load_secrets(env_file)

    assert secrets == Secrets(
        runpod_api_key=FAKE_RUNPOD_KEY, wandb_api_key=FAKE_WANDB_KEY
    )


@pytest.mark.parametrize("name", ["RUNPOD_API_KEY", "WANDB_API_KEY"])
@pytest.mark.parametrize("value", [None, "", "   "], ids=["absent", "empty", "blank"])
def test_missing_or_blank_key_raises_missing_secret_naming_key(
    tmp_path: Path, name: str, value: str | None
) -> None:
    other = {"RUNPOD_API_KEY": FAKE_RUNPOD_KEY, "WANDB_API_KEY": FAKE_WANDB_KEY}
    lines = [env_line(k, v) for k, v in other.items() if k != name]
    if value is not None:  # audit: ignore[BRANCHING] absent means no line at all
        lines.append(env_line(name, value))
    path = tmp_path / ".env"
    path.write_text("".join(lines), encoding="utf-8")

    with pytest.raises(MissingSecret, match=f"^{name} ") as info:
        load_secrets(path)

    assert info.value.name == name
    assert FAKE_RUNPOD_KEY not in str(info.value)
    assert FAKE_WANDB_KEY not in str(info.value)


def test_key_without_equals_sign_counts_as_missing(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        env_line("RUNPOD_API_KEY", FAKE_RUNPOD_KEY) + "WANDB_API_KEY\n",
        encoding="utf-8",
    )

    with pytest.raises(MissingSecret, match="^WANDB_API_KEY "):
        load_secrets(path)


def test_missing_env_file_raises_file_not_found_naming_path(tmp_path: Path) -> None:
    path = tmp_path / "nope.env"

    with pytest.raises(FileNotFoundError, match="nope.env"):
        load_secrets(path)


def test_dollar_in_value_is_kept_verbatim(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        env_line("RUNPOD_API_KEY", "rk$HOME-x")
        + env_line("WANDB_API_KEY", FAKE_WANDB_KEY),
        encoding="utf-8",
    )

    assert load_secrets(path).runpod_api_key == "rk$HOME-x"


def test_secrets_repr_hides_values() -> None:
    secrets = Secrets(runpod_api_key=FAKE_RUNPOD_KEY, wandb_api_key=FAKE_WANDB_KEY)

    assert repr(secrets) == "Secrets(***)"
    assert str(secrets) == "Secrets(***)"
    assert FAKE_RUNPOD_KEY not in f"{[secrets]}"

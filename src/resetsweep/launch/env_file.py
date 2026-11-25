"""Read the two pod secrets from `.env` without ever showing a value (Review Focus 4).

`dotenv_values(path, interpolate=False)` (python-dotenv 1.2.4, installed source
`dotenv/main.py`) returns a dict whose value is None for a key written without `=`.
Interpolation is off so a `$` inside a key is kept verbatim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import dotenv_values

SECRET_NAMES: Final = ("RUNPOD_API_KEY", "WANDB_API_KEY")


class MissingSecret(Exception):
    """A required key is absent from `.env` or holds only whitespace. Names the key only."""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is missing or blank in the .env file")
        self.name = name


@dataclass(frozen=True)
class Secrets:
    runpod_api_key: str
    wandb_api_key: str

    def __repr__(self) -> str:
        return "Secrets(***)"

    __str__ = __repr__


def load_secrets(path: Path) -> Secrets:
    """The secrets in `path`. Raises FileNotFoundError or MissingSecret; never echoes a value."""
    if not path.is_file():
        raise FileNotFoundError(f"no .env file at {path}")
    values = dotenv_values(path, interpolate=False)
    found: dict[str, str] = {}
    for name in SECRET_NAMES:
        value = values.get(name)
        if value is None or not value.strip():
            raise MissingSecret(name)
        found[name] = value.strip()
    return Secrets(
        runpod_api_key=found["RUNPOD_API_KEY"], wandb_api_key=found["WANDB_API_KEY"]
    )

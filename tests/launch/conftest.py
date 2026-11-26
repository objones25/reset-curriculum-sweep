from pathlib import Path

import pytest
from runpod_fakes import FAKE_RUNPOD_KEY, FAKE_WANDB_KEY, FakeRunPod, env_line


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        env_line("RUNPOD_API_KEY", FAKE_RUNPOD_KEY)
        + env_line("WANDB_API_KEY", FAKE_WANDB_KEY),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fake_runpod() -> FakeRunPod:
    return FakeRunPod()

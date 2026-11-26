import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    torch.manual_seed(0)

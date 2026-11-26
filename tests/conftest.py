import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    random.seed(0)
    np.random.seed(0)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests marked `gpu` when CUDA is unavailable."""
    import torch

    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(
        reason="needs CUDA (torch.cuda.is_available() is False)"
    )
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)

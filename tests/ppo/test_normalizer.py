import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from resetsweep.ppo.normalizer import RunningNorm

OBS_DIM = 3


@settings(deadline=None, max_examples=50)
@given(
    sizes=st.lists(st.integers(min_value=1, max_value=40), min_size=1, max_size=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_running_norm_matches_numpy_mean_var_over_batches(
    sizes: list[int], seed: int
) -> None:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=3.0, scale=2.0, size=(sum(sizes), OBS_DIM))
    norm = RunningNorm((OBS_DIM,))

    for chunk in np.split(data, np.cumsum(sizes)[:-1]):
        norm.update(torch.from_numpy(chunk).float())

    np.testing.assert_allclose(
        norm.mean.numpy(), data.mean(axis=0), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(norm.var.numpy(), data.var(axis=0), rtol=1e-3, atol=1e-4)


def test_running_norm_normalize_gives_zero_mean_unit_var() -> None:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1000, OBS_DIM, generator=gen) * 5.0 + 2.0
    norm = RunningNorm((OBS_DIM,))
    norm.update(x)

    z = norm.normalize(x)

    assert z.mean(dim=0).tolist() == pytest.approx([0.0] * OBS_DIM, abs=1e-5)
    assert z.var(dim=0, unbiased=False).tolist() == pytest.approx(
        [1.0] * OBS_DIM, abs=1e-4
    )


def test_running_norm_rejects_wrong_feature_shape() -> None:
    norm = RunningNorm((OBS_DIM,))

    with pytest.raises(ValueError, match="expected trailing shape"):
        norm.update(torch.zeros(4, OBS_DIM + 1))

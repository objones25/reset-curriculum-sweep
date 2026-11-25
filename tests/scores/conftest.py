from collections.abc import Callable, Sequence

import pytest
import torch

from resetsweep.types import Outcomes, RolloutStats

MakeOutcomes = Callable[..., Outcomes]


@pytest.fixture
def make_outcomes() -> MakeOutcomes:
    """Build `Outcomes` from per-row lists; rows are `(ic_id, success, ret)`."""

    def build(
        rows: Sequence[tuple[int, bool, float]], *, full: Sequence[bool] | None = None
    ) -> Outcomes:
        n = len(rows)
        full_flags = [True] * n if full is None else list(full)
        return Outcomes(
            env_ids=torch.arange(n),
            ic_ids=torch.tensor([r[0] for r in rows], dtype=torch.int64),
            ret=torch.tensor([r[2] for r in rows], dtype=torch.float32),
            success=torch.tensor([r[1] for r in rows], dtype=torch.bool),
            ep_len=torch.full((n,), 10),
            reached_reward=torch.zeros(n, dtype=torch.bool),
            full=torch.tensor(full_flags, dtype=torch.bool),
            start_step=torch.zeros(n, dtype=torch.int64),
        )

    return build


StatsFactory = Callable[..., RolloutStats]


@pytest.fixture
def make_stats() -> StatsFactory:
    """Build `RolloutStats` from nested lists shaped `(T, N)`; `start_obs` is `(K, D)`."""

    def build(
        *,
        ic_ids: Sequence[Sequence[int]],
        adv: Sequence[Sequence[float]] | None = None,
        returns: Sequence[Sequence[float]] | None = None,
        values: Sequence[Sequence[float]] | None = None,
        start_obs: Sequence[Sequence[float]] | None = None,
        critic: Callable[[torch.Tensor], torch.Tensor] | None = None,
        start_features: Sequence[Sequence[float]] | torch.Tensor | None = None,
    ) -> RolloutStats:
        ids = torch.tensor(ic_ids, dtype=torch.int64)
        zeros = [[0.0] * ids.shape[1] for _ in range(ids.shape[0])]
        return RolloutStats(
            ic_ids=ids,
            adv=torch.tensor(adv or zeros, dtype=torch.float32),
            returns=torch.tensor(returns or zeros, dtype=torch.float32),
            values=torch.tensor(values or zeros, dtype=torch.float32),
            start_obs=torch.tensor(start_obs or [[0.0]], dtype=torch.float32),
            critic=critic or (lambda obs: obs[:, 0]),
            start_features=torch.as_tensor(
                [[0.0]] if start_features is None else start_features,
                dtype=torch.float32,
            ),
        )

    return build

from collections.abc import Callable, Sequence

import pytest
import torch

from resetsweep.types import Outcomes

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

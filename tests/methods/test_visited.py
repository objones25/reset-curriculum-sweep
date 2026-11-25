from collections.abc import Callable

import pytest
import torch

from resetsweep.methods.base import MethodContext
from resetsweep.methods.visited import Visited
from resetsweep.tasks.base import TASKS

MakeCtx = Callable[..., MethodContext]
# chi-square 0.99 quantile, 9 degrees of freedom (standard table value)
CHI2_CRIT_DF9_P01 = 21.666


def _offer(method: Visited, qpos_values: list[float]) -> None:
    n = len(qpos_values)
    qpos = torch.tensor(qpos_values).unsqueeze(1)
    method.offer_visited(
        torch.arange(n),
        qpos,
        torch.zeros(n, 1),
        torch.zeros(n, 0),
        torch.zeros(n, dtype=torch.long),
    )


def test_visited_ring_buffer_overwrites_oldest(make_ctx: MakeCtx) -> None:
    method = Visited(make_ctx(num_envs=2), capacity=4)

    _offer(method, [0.1, 0.2])
    _offer(method, [0.3, 0.4])
    _offer(method, [0.5, 0.6])

    stored = sorted(method.buffer().qpos[:, 0].tolist())
    assert stored == pytest.approx([0.3, 0.4, 0.5, 0.6])
    assert method.diagnostics()["buffer_size"] == 4.0


def test_visited_stores_clones_not_views(make_ctx: MakeCtx) -> None:
    method = Visited(make_ctx(num_envs=2), capacity=4)
    qpos = torch.tensor([[0.1], [0.2]])

    method.offer_visited(
        torch.arange(2),
        qpos,
        torch.zeros(2, 1),
        torch.zeros(2, 0),
        torch.zeros(2, dtype=torch.long),
    )
    qpos.fill_(3.0)

    assert sorted(method.buffer().qpos[:, 0].tolist()) == pytest.approx([0.1, 0.2])


def test_visited_skips_states_the_task_would_reject(make_ctx: MakeCtx) -> None:
    method = Visited(make_ctx(num_envs=3), capacity=8)

    # pendulum start_box allows |qpos| <= pi; 50.0 is outside, nan is non-finite
    _offer(method, [0.5, 50.0, float("nan")])

    assert method.buffer().qpos[:, 0].tolist() == pytest.approx([0.5])


def test_visited_falls_back_until_min_fill(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(num_envs=4, p_task=0.0)
    method = Visited(ctx, capacity=16)
    _offer(method, [2.0, 2.0, 2.0])  # 3 < min_fill = num_envs = 4

    before, before_ids = method.reset(torch.arange(50), None)
    _offer(method, [2.0])
    after, _ = method.reset(torch.arange(50), None)

    assert bool((before_ids == -1).all())
    assert bool((before.qpos.abs() <= 0.1).all())  # pendulum task start noise
    assert bool((after.qpos == 2.0).all())


def test_visited_draw_uniform_over_buffer(make_ctx: MakeCtx) -> None:
    torch.manual_seed(0)
    ctx = make_ctx(num_envs=10, p_task=0.0)
    method = Visited(ctx, capacity=10)
    _offer(method, [i / 10 for i in range(10)])
    draws = 20_000

    states, _ = method.reset(torch.arange(draws), None)

    cells = torch.round(states.qpos[:, 0] * 10).long()
    counts = torch.bincount(cells, minlength=10).float()
    expected = draws / 10
    chi2 = ((counts - expected) ** 2 / expected).sum().item()
    assert chi2 < CHI2_CRIT_DF9_P01


def test_visited_p_task_mixes_task_starts(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(num_envs=4, p_task=0.5)
    method = Visited(ctx, capacity=8)
    _offer(method, [2.0] * 4)

    states, _ = method.reset(torch.arange(4000), None)

    from_task = states.qpos[:, 0].abs() <= 0.1  # buffered states sit at 2.0
    assert from_task.float().mean().item() == pytest.approx(0.5, abs=0.03)


def test_visited_round_trips_task_extra_with_its_state(make_ctx: MakeCtx) -> None:
    task = TASKS.construct("maze_coinflip")
    method = Visited(make_ctx(task=task, num_envs=3, p_task=0.0), capacity=3)
    base = task.canonical_start()
    shifts = torch.tensor([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]])
    extra = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    method.offer_visited(
        torch.arange(3),
        base.qpos + shifts,
        base.qvel.expand(3, -1).clone(),
        extra,
        torch.zeros(3, dtype=torch.long),
    )

    states, _ = method.reset(torch.arange(300), None)

    expected = extra[
        torch.round(states.qpos[:, 0:1].sub(base.qpos[0, 0]) * 100).long()[:, 0]
    ]
    assert torch.equal(states.extra, expected)
    assert torch.unique(states.extra, dim=0).shape[0] == 3


def test_visited_entropy_is_log_buffer_size_when_p_task_zero(make_ctx: MakeCtx) -> None:
    import math

    method = Visited(make_ctx(num_envs=4, p_task=0.0), capacity=8)
    _offer(method, [0.1, 0.2, 0.3, 0.4])

    diag = method.diagnostics()

    assert diag["sampling_entropy"] == pytest.approx(math.log(4))


@pytest.mark.parametrize(
    ("capacity", "min_fill", "num_envs"), [(0, None, 4), (4, 5, 4), (2, None, 4)]
)
def test_visited_rejects_bad_sizes(
    make_ctx: MakeCtx, capacity: int, min_fill: int | None, num_envs: int
) -> None:
    with pytest.raises(ValueError, match="capacity|min_fill"):
        Visited(make_ctx(num_envs=num_envs), capacity=capacity, min_fill=min_fill)

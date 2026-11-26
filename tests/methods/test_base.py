from collections.abc import Callable

import pytest
import torch

from resetsweep.methods.base import METHODS, MethodContext, mix_task_start
from resetsweep.methods.point import Point
from resetsweep.methods.uniform import Uniform
from resetsweep.types import StartStates

MakeCtx = Callable[..., MethodContext]
N_RESETS = 10_000
SHARE_TOLERANCE = 0.02
REST_IC_ID = 7


def _rest_draw(ctx: MethodContext) -> Callable[[int], tuple[StartStates, torch.Tensor]]:
    def draw(n: int) -> tuple[StartStates, torch.Tensor]:
        states = StartStates(
            torch.full((n, 1), 1.5), torch.zeros(n, 1), torch.zeros(n, 0)
        )
        return states, torch.full((n,), REST_IC_ID, dtype=torch.long)

    return draw


def test_mix_task_start_share_close_to_p_task(make_ctx: MakeCtx) -> None:
    ctx = make_ctx(p_task=0.3)

    _, ic_ids = mix_task_start(ctx, torch.arange(N_RESETS), _rest_draw(ctx))

    share = (ic_ids == -1).float().mean().item()
    assert share == pytest.approx(0.3, abs=SHARE_TOLERANCE)


def test_mix_task_start_keeps_row_order_and_pairs_states_with_ids(
    make_ctx: MakeCtx,
) -> None:
    ctx = make_ctx(p_task=0.5)

    states, ic_ids = mix_task_start(ctx, torch.arange(200), _rest_draw(ctx))

    from_rest = states.qpos[:, 0] == 1.5
    assert states.n == 200
    assert torch.equal(from_rest, ic_ids == REST_IC_ID)
    assert bool((ic_ids[~from_rest] == -1).all())


@pytest.mark.parametrize(("p_task", "expected_share"), [(0.0, 0.0), (1.0, 1.0)])
def test_p_task_zero_and_one_extremes(
    make_ctx: MakeCtx, p_task: float, expected_share: float
) -> None:
    ctx = make_ctx(p_task=p_task)

    _, ic_ids = mix_task_start(ctx, torch.arange(1000), _rest_draw(ctx))

    assert (ic_ids == -1).float().mean().item() == expected_share


@pytest.mark.parametrize("p_task", [-0.1, 1.1])
def test_context_rejects_p_task_outside_unit_interval(
    make_ctx: MakeCtx, p_task: float
) -> None:
    with pytest.raises(ValueError, match="p_task"):
        make_ctx(p_task=p_task)


def test_methods_register_under_their_names() -> None:
    METHODS.load("resetsweep.methods")

    assert {"uniform", "point", "visited"} <= set(METHODS.names())
    assert METHODS.get("uniform") is Uniform


def test_uniform_resets_from_task_sampler_with_task_start_ids(
    make_ctx: MakeCtx,
) -> None:
    ctx = make_ctx()
    method = Uniform(ctx)

    states, ic_ids = method.reset(torch.arange(5), None)

    assert states.n == 5
    assert ic_ids.tolist() == [-1] * 5
    assert bool(ctx.task.is_valid(states).all())


def test_point_always_same_state(make_ctx: MakeCtx) -> None:
    method = Point(make_ctx(), source="random")

    first, first_ids = method.reset(torch.arange(4), None)
    second, second_ids = method.reset(torch.arange(3), None)

    assert torch.equal(first.qpos[0], second.qpos[0])
    assert bool((first.qpos == first.qpos[0]).all())
    assert bool((second.qvel == first.qvel[0]).all())
    assert torch.equal(first_ids, torch.zeros(4, dtype=torch.long))
    assert torch.equal(second_ids, torch.zeros(3, dtype=torch.long))


def test_point_canonical_is_the_task_canonical_start(make_ctx: MakeCtx) -> None:
    ctx = make_ctx()
    method = Point(ctx)

    states, _ = method.reset(torch.arange(2), None)

    canonical = ctx.task.canonical_start()
    assert torch.equal(states.qpos[1], canonical.qpos[0])


def test_point_rejects_unknown_source(make_ctx: MakeCtx) -> None:
    with pytest.raises(ValueError, match="source"):
        Point(make_ctx(), source="nowhere")  # pyright: ignore[reportArgumentType] # deliberately invalid


@pytest.mark.parametrize("cls", [Uniform, Point])
def test_diagnostics_have_required_keys(make_ctx: MakeCtx, cls: type) -> None:
    method = cls(make_ctx())

    diag = method.diagnostics()

    assert {"sampling_entropy", "buffer_size", "never_succeeded_share"} <= set(diag)
    assert diag["sampling_entropy"] == 0.0
    assert diag["buffer_size"] == 0.0

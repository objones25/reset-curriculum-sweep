import math
from collections.abc import Callable

import pytest
import torch

from resetsweep.scores import SCORES
from resetsweep.scores.critic import CriticLoss, CriticValue
from resetsweep.types import RolloutStats

MakeStats = Callable[..., RolloutStats]


def _ids(*ids: int) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int64)


# T = 2 vector steps, N = 3 envs. The (0, 2) cell has ic_id -1 (task start) and must be
# ignored; its adv of 9 would show up in IC 2's numbers if it leaked in.
#   IC 0: rows (adv 1, ret 2, v 1) and (adv -3, ret 6, v 3)
#   IC 1: rows (adv -2, ret 1, v 0) and (adv 4, ret 3, v 2)
#   IC 2: row  (adv 5, ret 7, v 1)
#   IC 3: never seen -> untrusted
_IC = [[0, 1, -1], [0, 1, 2]]
_ADV = [[1.0, -2.0, 9.0], [-3.0, 4.0, 5.0]]
_RET = [[2.0, 1.0, 9.0], [6.0, 3.0, 7.0]]
_VAL = [[1.0, 0.0, 9.0], [3.0, 2.0, 1.0]]


def _rollout(make_stats: MakeStats) -> RolloutStats:
    return make_stats(ic_ids=_IC, adv=_ADV, returns=_RET, values=_VAL)


def test_pos_value_loss_matches_manual_scatter_mean(make_stats: MakeStats) -> None:
    score = CriticLoss(capacity=4, device="cpu", form="pos_value_loss")
    score.observe_rollout(_rollout(make_stats))
    lp, trusted = score.values(_ids(0, 1, 2, 3))
    # mean(relu(adv)): IC0 (1 + 0)/2 = 0.5; IC1 (0 + 4)/2 = 2; IC2 5.
    # IC3 untrusted -> median of [0.5, 2, 5] = 2.
    assert lp.tolist() == [0.5, 2.0, 5.0, 2.0]
    assert trusted.tolist() == [True, True, True, False]


def test_max_mc_matches_manual(make_stats: MakeStats) -> None:
    score = CriticLoss(capacity=4, device="cpu", form="max_mc")
    score.observe_rollout(_rollout(make_stats))
    lp, _ = score.values(_ids(0, 1, 2, 3))
    # max(returns) - mean(values): IC0 6 - 2 = 4; IC1 3 - 1 = 2; IC2 7 - 1 = 6.
    # IC3 untrusted -> median of [4, 2, 6] = 4.
    assert lp.tolist() == [4.0, 2.0, 6.0, 4.0]


def test_max_mc_keeps_running_rmax_across_rollouts(make_stats: MakeStats) -> None:
    score = CriticLoss(capacity=4, device="cpu", form="max_mc")
    score.observe_rollout(_rollout(make_stats))
    # Second rollout: IC 0 only, lower return 1, value 1. Rmax stays 6 -> 6 - 1 = 5.
    score.observe_rollout(make_stats(ic_ids=[[0]], returns=[[1.0]], values=[[1.0]]))
    lp, _ = score.values(_ids(0, 1, 2))
    assert lp.tolist() == [5.0, 2.0, 6.0]
    # A higher return raises Rmax: 9 - 1 = 8.
    score.observe_rollout(make_stats(ic_ids=[[0]], returns=[[9.0]], values=[[1.0]]))
    lp, _ = score.values(_ids(0))
    assert lp.tolist() == [8.0]


def test_max_mc_forget_clears_rmax(make_stats: MakeStats) -> None:
    score = CriticLoss(capacity=4, device="cpu", form="max_mc")
    score.observe_rollout(_rollout(make_stats))
    score.forget(_ids(0))
    score.observe_rollout(make_stats(ic_ids=[[0]], returns=[[1.0]], values=[[1.0]]))
    lp, _ = score.values(_ids(0))
    # Rmax restarted at 1 (not 6): 1 - 1 = 0.
    assert lp.tolist() == [0.0]


def test_abs_gae_matches_manual(make_stats: MakeStats) -> None:
    score = CriticLoss(capacity=4, device="cpu", form="abs_gae")
    score.observe_rollout(_rollout(make_stats))
    lp, _ = score.values(_ids(0, 1, 2, 3))
    # mean(|adv|): IC0 (1 + 3)/2 = 2; IC1 (2 + 4)/2 = 3; IC2 5. IC3 -> median 3.
    assert lp.tolist() == [2.0, 3.0, 5.0, 3.0]


def test_later_rollout_replaces_score_of_seen_ics_only(make_stats: MakeStats) -> None:
    score = CriticLoss(capacity=4, device="cpu", form="abs_gae")
    score.observe_rollout(_rollout(make_stats))
    score.observe_rollout(make_stats(ic_ids=[[1]], adv=[[-7.0]]))
    lp, _ = score.values(_ids(0, 1, 2))
    assert lp.tolist() == [2.0, 7.0, 5.0]


def test_forget_makes_ic_untrusted(make_stats: MakeStats) -> None:
    score = CriticLoss(capacity=4, device="cpu", form="abs_gae")
    score.observe_rollout(_rollout(make_stats))
    score.forget(_ids(1))
    _, trusted = score.values(_ids(0, 1, 2))
    assert trusted.tolist() == [True, False, True]


def test_critic_loss_rejects_id_out_of_range(make_stats: MakeStats) -> None:
    score = CriticLoss(capacity=2, device="cpu", form="abs_gae")
    with pytest.raises(ValueError, match="ic_ids"):
        score.observe_rollout(make_stats(ic_ids=[[0, 2]]))


def test_critic_loss_all_task_start_rollout_is_noop(make_stats: MakeStats) -> None:
    score = CriticLoss(capacity=2, device="cpu", form="abs_gae")
    score.observe_rollout(make_stats(ic_ids=[[-1, -1]], adv=[[3.0, 4.0]]))
    _, trusted = score.values(_ids(0, 1))
    assert trusted.tolist() == [False, False]


@pytest.mark.parametrize(
    ("settings", "message"),
    [({"form": "nope"}, "form"), ({"beta": 0.0}, "beta")],
)
def test_critic_loss_rejects_invalid_settings(
    settings: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CriticLoss(capacity=2, device="cpu", **{"form": "abs_gae", **settings})  # type: ignore[arg-type]  # deliberate bad input


def test_critic_value_peaks_at_half_vmax(make_stats: MakeStats) -> None:
    score = CriticValue(capacity=4, device="cpu", vmax=1.0)
    # start_obs row i is IC i's start observation; the stand-in critic returns column 0.
    stats = make_stats(ic_ids=[[0, 1, 2, 3]], start_obs=[[0.0], [0.25], [0.5], [1.0]])
    score.observe_rollout(stats)
    lp, trusted = score.values(_ids(0, 1, 2, 3))
    # V(vmax - V): 0, 0.25 * 0.75, 0.5 * 0.5, 0.
    assert lp.tolist() == pytest.approx([0.0, 0.1875, 0.25, 0.0])
    assert trusted.tolist() == [True] * 4
    assert int(torch.argmax(lp)) == 2


def test_critic_value_evaluates_only_rollout_rows(make_stats: MakeStats) -> None:
    seen_rows: list[int] = []

    def critic(obs: torch.Tensor) -> torch.Tensor:
        seen_rows.append(obs.shape[0])
        return obs[:, 0]

    score = CriticValue(capacity=6, device="cpu", vmax=1.0)
    # T*N = 4 rows, with IC 0 twice and one task-start (-1) cell; capacity is 6.
    stats = make_stats(
        ic_ids=[[0, 1], [0, -1]],
        start_obs=[[0.5], [0.25], [0.9], [0.9], [0.9], [0.9]],
        critic=critic,
    )
    score.observe_rollout(stats)
    assert seen_rows == [4]
    lp, trusted = score.values(_ids(0, 1, 2))
    # Same as evaluating every row: V(1 - V) = 0.25 and 0.1875; IC 2 untouched.
    assert lp[:2].tolist() == pytest.approx([0.25, 0.1875])
    assert trusted.tolist() == [True, True, False]


def test_critic_value_only_updates_ics_in_rollout(make_stats: MakeStats) -> None:
    score = CriticValue(capacity=3, device="cpu", vmax=1.0)
    score.observe_rollout(make_stats(ic_ids=[[0, 1]], start_obs=[[0.5], [0.5], [0.5]]))
    # IC 2 is not in the rollout, so its start_obs row is not trusted.
    lp, trusted = score.values(_ids(0, 1, 2))
    assert trusted.tolist() == [True, True, False]
    assert lp.tolist() == pytest.approx([0.25, 0.25, 0.25])


def test_critic_value_probs_are_softmax_of_normalised_lp_over_temperature(
    make_stats: MakeStats,
) -> None:
    # Paper: P(c) ~ exp(beta * V/Vmax * (Vmax - V)); here beta = 1 / temperature.
    score = CriticValue(capacity=2, device="cpu", vmax=2.0, temperature=0.5)
    score.observe_rollout(make_stats(ic_ids=[[0, 1]], start_obs=[[1.0], [0.5]]))
    # V = 1: 1*(2-1)/2 = 0.5; V = 0.5: 0.5*1.5/2 = 0.375; logits = those / 0.5.
    logits = [0.5 / 0.5, 0.375 / 0.5]
    denom = sum(math.exp(x) for x in logits)
    probs = score.probs(_ids(0, 1))
    assert probs.tolist() == pytest.approx([math.exp(x) / denom for x in logits])


def test_critic_value_rejects_wrong_start_obs_rows(make_stats: MakeStats) -> None:
    score = CriticValue(capacity=3, device="cpu", vmax=1.0)
    with pytest.raises(ValueError, match="start_obs"):
        score.observe_rollout(make_stats(ic_ids=[[0]], start_obs=[[0.5], [0.5]]))


@pytest.mark.parametrize(
    ("settings", "message"),
    [({"vmax": 0.0}, "vmax"), ({"temperature": 0.0}, "temperature")],
)
def test_critic_value_rejects_invalid_settings(
    settings: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CriticValue(capacity=2, device="cpu", **{"vmax": 1.0, **settings})  # type: ignore[arg-type]  # deliberate bad input


def test_registry_constructs_critic_scores() -> None:
    assert isinstance(
        SCORES.construct("critic_loss", capacity=2, device="cpu", form="max_mc"),
        CriticLoss,
    )
    assert isinstance(
        SCORES.construct("critic_value", capacity=2, device="cpu", vmax=1.0),
        CriticValue,
    )

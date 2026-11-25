import pytest
import torch

from resetsweep.ppo.gae import compute_gae

GAMMA = 0.5
LAM = 0.5


def test_gae_matches_hand_computed_three_step_case() -> None:
    # One env, three steps, no episode end, gamma = lam = 0.5 (so gamma * lam = 0.25).
    # bootstrap_values[t] = V(s_{t+1}), the critic on the pre-reset next observation.
    #   delta_0 = 1 + 0.5 * 1.0 - 0.5 = 1.0
    #   delta_1 = 2 + 0.5 * 1.5 - 1.0 = 1.75
    #   delta_2 = 3 + 0.5 * 2.0 - 1.5 = 2.5
    #   A_2 = 2.5
    #   A_1 = 1.75 + 0.25 * 2.5   = 2.375
    #   A_0 = 1.0  + 0.25 * 2.375 = 1.59375
    #   returns = A + V = 2.09375, 3.375, 4.0
    # Cross-checked 2026-11-24 against robot-learning-expert scripts/rl_math.py `gae(...)`,
    # which printed the same advantages and returns; frozen here after they agreed.
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [1.0], [1.5]])
    bootstrap = torch.tensor([[1.0], [1.5], [2.0]])
    flags = torch.zeros(3, 1, dtype=torch.bool)

    adv, returns = compute_gae(
        rewards, values, flags, flags, bootstrap, gamma=GAMMA, lam=LAM
    )

    assert adv.squeeze(1).tolist() == pytest.approx([1.59375, 2.375, 2.5])
    assert returns.squeeze(1).tolist() == pytest.approx([2.09375, 3.375, 4.0])


def test_gae_bootstraps_on_truncation_not_termination() -> None:
    # Two envs with identical rewards and values. At t = 1 env 0 is truncated and env 1 is
    # terminated; both have bootstrap value 10 there. Same gamma = lam = 0.5.
    #   env 0 (truncated): delta_1 = 2 + 0.5 * 10 - 1.0 = 6.0; A_1 = 6.0 (no carry across the end)
    #                      A_0 = 1.0 + 0.25 * 6.0 = 2.5
    #   env 1 (terminated): delta_1 = 2 - 1.0 = 1.0;  A_1 = 1.0
    #                      A_0 = 1.0 + 0.25 * 1.0 = 1.25
    #   both, t = 2: delta_2 = 3 + 0.5 * 2.0 - 1.5 = 2.5
    # Cross-checked 2026-11-24 against robot-learning-expert scripts/rl_math.py `gae(...)`
    # (advantages [[2.5, 1.25], [6.0, 1.0], [2.5, 2.5]], returns [[3, 1.75], [7, 2], [4, 4]]).
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    values = torch.tensor([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
    bootstrap = torch.tensor([[1.0, 1.0], [10.0, 10.0], [2.0, 2.0]])
    terminated = torch.tensor([[False, False], [False, True], [False, False]])
    truncated = torch.tensor([[False, False], [True, False], [False, False]])

    adv, returns = compute_gae(
        rewards, values, terminated, truncated, bootstrap, gamma=GAMMA, lam=LAM
    )

    assert adv.tolist() == [
        pytest.approx([2.5, 1.25]),
        pytest.approx([6.0, 1.0]),
        pytest.approx([2.5, 2.5]),
    ]
    assert returns.tolist() == [
        pytest.approx([3.0, 1.75]),
        pytest.approx([7.0, 2.0]),
        pytest.approx([4.0, 4.0]),
    ]


def test_gae_rejects_mismatched_shapes() -> None:
    good = torch.zeros(3, 2)
    flags = torch.zeros(3, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match=r"shape \(T, N\)"):
        compute_gae(good, torch.zeros(3, 1), flags, flags, good, gamma=GAMMA, lam=LAM)

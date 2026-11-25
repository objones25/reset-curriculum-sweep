import time

import pytest
import torch
from torch import Tensor

from resetsweep.ppo.gae import compute_gae
from resetsweep.ppo.networks import ActorCritic
from resetsweep.ppo.normalizer import RunningNorm
from resetsweep.ppo.update import START_CONFIG, Batch, PPOConfig, ppo_update
from resetsweep.sim.env import VecEnv
from resetsweep.tasks.base import TASKS, Task
from resetsweep.types import Outcomes, StartStates, StepCounter

TASKS.load("resetsweep.tasks")

OBS_DIM = 3
ACT_DIM = 2


def _config(**overrides: object) -> PPOConfig:
    base: dict[str, object] = {
        "gamma": 0.99,
        "lam": 0.95,
        "epochs": 4,
        "minibatch_size": 32,
        "lr": 1e-3,
        "ent_coef": 0.0,
        "vf_coef": 1.0,
        "target_kl": None,
    }
    base.update(overrides)
    return PPOConfig(**base)  # pyright: ignore[reportArgumentType] # test-only kwargs bag


def _fixed_batch(model: ActorCritic, *, t: int, n: int, seed: int) -> Batch:
    gen = torch.Generator().manual_seed(seed)
    obs = torch.randn(t, n, OBS_DIM, generator=gen)
    action, logp, value = model.act(obs)
    # Advantage tied to the action sign so there is a consistent direction to improve.
    advantages = action[..., 0].sign() + 0.1 * torch.randn(t, n, generator=gen)
    returns = obs.sum(dim=-1)
    return Batch(
        obs=obs,
        actions=action,
        logp=logp,
        values=value,
        advantages=advantages,
        returns=returns,
    )


def _losses(model: ActorCritic, batch: Batch) -> tuple[float, float]:
    """(value MSE, clipped-free surrogate -mean(ratio * adv)) on the whole batch."""
    with torch.no_grad():
        logp, _, value = model.evaluate(batch.obs, batch.actions)
        ratio = (logp - batch.logp).exp()
        adv = (batch.advantages - batch.advantages.mean()) / batch.advantages.std()
        return (
            float((value - batch.returns).square().mean()),
            float(-(ratio * adv).mean()),
        )


def test_ppo_update_reduces_loss_on_fixed_batch() -> None:
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(32, 32))
    batch = _fixed_batch(model, t=8, n=16, seed=1)
    cfg = _config()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    value_before, surrogate_before = _losses(model, batch)

    stats = ppo_update(model, opt, batch, cfg)
    value_after, surrogate_after = _losses(model, batch)

    assert value_after < 0.8 * value_before
    assert surrogate_after < surrogate_before - 0.01
    assert set(stats) == {
        "approx_kl",
        "clip_frac",
        "explained_var",
        "entropy",
        "value_loss",
        "policy_loss",
    }


@pytest.mark.parametrize(
    ("t", "n", "minibatch_size", "epochs", "expected_steps"),
    [
        (4, 8, 8, 1, 4),  # 32 // 8
        (4, 8, 5, 1, 6),  # 32 // 5, remainder dropped
        (4, 8, 32, 3, 3),  # one minibatch per epoch
        (8, 16, 32, 2, 8),  # 128 // 32 = 4, times 2 epochs
    ],
)
def test_minibatch_count_derived_from_batch(
    t: int, n: int, minibatch_size: int, epochs: int, expected_steps: int
) -> None:
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(16,))
    batch = _fixed_batch(model, t=t, n=n, seed=2)
    cfg = _config(minibatch_size=minibatch_size, epochs=epochs)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    steps: list[int] = []
    opt.register_step_post_hook(lambda *_: steps.append(1))

    ppo_update(model, opt, batch, cfg)

    assert len(steps) == expected_steps


def test_ppo_update_rejects_minibatch_larger_than_batch() -> None:
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(16,))
    batch = _fixed_batch(model, t=2, n=4, seed=3)
    cfg = _config(minibatch_size=9)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    with pytest.raises(ValueError, match="minibatch_size 9 exceeds the batch of 8"):
        ppo_update(model, opt, batch, cfg)


def test_target_kl_stops_update_early() -> None:
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(32, 32))
    batch = _fixed_batch(model, t=8, n=16, seed=4)
    # A huge lr makes the first minibatch move the policy far past the KL target.
    cfg = _config(lr=0.5, epochs=4, minibatch_size=32, target_kl=1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    steps: list[int] = []
    opt.register_step_post_hook(lambda *_: steps.append(1))

    ppo_update(model, opt, batch, cfg)

    assert 1 <= len(steps) < cfg.epochs * (8 * 16 // 32)


def test_act_log_prob_is_of_the_unclipped_sample() -> None:
    model = ActorCritic(OBS_DIM, ACT_DIM, hidden=(16,))
    obs = torch.randn(64, OBS_DIM)

    action, logp, _ = model.act(obs)
    logp_eval, _, _ = model.evaluate(obs, action)

    assert bool(
        (action.abs() > 1.0).any()
    )  # std starts at 1, so some samples exceed [-1, 1]
    torch.testing.assert_close(logp, logp_eval)


# --- learning test --------------------------------------------------------------------------


class _TaskStarts:
    """Minimal reset method: every episode starts from the task's own start distribution."""

    name = "task_start"

    def __init__(self, task: Task, gen: torch.Generator) -> None:
        self._task, self._gen = task, gen

    def reset(
        self, env_ids: Tensor, outcomes: Outcomes | None
    ) -> tuple[StartStates, Tensor]:
        return self._task.sample_start(env_ids.numel(), self._gen), env_ids.clone()

    def offer_visited(
        self,
        env_ids: Tensor,
        qpos: Tensor,
        qvel: Tensor,
        extra: Tensor,
        ep_step: Tensor,
    ) -> None:
        return None


NUM_ENVS = 256
ROLLOUT_STEPS = 32
WALL_CLOCK_S_MAX = 60.0
TRAIN_WALL_CLOCK_S = 48.0  # leaves room for env build and the baseline rollouts
BASELINE_ROLLOUTS = (
    6  # untrained policy; the first 2 are dropped (all envs start hanging)
)
# Threshold, measured 2026-11-24 on an Apple-silicon Mac CPU, seeds 0, 1, 2, with this exact
# setup (256 envs x 32 steps, START_CONFIG with a 2048 minibatch). The metric is the mean
# per-step dense reward of a rollout on pendulum_dense (height01 - 0.01 u^2: 0 hanging, 1 upright).
# - Untrained policy, steady state (rollouts 3-16): 0.017-0.022, 0.024-0.035, 0.006-0.010.
# - After 51 s of training (about 145 iterations), mean of the last 3 rollouts: 0.64, 0.63, 0.58.
#   Halfway (about iteration 70) the three seeds were already near 0.35-0.45.
# Requiring a gain over the measured baseline of more than 0.3 is about half of every seed's
# measured gain (0.55-0.62), so a machine twice as slow (half the iterations) still passes,
# while it is roughly 10x the untrained policy's rollout-to-rollout spread (about 0.03),
# so a policy that did not learn cannot pass.
IMPROVEMENT_MIN = 0.3


@pytest.mark.slow
def test_ppo_learns_pendulum_dense_on_cpu() -> None:
    start = time.monotonic()
    torch.manual_seed(0)
    task: Task = TASKS.construct("pendulum_dense")  # pyright: ignore[reportAssignmentType] # registry is untyped
    gen = torch.Generator().manual_seed(0)
    env = VecEnv(
        task,
        num_envs=NUM_ENVS,
        device="cpu",
        counter=StepCounter(budget=10**9),
        gen=gen,
    )
    env.bind(_TaskStarts(task, gen))
    model = ActorCritic(task.obs_dim, task.act_dim)
    norm = RunningNorm((task.obs_dim,))
    # START_CONFIG except the minibatch: 24,576 is larger than this CPU batch of 256 x 32 = 8192,
    # so the test holds 4 minibatches per epoch instead (2048 samples each).
    cfg = PPOConfig(**{**START_CONFIG.__dict__, "minibatch_size": 2048})
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    obs = env.reset_all()
    untrained: list[float] = []
    for _ in range(BASELINE_ROLLOUTS):
        _, obs, mean_reward = _rollout(env, model, norm, obs, cfg)
        untrained.append(mean_reward)
    baseline = sum(untrained[2:]) / len(untrained[2:])

    curve = _train(env, model, norm, opt, obs, cfg, wall_clock_s=TRAIN_WALL_CLOCK_S)

    final = sum(curve[-3:]) / 3
    assert final - baseline > IMPROVEMENT_MIN, (baseline, final, curve)
    assert time.monotonic() - start < WALL_CLOCK_S_MAX


def _train(
    env: VecEnv,
    model: ActorCritic,
    norm: RunningNorm,
    opt: torch.optim.Optimizer,
    obs: Tensor,
    cfg: PPOConfig,
    *,
    wall_clock_s: float,
) -> list[float]:
    """PPO until the wall-clock budget is spent; mean per-step reward of each rollout."""
    start = time.monotonic()
    curve: list[float] = []
    for _ in range(10_000):  # bound; the wall clock ends the loop long before this
        if time.monotonic() - start > wall_clock_s:
            break
        batch, obs, mean_reward = _rollout(env, model, norm, obs, cfg)
        curve.append(mean_reward)
        ppo_update(model, opt, batch, cfg)
    return curve


def _rollout(
    env: VecEnv, model: ActorCritic, norm: RunningNorm, obs: Tensor, cfg: PPOConfig
) -> tuple[Batch, Tensor, float]:
    n, t = env.num_envs, ROLLOUT_STEPS
    obs_buf = torch.zeros(t, n, env.task.obs_dim)
    act_buf = torch.zeros(t, n, env.task.act_dim)
    logp_buf, val_buf, rew_buf, boot_buf = (torch.zeros(t, n) for _ in range(4))
    term_buf = torch.zeros(t, n, dtype=torch.bool)
    trunc_buf = torch.zeros(t, n, dtype=torch.bool)
    raw_obs: list[Tensor] = []
    for step in range(t):
        raw_obs.append(obs)
        obs_n = norm.normalize(obs)
        action, logp, value = model.act(obs_n)
        out = env.step(action)
        with torch.no_grad():
            boot_buf[step] = model.value(norm.normalize(out.final_obs))
        obs_buf[step], act_buf[step] = obs_n, action
        logp_buf[step], val_buf[step], rew_buf[step] = logp, value, out.reward
        term_buf[step], trunc_buf[step] = out.terminated, out.truncated
        obs = out.obs
    norm.update(torch.cat(raw_obs))
    adv, returns = compute_gae(
        rew_buf, val_buf, term_buf, trunc_buf, boot_buf, gamma=cfg.gamma, lam=cfg.lam
    )
    batch = Batch(
        obs=obs_buf,
        actions=act_buf,
        logp=logp_buf,
        values=val_buf,
        advantages=adv,
        returns=returns,
    )
    return batch, obs, float(rew_buf.mean())

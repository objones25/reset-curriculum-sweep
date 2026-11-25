"""The PPO update: clipped surrogate, value regression, entropy bonus, KL early stop."""

from dataclasses import dataclass

import torch
from torch import Tensor

from resetsweep.ppo.networks import ActorCritic


@dataclass(frozen=True, kw_only=True)
class PPOConfig:
    """PPO hyperparameters. Starting values (`START_CONFIG`) and where each comes from:

    - gamma 0.99, lam 0.95, clip 0.2, epochs 5, ent_coef 0.01, vf_coef 1.0, lr 1e-3:
      the Isaac Lab / mjlab rsl_rl velocity defaults, massively-parallel-rl
      `references/pytorch-gpu-sim.md` section 2 "Velocity-task defaults" table (rows
      "gamma / lambda / clip", "epochs / minibatches", "learning rate", "entropy": mjlab 0.01
      velocity); the same values are listed for the mjlab G1 config in robot-learning-expert
      `references/concepts/policy-gradient.md` line 78.
    - minibatch_size 24,576 = 4096 envs x 24 steps / 4 minibatches, the shipped rsl_rl minibatch
      (massively-parallel-rl `references/large-batch-on-policy.md` section 3, "rsl_rl's adaptive
      KL": "4096 envs x 24 steps / 4 minibatches = 24,576"). It is held fixed and the minibatch
      count is derived as N*T // minibatch_size (same file, section 2, step 1, Beukman et al. 2026;
      massively-parallel-rl non-negotiable 2).
    - target_kl 0.01: desired_kl of the recipe (robot-learning-expert
      `references/training/on-policy-at-scale.md` "The recipe", step 5). rsl_rl uses it to adapt
      the lr; here the lr stays fixed (spec section 5.5) and target_kl instead stops the update
      early once a minibatch's approximate KL to the rollout policy passes it.
    - max_grad_norm 0.5, adam_eps 1e-5: spec section 5.5 and massively-parallel-rl SKILL.md
      "Defaults to reach for", row "Optimizer" (Adam, eps 1e-5, global-norm clip 0.5 for PPO).
      rsl_rl ships 1.0 for the clip; the spec's 0.5 wins.

    These are starting points, tuned once per task family on its dense sibling (spec 5.5).
    """

    gamma: float
    lam: float
    clip: float = 0.2
    epochs: int
    minibatch_size: int
    lr: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float = 0.5
    adam_eps: float = 1e-5
    target_kl: float | None

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError(
                f"epochs and minibatch_size must be positive, got {self.epochs}, {self.minibatch_size}"
            )
        if not (0.0 <= self.gamma <= 1.0 and 0.0 <= self.lam <= 1.0):
            raise ValueError(
                f"gamma and lam must be in [0, 1], got {self.gamma}, {self.lam}"
            )
        if self.lr <= 0 or self.clip <= 0 or self.max_grad_norm <= 0:
            raise ValueError("lr, clip and max_grad_norm must be positive")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError(
                f"target_kl must be positive or None, got {self.target_kl}"
            )


START_CONFIG = PPOConfig(
    gamma=0.99,
    lam=0.95,
    epochs=5,
    minibatch_size=24_576,
    lr=1e-3,
    ent_coef=0.01,
    vf_coef=1.0,
    target_kl=0.01,
)


@dataclass(frozen=True)
class Batch:
    """One rollout. `obs` `(T, N, obs_dim)` (already normalised), `actions` `(T, N, act_dim)`
    (unclipped samples), the rest `(T, N)`."""

    obs: Tensor
    actions: Tensor
    logp: Tensor
    values: Tensor
    advantages: Tensor
    returns: Tensor


def _explained_variance(values: Tensor, returns: Tensor) -> float:
    """1 - Var(R - V) / Var(R); NaN when the returns are constant (undefined)."""
    var_r = returns.var(unbiased=False)
    return float(1.0 - (returns - values).var(unbiased=False) / var_r)


def ppo_update(
    model: ActorCritic, opt: torch.optim.Optimizer, batch: Batch, cfg: PPOConfig
) -> dict[str, float]:
    """`cfg.epochs` passes over the batch in `N*T // minibatch_size` shuffled minibatches.

    Advantages are normalised once over the whole batch. Samples left over after the last
    full minibatch are skipped in that epoch (a fresh permutation each epoch). With `target_kl`
    set, the update stops before the first minibatch whose k3 approximate KL to the rollout
    policy exceeds it; that check reads one scalar per minibatch, not per env step.
    Returns minibatch means of the losses, KL, clip fraction and entropy, plus the explained
    variance of the rollout values. The optimizer's lr is left as the caller set it.
    """
    t, n = batch.advantages.shape
    total = t * n
    if cfg.minibatch_size > total:
        raise ValueError(
            f"minibatch_size {cfg.minibatch_size} exceeds the batch of {total}"
        )
    obs = batch.obs.reshape(total, -1)
    actions = batch.actions.reshape(total, -1)
    old_logp = batch.logp.reshape(total)
    returns = batch.returns.reshape(total)
    adv = batch.advantages.reshape(total)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    num_minibatches = total // cfg.minibatch_size

    sums = torch.zeros(5, device=adv.device)  # policy, value, entropy, kl, clip_frac
    steps = 0
    stop = False
    for _ in range(cfg.epochs):
        perm = torch.randperm(total, device=adv.device)
        for k in range(num_minibatches):
            idx = perm[k * cfg.minibatch_size : (k + 1) * cfg.minibatch_size]
            logp, entropy, value = model.evaluate(obs[idx], actions[idx])
            log_ratio = logp - old_logp[idx]
            ratio = log_ratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
            if cfg.target_kl is not None and float(approx_kl) > cfg.target_kl:
                stop = True
                break
            a = adv[idx]
            policy_loss = -torch.min(
                ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a
            ).mean()
            value_loss = 0.5 * (value - returns[idx]).square().mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_mean
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                clip_frac = ((ratio - 1).abs() > cfg.clip).float().mean()
                sums += torch.stack(
                    [policy_loss, value_loss, entropy_mean, approx_kl, clip_frac]
                ).detach()
            steps += 1
        if stop:
            break

    means = (sums / max(steps, 1)).tolist()
    return {
        "policy_loss": means[0],
        "value_loss": means[1],
        "entropy": means[2],
        "approx_kl": means[3],
        "clip_frac": means[4],
        "explained_var": _explained_variance(batch.values, batch.returns),
    }

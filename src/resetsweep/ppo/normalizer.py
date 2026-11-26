"""Running observation mean and variance, merged batch by batch (Chan et al. parallel update)."""

import torch
from torch import Tensor, nn

_EPS = 1e-8


class RunningNorm(nn.Module):
    """Population mean and variance of every sample passed to `update`, per feature.

    Buffers, so `.to(device)` and `state_dict()` carry the statistics. `count` stays a tensor
    so an update never syncs with the host. Before any update, `normalize` is the identity
    (mean 0, var 1).
    """

    mean: Tensor
    var: Tensor
    count: Tensor

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = tuple(shape)
        self.register_buffer("mean", torch.zeros(self.shape))
        self.register_buffer("var", torch.ones(self.shape))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    def update(self, x: Tensor) -> None:
        """Merge a batch `(B, *shape)` into the statistics."""
        if tuple(x.shape[1:]) != self.shape or x.shape[0] == 0:
            raise ValueError(
                f"expected trailing shape {self.shape} and a non-empty batch, got {tuple(x.shape)}"
            )
        x = x.detach().to(self.mean.dtype)
        b = x.shape[0]
        b_mean = x.mean(dim=0)
        b_var = x.var(dim=0, unbiased=False)
        n = self.count.to(self.mean.dtype)
        tot = n + b
        delta = b_mean - self.mean
        # Chan et al.: M2 = M2_a + M2_b + delta^2 * n_a * n_b / n; with n_a = 0 this is the batch's own.
        m2 = self.var * n + b_var * b + delta.square() * n * b / tot
        self.mean.add_(delta * b / tot)
        self.var.copy_(m2 / tot)
        self.count.add_(b)

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / torch.sqrt(self.var + _EPS)

"""Interpenetration check shared by the contact tasks (`arm`, `claw`)."""

import torch
import warp as wp
from torch import Tensor

from resetsweep.sim.batch import BatchSim
from resetsweep.types import StartStates

_CHUNK = 128


class InterpenetrationCheck:
    """Rejects start poses whose geoms overlap by more than `tolerance_m`.

    Contacts are soft, so a body pressed against another sags a little into it; that is a
    reachable state and must pass. Only overlap deeper than `tolerance_m` is a real
    interpenetration. The check builds one CPU `BatchSim` of `_CHUNK` worlds on first use.
    """

    def __init__(
        self,
        mjcf: str,
        *,
        frame_skip: int,
        canonical: StartStates,
        tolerance_m: float,
    ) -> None:
        if tolerance_m <= 0:
            raise ValueError(f"tolerance_m must be positive, got {tolerance_m}")
        self._mjcf = mjcf
        self._frame_skip = frame_skip
        self._canonical = canonical
        self._tolerance_m = tolerance_m
        self._sim: BatchSim | None = None

    def ok(self, qpos: Tensor, checkable: Tensor) -> Tensor:
        """True per row unless a geom overlaps another by more than the tolerance.

        Runs `mjw.forward` (inside `BatchSim.set_state`), then reads the contact list. Margins are
        zero, so a contact exists only when `dist < 0`. Rows where `checkable` is False are
        replaced by the canonical start so a NaN or an absurd pose never reaches the solver; the
        caller masks them out anyway. `qpos` must be on the CPU.
        """
        if self._sim is None:
            self._sim = BatchSim(
                self._mjcf, nworld=_CHUNK, device="cpu", frame_skip=self._frame_skip
            )
        sim = self._sim
        rest_q, rest_v = self._canonical.qpos, self._canonical.qvel
        safe = torch.where(checkable.unsqueeze(1), qpos, rest_q)
        ok = torch.ones(safe.shape[0], dtype=torch.bool)
        for lo in range(0, safe.shape[0], _CHUNK):
            chunk = safe[lo : lo + _CHUNK]
            size = chunk.shape[0]
            padded = rest_q.repeat(_CHUNK, 1)
            padded[:size] = chunk
            sim.set_state(torch.arange(_CHUNK), padded, rest_v.repeat(_CHUNK, 1))
            deepest_ok = _deepest_penetration_ok(sim, _CHUNK, self._tolerance_m)
            ok[lo : lo + size] = deepest_ok[:size]
        return ok


def _deepest_penetration_ok(sim: BatchSim, nworld: int, tolerance_m: float) -> Tensor:
    nacon = int(sim.data.nacon.numpy()[0])
    if nacon > sim.data.naconmax:
        raise RuntimeError(f"contact buffer overflowed: {nacon} > {sim.data.naconmax}")
    # `data.contact.dist` is negative for penetration; entries past `nacon` are stale.
    dist = wp.to_torch(sim.data.contact.dist)[:nacon]
    world = wp.to_torch(sim.data.contact.worldid)[:nacon].long()
    deepest = torch.zeros(nworld).scatter_reduce(
        0, world, dist, reduce="amin", include_self=True
    )
    return deepest >= -tolerance_m

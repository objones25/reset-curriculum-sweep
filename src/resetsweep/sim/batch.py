"""Batched MuJoCo Warp simulator exposed through zero-copy torch views."""

from collections.abc import Callable

import mujoco
import mujoco_warp as mjw
import warp as wp
from torch import Tensor


class BatchSim:
    """`nworld` copies of one MJCF model stepped in lockstep on `device`."""

    def __init__(self, mjcf: str, *, nworld: int, device: str, frame_skip: int) -> None:
        if nworld <= 0 or frame_skip <= 0:
            raise ValueError(
                f"nworld and frame_skip must be positive, got {nworld}, {frame_skip}"
            )
        # Warp 1.19 has no `config.quiet`; log_level hides the per-module "load ... (cached)" lines.
        wp.config.log_level = wp.LOG_WARNING
        self._frame_skip = frame_skip
        self._device = device
        self._graph: wp.Graph | None = None
        self._substep_graph: wp.Graph | None = None
        self._force_applied = False
        mjm = mujoco.MjModel.from_xml_string(mjcf)  # pyright: ignore[reportAttributeAccessIssue] # mujoco ships no stub for the C-extension class
        with wp.ScopedDevice(device):
            self.model = mjw.put_model(mjm)
            self.data = mjw.make_data(mjm, nworld=nworld)
            self.qpos: Tensor = wp.to_torch(self.data.qpos)
            self.qvel: Tensor = wp.to_torch(self.data.qvel)
            self.ctrl: Tensor = wp.to_torch(self.data.ctrl)
            # Applied generalized force (nworld, nv): `mjw.forward` adds it to the smooth force
            # (mujoco_warp/_src/forward.py:3439) and `mjw.step` runs forward, so a value written
            # here before a substep acts during it.
            self.qfrc_applied: Tensor = wp.to_torch(self.data.qfrc_applied)
            self.xpos: Tensor = wp.to_torch(self.data.xpos)
            self.site_xpos: Tensor = wp.to_torch(self.data.site_xpos)
            self._warmstart: Tensor = wp.to_torch(self.data.qacc_warmstart)
            mjw.forward(self.model, self.data)
            if wp.get_device(device).is_cuda:
                # Capture only on CUDA. Run once uncaptured first so kernels are
                # compiled; capture then records launches without executing them.
                self._run_steps()
                with wp.ScopedCapture() as capture:
                    self._run_steps()
                self._graph = capture.graph
                # One-substep graph, for steps whose applied force is recomputed between substeps.
                with wp.ScopedCapture() as substep_capture:
                    mjw.step(self.model, self.data)
                self._substep_graph = substep_capture.graph
                self._reset_to_initial()

    def _reset_to_initial(self) -> None:
        """Restore the freshly-made-data state (also undoes the CUDA warm-up steps)."""
        with wp.ScopedDevice(self._device):
            mjw.reset_data(self.model, self.data)
            mjw.forward(self.model, self.data)

    def get_state(self, idx: Tensor) -> tuple[Tensor, Tensor]:
        return self.qpos[idx].clone(), self.qvel[idx].clone()

    def set_state(self, idx: Tensor, qpos: Tensor, qvel: Tensor) -> None:
        if not (qpos.shape[0] == qvel.shape[0] == idx.numel()):
            raise ValueError(
                f"row count mismatch: idx {idx.numel()}, qpos {qpos.shape[0]}, qvel {qvel.shape[0]}"
            )
        if qpos.device != self.qpos.device or qvel.device != self.qpos.device:
            raise ValueError(
                f"qpos/qvel on {qpos.device}/{qvel.device}, sim is on {self.qpos.device}"
            )
        self.qpos[idx] = qpos
        self.qvel[idx] = qvel
        self._warmstart[idx] = 0.0
        with wp.ScopedDevice(self._device):
            mjw.forward(self.model, self.data)

    def step(
        self,
        ctrl: Tensor,
        applied_force: Callable[["BatchSim"], Tensor | None] | None = None,
    ) -> None:
        """Advance `frame_skip` physics substeps with `ctrl` held.

        `applied_force`, if given, is called before every substep and returns the generalized
        force `(nworld, nv)` for it (or None for none), so forces that depend on the state
        (drag, a pit's pull) are evaluated at the substep's own state, not once per control step.
        """
        if ctrl.shape != self.ctrl.shape:
            raise ValueError(
                f"ctrl shape {tuple(ctrl.shape)}, expected {tuple(self.ctrl.shape)}"
            )
        self.ctrl.copy_(ctrl)
        with wp.ScopedDevice(self._device):
            if applied_force is None:
                if self._force_applied:
                    self.qfrc_applied.zero_()
                    self._force_applied = False
                if self._graph is not None:
                    wp.capture_launch(self._graph)
                else:
                    self._run_steps()
                return
            for _ in range(self._frame_skip):
                self._write_force(applied_force(self))
                self._run_substep()

    def _write_force(self, force: Tensor | None) -> None:
        if force is None:
            self.qfrc_applied.zero_()
            return
        if force.shape != self.qfrc_applied.shape:
            raise ValueError(
                f"applied force shape {tuple(force.shape)}, expected {tuple(self.qfrc_applied.shape)}"
            )
        self.qfrc_applied.copy_(force)
        self._force_applied = True

    def _run_substep(self) -> None:
        if self._substep_graph is not None:
            wp.capture_launch(self._substep_graph)
        else:
            mjw.step(self.model, self.data)

    def _run_steps(self) -> None:
        for _ in range(self._frame_skip):
            mjw.step(self.model, self.data)

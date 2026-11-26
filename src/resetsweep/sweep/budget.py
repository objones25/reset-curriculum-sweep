"""In-pod budget guard (spec section 8): spend = wall clock x price, with a ledger that
survives pod restarts (Review Focus 2).

- Spend is pod time, not job time: the pod is billed whether one worker or four are busy, so a
  job that gets killed still counted, through the wall clock, for every second it ran.
- The ledger (`ledger_path`, JSON lines ``{"t": session seconds, "spent": cumulative USD,
  "utc": ISO wall-clock time}``)
  holds the cumulative spend of every session in this state dir. A new session starts from the
  last line plus `LEDGER_GAP_S` worth of spend: the time between the last heartbeat before a
  crash and the crash itself is unrecorded, and that is at most one heartbeat interval plus
  the longest gap between heartbeat calls (`HEARTBEAT_S` 15 s + a 5 s poll, or one result
  report, < 60 s). It then writes a session-start line at once.
- `can_start` refuses a job once ``spent + est_job_s + in_flight_s`` (pod-seconds the sweep
  will still pay for, at the current price) would pass ``cap - margin``. The caller decides how
  jobs map to pod-seconds (the sweep loop: in_flight_s = the longest remaining in-flight wall
  clock, est_job_s = how far the new job reaches past it).
- Work deadline (final-review I2): with `stop_at` (a time on the guard's clock), the limit is
  ``min(cap - margin, spend at stop_at)``, so `can_start` refuses a job that would run past it
  and `remaining()` reaches 0 at `stop_at`: the sweep loop then kills every job and writes the
  report while the pod still has its report reserve.
- A torn last ledger line (a crash mid-append) is truncated with a warning; any earlier bad
  line, or a spend that goes down, raises `LedgerError`: the ledger is the record of money.

Constants (global constraints, spec section 8):
`NETWORK_VOLUME_USD_PER_GB_MONTH` 0.07 is from docs.runpod.io/storage/network-volumes, read
2026-11-24 (the API exposes no storage price).
"""

import json
import logging
import math
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

_log = logging.getLogger(__name__)

CAP_USD_DEFAULT: Final = 100.0
MARGIN_USD: Final = 5.0
# Final-review I6: a smoke run (a few dollars, `--cap 5`) keeps 0.50 for its report and
# self-delete; a full sweep keeps MARGIN_USD.
SMOKE_MARGIN_USD: Final = 0.5
CALIBRATION_CAP_USD: Final = 10.0
NETWORK_VOLUME_USD_PER_GB_MONTH: Final = 0.07
NETWORK_VOLUME_GB: Final = 20
HOURS_PER_MONTH: Final = 730.0
HEARTBEAT_S: Final = 15.0
LEDGER_GAP_S: Final = 60.0


class LedgerError(ValueError):
    """The spend ledger is unreadable or inconsistent."""


def margin_usd(*, smoke: bool) -> float:
    """The margin kept below the cap: `SMOKE_MARGIN_USD` for smoke runs, else `MARGIN_USD`."""
    return SMOKE_MARGIN_USD if smoke else MARGIN_USD


def volume_usd_per_h(gb: float) -> float:
    """Network volume price per hour: ``gb * 0.07 / 730``, assuming 730 h per month (the
    RunPod rate is per GB-month)."""
    if not (math.isfinite(gb) and gb >= 0):
        raise ValueError(f"gb must be >= 0, got {gb}")
    return gb * NETWORK_VOLUME_USD_PER_GB_MONTH / HOURS_PER_MONTH


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _check_price(name: str, value: float, *, allow_zero: bool) -> None:
    ok = math.isfinite(value) and (value >= 0 if allow_zero else value > 0)
    if not ok:
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {bound}, got {value}")


class BudgetGuard:
    """See the module docstring."""

    def __init__(
        self,
        *,
        cap_usd: float,
        margin_usd: float,
        gpu_usd_per_h: float,
        volume_usd_per_h: float,
        ledger_path: Path,
        clock: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], str] = _utc_now,
        stop_at: float | None = None,
    ) -> None:
        _check_price("cap_usd", cap_usd, allow_zero=False)
        _check_price("margin_usd", margin_usd, allow_zero=True)
        _check_price("gpu_usd_per_h", gpu_usd_per_h, allow_zero=False)
        _check_price("volume_usd_per_h", volume_usd_per_h, allow_zero=True)
        self.cap_usd = cap_usd
        self.margin_usd = margin_usd
        self.usd_per_s = (gpu_usd_per_h + volume_usd_per_h) / 3600.0
        self._path = ledger_path
        self._clock = clock
        self._utc_now = utc_now
        self._t0 = clock()
        if stop_at is not None and not math.isfinite(stop_at):
            raise ValueError(f"stop_at must be finite, got {stop_at}")
        self._stop_at = stop_at
        last = self._read_last_spent()
        self._prior = 0.0 if last is None else last + LEDGER_GAP_S * self.usd_per_s
        self._last_write = -math.inf
        self.flush()

    # --- public interface ----------------------------------------------------------------------

    def elapsed_s(self) -> float:
        """Seconds since this session's guard was created."""
        dt = self._clock() - self._t0
        if dt < 0:
            raise RuntimeError(f"clock went backwards by {-dt} s")
        return dt

    def spent(self) -> float:
        """USD spent in every session so far, this one included."""
        return self._prior + self.elapsed_s() * self.usd_per_s

    def limit_usd(self) -> float:
        """The spend that scheduling may not pass: cap - margin, or the spend at `stop_at` if
        that is lower."""
        limit = self.cap_usd - self.margin_usd
        if self._stop_at is not None:
            limit = min(
                limit, self._prior + (self._stop_at - self._t0) * self.usd_per_s
            )
        return limit

    def deadline_bound(self) -> bool:
        """True when `stop_at`, not cap - margin, sets the limit."""
        return self.limit_usd() < self.cap_usd - self.margin_usd

    def deadline_reached(self) -> bool:
        """True once the clock has reached `stop_at` (never without one)."""
        return self._stop_at is not None and self._clock() >= self._stop_at

    def remaining(self) -> float:
        """USD left before `limit_usd`; negative once passed."""
        return self.limit_usd() - self.spent()

    def can_start(self, est_job_s: float, in_flight_s: float) -> bool:
        """True if spent plus the new job's and the in-flight jobs' remaining pod-seconds stays
        within cap - margin."""
        for name, value in (("est_job_s", est_job_s), ("in_flight_s", in_flight_s)):
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be finite and >= 0, got {value}")
        projected = self.spent() + (est_job_s + in_flight_s) * self.usd_per_s
        return projected <= self.limit_usd()

    def heartbeat(self) -> None:
        """Append ``{t, spent}`` if `HEARTBEAT_S` passed since the last line. The caller polls
        at least every few seconds, so a line lands at least every 60 s."""
        if self.elapsed_s() - self._last_write >= HEARTBEAT_S:
            self.flush()

    def flush(self) -> None:
        """Append ``{t, spent, utc}`` now, fsynced."""
        t = self.elapsed_s()
        row = {"t": t, "spent": self.spent(), "utc": self._utc_now()}
        line = json.dumps(row, allow_nan=False)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._last_write = t

    # --- ledger --------------------------------------------------------------------------------

    def _read_last_spent(self) -> float | None:
        if not self._path.exists():
            return None
        lines = self._path.read_text(encoding="utf-8").splitlines(keepends=True)
        values: list[float] = []
        for n, line in enumerate(lines, start=1):
            try:
                spent = float(json.loads(line)["spent"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                if n == len(lines):
                    self._truncate(keep=lines[:-1], error=e)
                    break
                raise LedgerError(f"{self._path} line {n}: unreadable ({e})") from e
            if not (math.isfinite(spent) and spent >= 0):
                raise LedgerError(f"{self._path} line {n}: bad spend {spent}")
            if values and spent < values[-1]:
                raise LedgerError(
                    f"{self._path} line {n}: spend decreases from {values[-1]} to {spent}"
                )
            values.append(spent)
        return values[-1] if values else None

    def _truncate(self, *, keep: list[str], error: Exception) -> None:
        size = len("".join(keep).encode("utf-8"))
        _log.warning(
            "%s: last line is a torn append (%s); truncating to %d bytes",
            self._path,
            error,
            size,
        )
        with self._path.open("r+b") as f:
            f.truncate(size)
            f.flush()
            os.fsync(f.fileno())

"""Metric sinks for one training run: a protocol plus null, JSONL, fan-out and W&B sinks.

`log(step, data)` records a point of the run's time series, with `step` the run's total
env-step count; `summary(data)` records final per-run values; `close()` releases the sink.
`run()` never closes the sink it is given: its caller owns it.

W&B API checked against the installed wandb 0.31.2 source
(`.venv/lib/python3.12/site-packages/wandb/sdk/wandb_init.py` `init`,
`wandb_run.py` `Run.log`, `Run.define_metric`, `Run.summary`, `Run.finish`).
"""

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol

import wandb

# Config keys that look like credentials; secrets must never reach W&B config (global
# constraints: "never in W&B config").
_SECRET_KEY = re.compile(r"(?i)key|token|secret")
_STEP_KEY = "step"

WandbMode = Literal["online", "offline", "disabled"]


class MetricsSink(Protocol):
    def log(self, step: int, data: dict[str, float]) -> None: ...

    def summary(self, data: dict[str, float | str]) -> None: ...

    def close(self) -> None: ...


class NullSink:
    """Discards everything."""

    def log(self, step: int, data: dict[str, float]) -> None:
        del step, data

    def summary(self, data: dict[str, float | str]) -> None:
        del data

    def close(self) -> None:
        pass


def _finite_or_none(value: float | str) -> float | str | None:
    # JSON has no NaN or Infinity; null keeps every line parseable by strict readers.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class JsonlSink:
    """Appends one JSON object per call to `path`, flushed at once so a crash loses nothing.

    Log lines are `{"type": "log", "step": s, **data}`, summary lines
    `{"type": "summary", **data}`; NaN and infinities are written as null.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def log(self, step: int, data: dict[str, float]) -> None:
        self._write({"type": "log", _STEP_KEY: step, **data})

    def summary(self, data: dict[str, float | str]) -> None:
        self._write({"type": "summary", **data})

    def close(self) -> None:
        self._file.close()

    def _write(self, record: Mapping[str, float | str | int]) -> None:
        clean = {k: _finite_or_none(v) for k, v in record.items()}
        self._file.write(json.dumps(clean, allow_nan=False) + "\n")
        self._file.flush()


class MultiSink:
    """Forwards every call to each sink, in order."""

    def __init__(self, *sinks: MetricsSink) -> None:
        self._sinks = sinks

    def log(self, step: int, data: dict[str, float]) -> None:
        for sink in self._sinks:
            sink.log(step, data)

    def summary(self, data: dict[str, float | str]) -> None:
        for sink in self._sinks:
            sink.summary(data)

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()


def _secret_like_keys(config: Mapping[str, object], prefix: str = "") -> list[str]:
    found: list[str] = []
    for key, value in config.items():
        path = f"{prefix}{key}"
        if _SECRET_KEY.search(str(key)):
            found.append(path)
        if isinstance(value, Mapping):
            found.extend(_secret_like_keys(value, f"{path}."))  # pyright: ignore[reportUnknownArgumentType] # nested config values are arbitrary
    return found


class WandbSink:
    """One W&B run. Every logged metric is plotted against the logged `step` (the run's
    total env steps), declared with `define_metric` instead of passing `step=` to `log`, so
    W&B's own step counter never drops a point.

    `reinit="create_new"` lets one worker process hold several runs in turn without
    `wandb.init` returning the previous one. Raises ValueError, naming the key but never
    the value, when any config key (at any depth) matches `(?i)key|token|secret`.
    """

    def __init__(
        self,
        *,
        project: str,
        group: str,
        name: str,
        config: Mapping[str, object],
        mode: WandbMode,
    ) -> None:
        bad = _secret_like_keys(config)
        if bad:
            raise ValueError(
                f"W&B config has secret-like keys {bad}; secrets must never be logged"
            )
        self._run = wandb.init(
            project=project,
            group=group,
            name=name,
            config=dict(config),
            mode=mode,
            reinit="create_new",
        )
        self._run.define_metric(_STEP_KEY)
        self._run.define_metric("*", step_metric=_STEP_KEY)
        self.closed = False

    def log(self, step: int, data: dict[str, float]) -> None:
        self._run.log({**data, _STEP_KEY: step})

    def summary(self, data: dict[str, float | str]) -> None:
        self._run.summary.update(data)

    def close(self) -> None:
        if not self.closed:
            self._run.finish()
            self.closed = True

import json
import math
from pathlib import Path

import pytest

from resetsweep.metrics import JsonlSink, MultiSink, NullSink, WandbSink


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_jsonl_sink_writes_one_line_per_log(tmp_path: Path) -> None:
    path = tmp_path / "run" / "metrics.jsonl"
    sink = JsonlSink(path)

    sink.log(10, {"eval/success": 0.25})
    sink.log(20, {"eval/success": 0.5, "ppo/approx_kl": 0.01})

    # Read before close: every line must already be flushed.
    assert _lines(path) == [
        {"type": "log", "step": 10, "eval/success": 0.25},
        {"type": "log", "step": 20, "eval/success": 0.5, "ppo/approx_kl": 0.01},
    ]
    sink.close()


def test_jsonl_sink_appends_to_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    first = JsonlSink(path)
    first.log(1, {"a": 1.0})
    first.close()

    second = JsonlSink(path)
    second.log(2, {"a": 2.0})
    second.close()

    assert [line["step"] for line in _lines(path)] == [1, 2]


def test_jsonl_sink_writes_non_finite_values_as_null(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = JsonlSink(path)

    sink.log(5, {"ic/never_succeeded_share": math.nan, "x": math.inf})
    sink.close()

    assert _lines(path) == [
        {"type": "log", "step": 5, "ic/never_succeeded_share": None, "x": None}
    ]


def test_jsonl_sink_writes_summary_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = JsonlSink(path)

    sink.summary({"auc": 0.4, "status": "ok"})
    sink.close()

    assert _lines(path) == [{"type": "summary", "auc": 0.4, "status": "ok"}]


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def log(self, step: int, data: dict[str, float]) -> None:
        self.calls.append(("log", (step, data)))

    def summary(self, data: dict[str, float | str]) -> None:
        self.calls.append(("summary", data))

    def close(self) -> None:
        self.calls.append(("close", None))


def test_multi_sink_forwards_every_call_to_every_sink() -> None:
    a, b = _Recorder(), _Recorder()
    sink = MultiSink(a, NullSink(), b)

    sink.log(3, {"m": 1.0})
    sink.summary({"auc": 0.0})
    sink.close()

    expected = [("log", (3, {"m": 1.0})), ("summary", {"auc": 0.0}), ("close", None)]
    assert a.calls == expected
    assert b.calls == expected


@pytest.mark.parametrize(
    "config",
    [
        {"api_key": "x"},
        {"WANDB_TOKEN": "x"},
        {"Secret": 1},
        {"ppo": {"lr": 1e-3}, "method_kw": {"auth_token": "x"}},
    ],
)
def test_wandb_sink_rejects_secret_like_config_keys(config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"secret-like"):
        WandbSink(project="p", group="g", name="n", config=config, mode="disabled")


def test_wandb_sink_error_does_not_echo_the_value() -> None:
    with pytest.raises(ValueError, match=r"'api_key'") as info:
        WandbSink(
            project="p",
            group="g",
            name="n",
            config={"api_key": "hunter2"},
            mode="disabled",
        )

    assert "hunter2" not in str(info.value)


def test_wandb_sink_disabled_mode_logs_summarises_and_closes() -> None:
    sink = WandbSink(
        project="p",
        group="g",
        name="n",
        config={"task": "pendulum_sparse", "ppo": {"lr": 1e-3}},
        mode="disabled",
    )

    sink.log(100, {"eval/success": 0.5, "ic/never_succeeded_share": math.nan})
    sink.summary({"auc": 0.25, "status": "ok"})
    sink.close()

    assert sink.closed


def test_live_wandb_runs_counts_sinks_until_each_is_closed_once() -> None:
    from resetsweep.metrics import live_wandb_runs

    before = live_wandb_runs()
    a = WandbSink(project="p", group="g", name="a", config={}, mode="disabled")
    b = WandbSink(project="p", group="g", name="b", config={}, mode="disabled")
    opened = live_wandb_runs()
    a.close()
    a.close()  # idempotent: counted once
    after_one = live_wandb_runs()
    b.close()

    assert (before, opened, after_one, live_wandb_runs()) == (0, 2, 1, 0)

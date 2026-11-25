"""Final sweep report (spec section 6): `report.md`, `report.json`, and W&B tables.

Rules:
- Ranked arms are the live arms that reached the ranking rung (the final rung by default; smoke
  mode ranks rung 0) and have ok results there. Only results at that rung count, so ranked arms
  in a group have equal step budgets; the `seeds` columns show whether they also have equal
  seed counts (a sweep stopped by the budget mid-rung may not), flagged per group.
- Per (group, arm): IQM of ok AUCs, 95% percentile bootstrap CI (`stats.bootstrap_ci`, seed 0),
  ok and total seeds, seeds whose final success is 0, the rung.
- Normalisation per group: `uniform` -> 0, the best ranked arm -> 1 (`stats.normalise`).
  A group is *anchorless* when the scheduler says so (`anchorless_groups()`) or when no
  `uniform` arm is ranked in it; it is ranked by raw AUC within its task, with no normalised
  column, and left out of the overall ranking. A group whose best arm is `uniform` itself (best
  == uniform, nothing to scale by) is *unscaled* and also left out of the overall ranking.
- Overall: per arm configuration (the label without its task), the IQM over groups of its
  normalised score, with the number of groups it counts.
- Unranked arms (pruned, failed, or stopped below the ranking rung) are listed separately with
  the rung they reached and their IQM there, never ranked. Arms that never ran are only counted.

W&B: `WandbReportSink` logs each table as a `wandb.Table` in one sweep-level run. Checked
against the installed wandb 0.31.2 source: `wandb/sdk/data_types/table.py` `Table.__init__`
(`columns`, `data`, `allow_mixed_types`); `wandb.init(..., job_type=..., reinit="create_new")`
as in `resetsweep.metrics.WandbSink`.
"""

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from resetsweep.metrics import WandbMode
from resetsweep.sweep.arms import Arm
from resetsweep.sweep.scheduler import (
    ANCHOR_METHOD,
    LIVE_STATUSES,
    ArmState,
    GroupKey,
    group_of,
)
from resetsweep.sweep.stats import bootstrap_ci, iqm, normalise
from resetsweep.train import RunResult

PER_TASK_COLUMNS = [
    "group",
    "arm",
    "iqm_auc",
    "ci_low",
    "ci_high",
    "seeds_ok",
    "seeds_total",
    "zero_seeds",
    "rung",
    "normalised",
]
OVERALL_COLUMNS = ["config", "iqm_normalised", "n_tasks"]
UNRANKED_COLUMNS = ["arm", "status", "rung", "iqm_auc", "seeds_ok"]


class ReportSource(Protocol):
    """What the report reads from the scheduler."""

    @property
    def rung_budgets(self) -> tuple[int, ...]: ...

    def states(self) -> tuple[ArmState, ...]: ...

    def rung_results(self, arm: Arm, rung: int) -> list[RunResult]: ...

    def anchorless_groups(self) -> list[GroupKey]: ...


class ReportSink(Protocol):
    def log_table(self, name: str, columns: list[str], rows: list[list[object]]) -> None: ...

    def summary(self, data: dict[str, float | str]) -> None: ...

    def close(self) -> None: ...


class NullReportSink:
    def log_table(self, name: str, columns: list[str], rows: list[list[object]]) -> None:
        del name, columns, rows

    def summary(self, data: dict[str, float | str]) -> None:
        del data

    def close(self) -> None:
        pass


class WandbReportSink:
    """One sweep-level W&B run holding the report tables. `config` must hold no secrets."""

    def __init__(
        self, *, project: str, group: str, mode: WandbMode, config: Mapping[str, object]
    ) -> None:
        import wandb

        from resetsweep.metrics import _secret_like_keys  # pyright: ignore[reportPrivateUsage] # one shared secret check

        bad = _secret_like_keys(config)
        if bad:
            raise ValueError(f"W&B config has secret-like keys {bad}; secrets must never be logged")
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            group=group,
            name=f"{group}-report",
            job_type="report",
            config=dict(config),
            mode=mode,
            reinit="create_new",
        )
        self._closed = False

    def log_table(self, name: str, columns: list[str], rows: list[list[object]]) -> None:
        # Table's `columns` is list[ColumnKey] (str | int), invariant; copy into that type.
        keys: list[str | int] = list(columns)
        table = self._wandb.Table(columns=keys, data=rows, allow_mixed_types=True)
        self._run.log({name: table})

    def summary(self, data: dict[str, float | str]) -> None:
        self._run.summary.update(data)

    def close(self) -> None:
        if not self._closed:
            self._run.finish()
            self._closed = True


# --- building ----------------------------------------------------------------------------------


def _config_of(arm: Arm) -> str:
    """The arm label without its task: the same method x score across tasks."""
    return arm.label().split("/", 1)[1]


def group_label(group: GroupKey) -> str:
    task, kw = group
    if not kw:
        return task
    return task + "[" + ",".join(f"{k}={v}" for k, v in kw) + "]"


def _group_json(group: GroupKey) -> dict[str, Any]:
    return {"task": group[0], "task_kw": dict(group[1])}


def _ok_aucs(results: Sequence[RunResult]) -> list[float]:
    return [r.auc for r in results if r.status == "ok"]


def _arm_row(st: ArmState, results: list[RunResult], rung: int) -> dict[str, Any]:
    aucs = np.asarray(_ok_aucs(results), dtype=float)
    lo, hi = bootstrap_ci(aucs, seed=0)
    return {
        "label": st.arm.label(),
        "config": _config_of(st.arm),
        "method": st.arm.method,
        "iqm_auc": iqm(aucs),
        "ci": [lo, hi],
        "seeds_ok": int(aucs.size),
        "seeds_total": len(results),
        "zero_seeds": sum(1 for r in results if r.status == "ok" and r.final_success == 0.0),
        "rung": rung,
        "normalised": None,
    }


def _normalise_group(rows: list[dict[str, Any]], anchored: bool) -> bool:
    """Fill `normalised` in place; return whether the group has a scale."""
    if not anchored:
        return False
    low = next(r["iqm_auc"] for r in rows if r["method"] == ANCHOR_METHOD)
    high = max(r["iqm_auc"] for r in rows)
    if high <= low:
        return False
    for r in rows:
        r["normalised"] = normalise(r["iqm_auc"], low=low, high=high)
    return True


def build_report(source: ReportSource, *, rank_rung: int | None = None) -> dict[str, Any]:
    """The report as plain JSON-ready data (see the module docstring)."""
    last = len(source.rung_budgets) - 1
    rung = last if rank_rung is None else rank_rung
    if not 0 <= rung <= last:
        raise ValueError(f"rank_rung {rank_rung} out of range 0..{last}")
    by_group: dict[GroupKey, list[dict[str, Any]]] = {}
    unranked: list[dict[str, Any]] = []
    not_started = 0
    for st in source.states():
        at_rung = source.rung_results(st.arm, rung)
        if st.status in LIVE_STATUSES and st.rung >= rung and _ok_aucs(at_rung):
            by_group.setdefault(group_of(st.arm), []).append(_arm_row(st, at_rung, rung))
        elif st.results:
            own = source.rung_results(st.arm, st.rung)
            aucs = _ok_aucs(own)
            unranked.append(
                {
                    "label": st.arm.label(),
                    "status": st.status,
                    "rung": st.rung,
                    "iqm_auc": iqm(np.asarray(aucs)) if aucs else None,
                    "seeds_ok": len(aucs),
                }
            )
        else:
            not_started += 1

    flagged = set(source.anchorless_groups())
    groups: list[dict[str, Any]] = []
    scores: dict[str, list[float]] = {}
    anchorless: list[GroupKey] = []
    for group, rows in by_group.items():
        anchored = group not in flagged and any(r["method"] == ANCHOR_METHOD for r in rows)
        if not anchored:
            anchorless.append(group)
        scaled = _normalise_group(rows, anchored)
        rows.sort(key=lambda r: (-r["iqm_auc"], r["label"]))
        if scaled:
            for r in rows:
                scores.setdefault(r["config"], []).append(r["normalised"])
        groups.append(
            {
                **_group_json(group),
                "anchored": anchored,
                "scaled": scaled,
                "equal_seeds": len({r["seeds_ok"] for r in rows}) == 1,
                "arms": rows,
            }
        )
    anchorless += [g for g in flagged if g not in by_group and g not in anchorless]
    overall = sorted(
        (
            {"config": c, "iqm_normalised": iqm(np.asarray(v)), "n_tasks": len(v)}
            for c, v in scores.items()
        ),
        key=lambda row: (-row["iqm_normalised"], row["config"]),
    )
    unranked.sort(key=lambda r: (r["label"]))
    return {
        "rank_rung": rung,
        "step_budget": source.rung_budgets[rung],
        "groups": groups,
        "overall": overall,
        "anchorless_groups": [_group_json(g) for g in anchorless],
        "unranked": unranked,
        "not_started": not_started,
    }


# --- rendering ---------------------------------------------------------------------------------


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Reset-curriculum sweep report",
        "",
        f"Ranked at rung {report['rank_rung']} ({report['step_budget']} env steps per run).",
    ]
    for key, value in report.get("extra", {}).items():
        lines.append(f"- {key}: {_fmt(value)}")
    lines += ["", "## Overall (IQM of per-task normalised AUC)", ""]
    lines += ["| config | IQM normalised | tasks |", "|---|---|---|"]
    lines += [
        f"| {r['config']} | {_fmt(r['iqm_normalised'])} | {r['n_tasks']} |"
        for r in report["overall"]
    ]
    lines += ["", "## Per task", ""]
    for g in report["groups"]:
        name = group_label((g["task"], tuple(g["task_kw"].items())))
        notes = []
        if not g["anchored"]:
            notes.append("anchorless: ranked by raw AUC, not in the overall ranking")
        elif not g["scaled"]:
            notes.append("unscaled: uniform is the best arm, not in the overall ranking")
        if not g["equal_seeds"]:
            notes.append("unequal seed counts")
        lines += [f"### {name}" + (f" ({'; '.join(notes)})" if notes else ""), ""]
        lines += [
            "| arm | IQM AUC | 95% CI | seeds ok/total | zero seeds | rung | normalised |",
            "|---|---|---|---|---|---|---|",
        ]
        for a in g["arms"]:
            ci = f"[{_fmt(a['ci'][0])}, {_fmt(a['ci'][1])}]"
            lines.append(
                f"| {a['label']} | {_fmt(a['iqm_auc'])} | {ci} | {a['seeds_ok']}/{a['seeds_total']} "
                f"| {a['zero_seeds']} | {a['rung']} | {_fmt(a['normalised'])} |"
            )
        lines.append("")
    lines += ["## Unranked arms", ""]
    lines += [f"{report['not_started']} arms never ran.", ""]
    lines += ["| arm | status | rung reached | IQM AUC there | ok seeds |", "|---|---|---|---|---|"]
    lines += [
        f"| {u['label']} | {u['status']} | {u['rung']} | {_fmt(u['iqm_auc'])} | {u['seeds_ok']} |"
        for u in report["unranked"]
    ]
    return "\n".join(lines) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _tables(report: Mapping[str, Any]) -> dict[str, tuple[list[str], list[list[object]]]]:
    per_task: list[list[object]] = [
        [
            group_label((g["task"], tuple(g["task_kw"].items()))),
            a["label"],
            a["iqm_auc"],
            a["ci"][0],
            a["ci"][1],
            a["seeds_ok"],
            a["seeds_total"],
            a["zero_seeds"],
            a["rung"],
            a["normalised"],
        ]
        for g in report["groups"]
        for a in g["arms"]
    ]
    overall: list[list[object]] = [
        [r["config"], r["iqm_normalised"], r["n_tasks"]] for r in report["overall"]
    ]
    unranked: list[list[object]] = [
        [u["label"], u["status"], u["rung"], u["iqm_auc"], u["seeds_ok"]]
        for u in report["unranked"]
    ]
    return {
        "per_task": (PER_TASK_COLUMNS, per_task),
        "overall": (OVERALL_COLUMNS, overall),
        "unranked": (UNRANKED_COLUMNS, unranked),
    }


def write_report(
    scheduler: ReportSource,
    path_md: Path,
    path_json: Path,
    sink: ReportSink,
    *,
    rank_rung: int | None = None,
    extra: Mapping[str, float | str] | None = None,
) -> dict[str, Any]:
    """Build the report, write `path_md` and `path_json`, log the tables and a summary to
    `sink` (which the caller owns and closes). Returns the report data."""
    report = build_report(scheduler, rank_rung=rank_rung)
    report["extra"] = dict(extra or {})
    _write_atomic(path_json, json.dumps(report, indent=2, allow_nan=False) + "\n")
    _write_atomic(path_md, render_markdown(report))
    for name, (columns, rows) in _tables(report).items():
        sink.log_table(name, columns, rows)
    sink.summary(
        {
            **report["extra"],
            "ranked_groups": float(len(report["groups"])),
            "unranked_arms": float(len(report["unranked"])),
        }
    )
    return report

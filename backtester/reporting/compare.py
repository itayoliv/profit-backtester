"""Side-by-side comparison of finished runs, shared by ``bt compare`` and sweep parents."""

from __future__ import annotations

from typing import Any

SORT_KEYS = (
    "sharpe",
    "total_return",
    "return_over_drawdown",
    "max_drawdown",
    "win_rate",
    "trade_count",
)
"""Every sort key ranks higher first; ``max_drawdown`` is negative, so shallower wins."""

MIN_TRADES = 10
"""Below this many trades a run's metrics are mostly noise."""

_SETUP_FIELDS = ("strategy", "engine", "capital", "rules", "filters")


def flatten_runs(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Single-run summaries, with batch and sweep parents replaced by their children."""
    out: list[dict[str, Any]] = []
    for summary in summaries:
        children = summary.get("children")
        out.extend(children if children else [summary])
    return out


def row(summary: dict[str, Any]) -> dict[str, Any]:
    coverage = summary.get("data_coverage") or {}
    total_return = _float(summary.get("total_return"))
    drawdown = _float(summary.get("max_drawdown"))
    return {
        "job_id": summary.get("job_id"),
        "ticker": ",".join(summary.get("tickers") or []),
        "strategy": summary.get("strategy"),
        "strategy_author": (summary.get("strategy_origin") or {}).get("author")
        or (summary.get("strategy_origin") or {}).get("origin"),
        "engine": summary.get("engine"),
        "timeframe": coverage.get("timeframe"),
        "start": coverage.get("requested_start"),
        "end": coverage.get("requested_end"),
        "params": summary.get("params") or {},
        "total_return": total_return,
        "sharpe": _float(summary.get("sharpe")),
        "max_drawdown": drawdown,
        "return_over_drawdown": (
            round(total_return / abs(drawdown), 6)
            if total_return is not None and drawdown is not None and drawdown < 0
            else None
        ),
        "win_rate": _float(summary.get("win_rate")),
        "trade_count": int(summary.get("trade_count") or 0),
        "rule_violations": int(summary.get("rule_violations") or 0),
    }


def compare(summaries: list[dict[str, Any]], sort_by: str = "sharpe") -> dict[str, Any]:
    """Rank runs best first and say which setup fields differ between them.

    Runs with rule violations always sort last, whatever their metrics.
    """
    if sort_by not in SORT_KEYS:
        raise ValueError(f"sort_by must be one of {', '.join(SORT_KEYS)}")
    runs = flatten_runs(summaries)
    rows = [row(s) for s in runs]
    rows.sort(
        key=lambda r: (
            r["rule_violations"] > 0,
            r[sort_by] is None,
            -(r[sort_by] or 0.0),
        )
    )
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank

    warnings = []
    for item in rows:
        if item["rule_violations"]:
            warnings.append(
                f"{item['job_id']}: rule_violations={item['rule_violations']}, do not trust its metrics"
            )
        elif item["trade_count"] < MIN_TRADES:
            warnings.append(
                f"{item['job_id']}: only {item['trade_count']} trades, treat its metrics as noise"
            )

    varied = varied_fields(runs)
    setup_diff = [v for v in varied if not v.startswith("params.")]
    if setup_diff:
        warnings.append(
            f"runs differ in more than parameters ({', '.join(setup_diff)}); "
            "the ranking mixes those effects"
        )

    return {"sort_by": sort_by, "varied": varied, "rows": rows, "warnings": warnings}


def varied_fields(runs: list[dict[str, Any]]) -> list[str]:
    """Setup fields whose value is not the same in every run, e.g. ``params.fast``."""
    if len(runs) < 2:
        return []

    def setup(summary: dict[str, Any]) -> dict[str, Any]:
        coverage = summary.get("data_coverage") or {}
        out: dict[str, Any] = {
            "ticker": tuple(summary.get("tickers") or []),
            "timeframe": coverage.get("timeframe"),
            "window": (coverage.get("requested_start"), coverage.get("requested_end")),
        }
        for name in _SETUP_FIELDS:
            out[name] = repr(summary.get(name))
        for name, value in (summary.get("params") or {}).items():
            out[f"params.{name}"] = repr(value)
        return out

    setups = [setup(s) for s in runs]
    names: list[str] = []
    for item in setups:
        names += [n for n in item if n not in names]
    return [n for n in names if len({repr(s.get(n)) for s in setups}) > 1]


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None

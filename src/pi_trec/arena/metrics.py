from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: float) -> float | str:
    return "" if math.isnan(value) else round(value, 6)


def pairwise_rows(judgments: Iterable[dict[str, Any]], coverage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"a_wins": 0, "b_wins": 0, "ties": 0})
    for row in judgments:
        if row.get("status") != "completed":
            continue
        pair = row.get("pair")
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        run_a, run_b = str(pair[0]), str(pair[1])
        preferred = row.get("preferred_run_id")
        if row.get("judge_verdict") == "Tie":
            counts[(run_a, run_b)]["ties"] += 1
        elif preferred == run_a:
            counts[(run_a, run_b)]["a_wins"] += 1
        elif preferred == run_b:
            counts[(run_a, run_b)]["b_wins"] += 1

    rows: list[dict[str, Any]] = []
    for cov in coverage:
        run_a, run_b = str(cov["run_a"]), str(cov["run_b"])
        item = counts[(run_a, run_b)]
        valid = item["a_wins"] + item["b_wins"] + item["ties"]
        a_rate = (item["a_wins"] + 0.5 * item["ties"]) / valid if valid else math.nan
        b_rate = (item["b_wins"] + 0.5 * item["ties"]) / valid if valid else math.nan
        rows.append(
            {
                "run_a": run_a,
                "run_b": run_b,
                "shared_topics": cov["shared_topics"],
                "valid_judgments": valid,
                "run_a_wins": item["a_wins"],
                "run_b_wins": item["b_wins"],
                "ties": item["ties"],
                "run_a_preference_rate": _fmt(a_rate),
                "run_b_preference_rate": _fmt(b_rate),
            }
        )
    return rows


def system_counts(judgments: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"n_judgments": 0, "wins": 0, "losses": 0, "ties": 0})
    for row in judgments:
        if row.get("status") != "completed":
            continue
        pair = row.get("pair")
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        run_a, run_b = str(pair[0]), str(pair[1])
        counts[run_a]["n_judgments"] += 1
        counts[run_b]["n_judgments"] += 1
        preferred = row.get("preferred_run_id")
        if row.get("judge_verdict") == "Tie":
            counts[run_a]["ties"] += 1
            counts[run_b]["ties"] += 1
        elif preferred == run_a:
            counts[run_a]["wins"] += 1
            counts[run_b]["losses"] += 1
        elif preferred == run_b:
            counts[run_b]["wins"] += 1
            counts[run_a]["losses"] += 1
    return counts


def _valid_outcomes(judgments: Iterable[dict[str, Any]], run_ids: list[str]) -> list[tuple[int, int, float]]:
    index = {run_id: i for i, run_id in enumerate(run_ids)}
    outcomes: list[tuple[int, int, float]] = []
    for row in judgments:
        if row.get("status") != "completed":
            continue
        pair = row.get("pair")
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        run_a, run_b = str(pair[0]), str(pair[1])
        if run_a not in index or run_b not in index:
            continue
        if row.get("judge_verdict") == "Tie":
            y = 0.5
        elif row.get("preferred_run_id") == run_a:
            y = 1.0
        elif row.get("preferred_run_id") == run_b:
            y = 0.0
        else:
            continue
        outcomes.append((index[run_a], index[run_b], y))
    return outcomes


def fit_arena_ratings(
    judgments: Iterable[dict[str, Any]],
    run_ids: list[str],
    *,
    scale: float = 400.0,
    init_rating: float = 1000.0,
) -> dict[str, float]:
    """Fit Arena-style Bradley-Terry ratings on an Elo-like scale."""
    if not run_ids:
        return {}

    outcomes = _valid_outcomes(judgments, run_ids)
    if len(run_ids) == 1 or not outcomes:
        return dict(zip(run_ids, [init_rating for _ in run_ids], strict=True))

    try:
        import numpy as np
        from scipy.optimize import minimize
        from scipy.special import expit
    except ImportError as exc:
        raise RuntimeError("Bradley-Terry ranking requires scipy; install project dependencies with `uv sync`.") from exc

    n_systems = len(run_ids)
    alpha = math.log(10.0)

    def objective(ratings: Any) -> tuple[float, Any]:
        ratings = np.asarray(ratings, dtype=float)
        loss = 0.0
        grad = np.zeros(n_systems, dtype=float)
        for i, j, y in outcomes:
            diff = ratings[i] - ratings[j]
            logit = alpha * diff
            p = float(expit(logit))
            loss += float(np.logaddexp(0.0, logit)) - y * logit
            delta = alpha * (p - y)
            grad[i] += delta
            grad[j] -= delta
        return loss, grad

    result = minimize(
        lambda params: objective(params)[0],
        np.zeros(n_systems, dtype=float),
        jac=lambda params: objective(params)[1],
        method="L-BFGS-B",
        options={"gtol": 1e-8, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Arena rating optimizer failed to converge: {result.message}")

    raw_ratings = cast(Any, np.asarray(result.x, dtype=float))
    raw_ratings = raw_ratings - float(np.mean(raw_ratings))
    arena_scores = raw_ratings * scale + init_rating
    return dict(zip(run_ids, [float(score) for score in arena_scores], strict=True))


def fit_bradley_terry(judgments: Iterable[dict[str, Any]], run_ids: list[str]) -> dict[str, float]:
    return fit_arena_ratings(judgments, run_ids)


def leaderboard_rows(judgments: Iterable[dict[str, Any]], run_ids: list[str]) -> list[dict[str, Any]]:
    judgment_list = list(judgments)
    counts = system_counts(judgment_list)
    scores = fit_arena_ratings(judgment_list, run_ids)
    ranked = sorted(run_ids, key=lambda run_id: (-scores[run_id], run_id))
    rows: list[dict[str, Any]] = []
    for rank, run_id in enumerate(ranked, start=1):
        item = counts[run_id]
        rows.append(
            {
                "rank": rank,
                "run_id": run_id,
                "arena_score": _fmt(scores[run_id]),
                "n_judgments": item["n_judgments"],
                "wins": item["wins"],
                "losses": item["losses"],
                "ties": item["ties"],
            }
        )
    return rows

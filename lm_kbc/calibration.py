"""Train-only calibration for relation-wise self-consistency aggregation.

This module is deliberately offline: it consumes gold training rows and raw
traces that have already been generated.  It never calls an LLM.  Thresholds
are selected with the repository's copy of the official evaluator so that the
calibration objective matches challenge scoring.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from evaluate import RELATION_TYPE, evaluate_per_sr_pair, macro_average_per_relation
from lm_kbc.io import read_jsonl
from lm_kbc.parsing import (
    evaluator_normalize,
    majority_single_with_none,
    support_fraction_candidates,
    threshold_multi_with_none,
)


CALIBRATABLE_RELATIONS = (
    "personHasCityOfDeath",
    "companyTradesAtStockExchange",
    "awardWonBy",
)
NONE_AWARE_RELATIONS = {
    "personHasCityOfDeath",
    "companyTradesAtStockExchange",
}
_NONTRAIN_TOKEN = re.compile(r"(?:^|[._-])(val|validation|test)(?:$|[._-])")
CANONICAL_TRAIN_PATH = Path(__file__).resolve().parents[1] / "data" / "train.jsonl"


def reject_obvious_nontrain_path(path: str | Path, *, label: str = "input") -> None:
    """Reject filenames that plainly identify validation or test material.

    This is a guardrail rather than a claim that filenames prove provenance.
    Key-level checks below additionally require every trace to match a train
    row.
    """

    # Inspect every component, not only the basename: row-shard inputs commonly
    # end in a generic ``rows`` directory while their parent identifies the run
    # as validation/test material (for example ``val-run.state/rows``).
    components = (component.casefold() for component in Path(path).parts)
    if any(_NONTRAIN_TOKEN.search(component) for component in components):
        raise ValueError(
            f"{label} path {str(path)!r} looks like validation/test data; "
            "calibration and SyntheticCoT construction must use train only"
        )


def load_verified_train_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load only rows that are exact subsets of the bundled challenge train set."""

    reject_obvious_nontrain_path(path, label="training data")
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("training data must contain at least one row")
    canonical_rows = read_jsonl(CANONICAL_TRAIN_PATH)

    def encoded(row: dict[str, Any]) -> str:
        return json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    allowed = Counter(encoded(row) for row in canonical_rows)
    observed = Counter(encoded(row) for row in rows)
    unexpected = observed - allowed
    if unexpected:
        sample = json.loads(next(iter(unexpected)))
        key = (sample.get("SubjectEntity"), sample.get("Relation"))
        raise ValueError(
            f"training data contains a row not copied exactly from bundled "
            f"data/train.jsonl: {key!r}"
        )
    return rows


def _sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    if target.is_dir():
        files = sorted(p for p in target.rglob("*.json*") if p.is_file())
        for file_path in files:
            digest.update(str(file_path.relative_to(target)).encode("utf-8"))
            with file_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    else:
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _trace_dicts_from_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        # Immutable RowCheckpointStore shards wrap the model trace in `raw`.
        # Prefer that payload over the shard's top-level identity fields.
        for key in ("raw", "trace"):
            nested_trace = value.get(key)
            if (
                isinstance(nested_trace, dict)
                and "SubjectEntity" in nested_trace
                and "Relation" in nested_trace
            ):
                return [nested_trace]
        if "SubjectEntity" in value and "Relation" in value:
            return [value]
        for key in ("traces", "rows", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
    return []


def _read_trace_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        return _trace_dicts_from_json(json.loads(text))
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in trace file {path} on line {line_number}"
                ) from exc
            rows.extend(_trace_dicts_from_json(value))
        return rows


def load_raw_traces(path: str | Path) -> list[dict[str, Any]]:
    """Load traces from a JSON array, JSONL file, or row-shard directory."""

    reject_obvious_nontrain_path(path, label="trace")
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    if target.is_file():
        traces = _read_trace_file(target)
    else:
        traces = []
        for trace_path in sorted(target.rglob("*.json")):
            traces.extend(_read_trace_file(trace_path))
    if not traces:
        raise ValueError(f"no trace rows found in {target}")
    return traces


def self_consistency_passes(trace: dict[str, Any]) -> list[list[str]]:
    """Validate and return the parsed pass lists from a raw trace."""

    subject = trace.get("SubjectEntity", "<unknown>")
    relation = trace.get("Relation", "<unknown>")
    self_consistency = trace.get("self_consistency")
    passes = self_consistency.get("passes") if isinstance(self_consistency, dict) else None
    if not isinstance(passes, list) or not passes:
        raise ValueError(
            f"trace for ({subject!r}, {relation!r}) has no non-empty "
            "self_consistency.passes list"
        )
    clean: list[list[str]] = []
    for pass_index, values in enumerate(passes, start=1):
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(
                f"trace for ({subject!r}, {relation!r}) pass {pass_index} "
                "must be a list of strings"
            )
        clean.append([value.strip() for value in values if value.strip()])
    return clean


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    subject = row.get("SubjectEntity")
    relation = row.get("Relation")
    if not isinstance(subject, str) or not isinstance(relation, str):
        raise ValueError("every row must contain string SubjectEntity and Relation")
    return subject, relation


def _index_unique(
    rows: Iterable[dict[str, Any]], *, label: str
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        if key in result:
            raise ValueError(f"duplicate {label} row for {key!r}")
        result[key] = row
    return result


def _candidate_fractions(passes: Sequence[list[str]]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for values in passes:
        seen: set[str] = set()
        for value in values:
            key = evaluator_normalize(value)
            if key and key not in seen:
                counts[key] = counts.get(key, 0) + 1
                seen.add(key)
    denominator = len(passes)
    return {key: count / denominator for key, count in counts.items()}


def _observed_thresholds(
    traces: Sequence[dict[str, Any]], *, include_none: bool
) -> tuple[list[float], list[float]]:
    candidate_values = {0.0, 1.0}
    none_values = {0.0, 1.0}
    for trace in traces:
        passes = self_consistency_passes(trace)
        candidate_values.update(_candidate_fractions(passes).values())
        if include_none:
            none_values.add(sum(not values for values in passes) / len(passes))
    return sorted(candidate_values), sorted(none_values)


def aggregate_passes(
    relation: str,
    passes: Sequence[list[str]],
    *,
    candidate_threshold: float,
    none_threshold: float | None = None,
) -> list[str]:
    """Apply the challenge relation's configured self-consistency rule."""

    if relation == "personHasCityOfDeath":
        if none_threshold is None:
            raise ValueError("death aggregation requires a None threshold")
        return majority_single_with_none(passes, none_threshold)
    if relation == "companyTradesAtStockExchange":
        if none_threshold is None:
            raise ValueError("exchange aggregation requires a None threshold")
        return threshold_multi_with_none(
            passes, candidate_threshold, none_threshold
        )
    if relation == "awardWonBy":
        return support_fraction_candidates(passes, candidate_threshold)
    raise ValueError(f"unsupported calibration relation: {relation}")


def _score_relation(
    relation: str,
    relation_traces: Sequence[dict[str, Any]],
    gold_index: dict[tuple[str, str], dict[str, Any]],
    *,
    candidate_threshold: float,
    none_threshold: float | None,
) -> dict[str, float]:
    predictions: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for trace in relation_traces:
        key = _row_key(trace)
        predictions.append(
            {
                "SubjectEntity": key[0],
                "Relation": key[1],
                "ObjectEntities": aggregate_passes(
                    relation,
                    self_consistency_passes(trace),
                    candidate_threshold=candidate_threshold,
                    none_threshold=none_threshold,
                ),
            }
        )
        gold_rows.append(gold_index[key])
    scores = evaluate_per_sr_pair(
        predictions, gold_rows, RELATION_TYPE, tolerance=0.05
    )
    macro = macro_average_per_relation(scores)[relation]
    return {
        "macro_p": float(macro["macro-p"]),
        "macro_r": float(macro["macro-r"]),
        "macro_f1": float(macro["macro-f1"]),
        "predicted_entities": float(sum(score["total_pred"] for score in scores)),
        "empty_predictions": float(sum(score["total_pred"] == 0 for score in scores)),
    }


def _tie_break_key(result: dict[str, float]) -> tuple[float, ...]:
    """F1 first, then a deliberately precision-oriented deterministic tie break."""

    return (
        result["macro_f1"],
        result["macro_p"],
        -result["predicted_entities"],
        result["candidate_threshold"],
        result.get("none_threshold", 0.0),
    )


def calibrate_thresholds(
    train_rows: Sequence[dict[str, Any]],
    trace_rows: Sequence[dict[str, Any]],
    *,
    relations: Sequence[str] | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Sweep observed support fractions and return best train-only settings."""

    if not isinstance(allow_partial, bool):
        raise ValueError("allow_partial must be a boolean")

    gold_index = _index_unique(train_rows, label="gold")
    trace_index = _index_unique(trace_rows, label="trace")
    for key in trace_index:
        if key not in gold_index:
            raise ValueError(
                f"trace row {key!r} is not present in the supplied training gold"
            )
        self_consistency_passes(trace_index[key])

    represented = {relation for _, relation in trace_index}
    if relations is None:
        selected_relations = [
            relation for relation in CALIBRATABLE_RELATIONS if relation in represented
        ]
    else:
        selected_relations = list(relations)
    unknown = set(selected_relations) - set(CALIBRATABLE_RELATIONS)
    if unknown:
        raise ValueError(f"unsupported calibration relations: {sorted(unknown)!r}")
    if not selected_relations:
        raise ValueError("no calibratable relations are represented in the traces")

    best_by_relation: dict[str, dict[str, Any]] = {}
    grids: dict[str, dict[str, int]] = {}
    for relation in selected_relations:
        relation_traces = [
            trace for trace in trace_rows if trace.get("Relation") == relation
        ]
        if not relation_traces:
            raise ValueError(f"no trace rows found for requested relation {relation}")
        expected_keys = {key for key in gold_index if key[1] == relation}
        observed_keys = {_row_key(trace) for trace in relation_traces}
        missing_keys = expected_keys - observed_keys
        if missing_keys and not allow_partial:
            examples = ", ".join(
                repr(subject) for subject, _ in sorted(missing_keys)[:3]
            )
            suffix = "" if len(missing_keys) <= 3 else ", ..."
            raise ValueError(
                f"incomplete train trace coverage for {relation}: found "
                f"{len(observed_keys)} of {len(expected_keys)} rows; missing "
                f"{examples}{suffix}. Finish/resume the train run, or pass "
                "allow_partial=True only for an explicitly partial diagnostic"
            )
        candidate_grid, none_grid = _observed_thresholds(
            relation_traces, include_none=relation in NONE_AWARE_RELATIONS
        )
        # The single-valued death rule always chooses the highest-support entity;
        # only its competing None vote has a configurable threshold.
        if relation == "personHasCityOfDeath":
            candidate_grid = [0.0]
        grids[relation] = {
            "candidate_thresholds": len(candidate_grid),
            "none_thresholds": len(none_grid) if relation in NONE_AWARE_RELATIONS else 0,
        }
        candidates: list[dict[str, float]] = []
        effective_none_grid: Sequence[float | None] = (
            none_grid if relation in NONE_AWARE_RELATIONS else [None]
        )
        for candidate_threshold in candidate_grid:
            for none_threshold in effective_none_grid:
                score = _score_relation(
                    relation,
                    relation_traces,
                    gold_index,
                    candidate_threshold=candidate_threshold,
                    none_threshold=none_threshold,
                )
                score["candidate_threshold"] = candidate_threshold
                if none_threshold is not None:
                    score["none_threshold"] = none_threshold
                candidates.append(score)
        best = max(candidates, key=_tie_break_key)
        best_by_relation[relation] = {
            **(
                {"candidate_threshold": best["candidate_threshold"]}
                if relation != "personHasCityOfDeath"
                else {}
            ),
            **(
                {"none_threshold": best["none_threshold"]}
                if relation in NONE_AWARE_RELATIONS
                else {}
            ),
            "train_macro_p": best["macro_p"],
            "train_macro_r": best["macro_r"],
            "train_macro_f1": best["macro_f1"],
            "rows": len(relation_traces),
            "train_rows_available": len(expected_keys),
            "coverage_complete": not missing_keys,
            "predicted_entities": int(best["predicted_entities"]),
            "empty_predictions": int(best["empty_predictions"]),
        }

    return {
        "relations": best_by_relation,
        "grid_sizes": grids,
        "allow_partial": allow_partial,
        "tie_break": (
            "maximize macro F1, then macro precision, then minimize predicted "
            "entities, then prefer higher candidate and None thresholds"
        ),
    }


def build_yaml_snippet(
    calibration: dict[str, Any],
    *,
    train_path: str | Path,
    traces_path: str | Path,
) -> dict[str, Any]:
    relations = calibration["relations"]
    return {
        "generation": {
            "candidate_thresholds": {
                relation: values["candidate_threshold"]
                for relation, values in relations.items()
                if "candidate_threshold" in values
            },
            "abstention_thresholds": {
                relation: values["none_threshold"]
                for relation, values in relations.items()
                if "none_threshold" in values
            },
        },
        "calibration_provenance": {
            "method": "train-only observed-fraction grid search",
            "objective": "official per-row macro F1 within each relation",
            "numeric_tolerance": 0.05,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_gold_path": str(Path(train_path)),
            "train_gold_sha256": _sha256(train_path),
            "raw_traces_path": str(Path(traces_path)),
            "raw_traces_sha256": _sha256(traces_path),
            "tie_break": calibration["tie_break"],
            "allow_partial": calibration["allow_partial"],
            "grid_sizes": calibration["grid_sizes"],
            "results": relations,
        },
    }


def calibrate_files(
    train_path: str | Path,
    traces_path: str | Path,
    output_path: str | Path,
    *,
    relations: Sequence[str] | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    reject_obvious_nontrain_path(traces_path, label="trace")
    train_rows = load_verified_train_rows(train_path)
    trace_rows = load_raw_traces(traces_path)
    result = calibrate_thresholds(
        train_rows,
        trace_rows,
        relations=relations,
        allow_partial=allow_partial,
    )
    snippet = build_yaml_snippet(
        result, train_path=train_path, traces_path=traces_path
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(snippet, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return snippet

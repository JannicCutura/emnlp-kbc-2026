"""Build a strictly filtered SyntheticCoT store from training traces.

Only rationales whose parsed answers pass the official evaluator are retained.
The builder is offline and does not call a model.  Award rows use a deliberately
different criterion: a non-empty, precision-one subset is useful because award
gold sets can be very large and a single reasoning path is rarely exhaustive.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from evaluate import RELATION_TYPE, evaluate_per_sr_pair
from lm_kbc.calibration import (
    _index_unique,
    _row_key,
    load_raw_traces,
    load_verified_train_rows,
    reject_obvious_nontrain_path,
)
from lm_kbc.io import atomic_write_jsonl, atomic_write_text, read_jsonl
from lm_kbc.parsing import normalize


SYNTHETIC_COT_MANIFEST_VERSION = 1
SYNTHETIC_COT_ARTIFACT_TYPE = "akbc-synthetic-cot"
REQUIRED_SYNTHETIC_KEYS = {
    "SubjectEntity",
    "Relation",
    "ObjectEntities",
    "Rationale",
    "Model",
    "ModelParametersBillion",
}


def synthetic_cot_manifest_path(path: str | Path) -> Path:
    """Return the sidecar path associated with a SyntheticCoT JSONL file."""

    target = Path(path)
    return target.with_name(f"{target.name}.manifest.json")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_source_sha256(path: str | Path) -> str:
    """Hash the trace bytes consumed by ``load_raw_traces`` deterministically."""

    target = Path(path)
    if target.is_file():
        return _file_sha256(target)
    if not target.is_dir():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    files = sorted(
        candidate for candidate in target.rglob("*.json") if candidate.is_file()
    )
    if not files:
        raise ValueError(f"no JSON trace files found in {target}")
    for candidate in files:
        relative = candidate.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(candidate.stat().st_size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _canonical_rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    """Hash logical JSON rows independently of whitespace and key ordering."""

    digest = hashlib.sha256()
    for row in rows:
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _parameter_count(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= 32
    ):
        raise ValueError(f"{label} must be a finite number in (0, 32]")
    return float(value)


def _trace_provenance(
    trace_rows: Sequence[dict[str, Any]],
) -> tuple[str, float]:
    models: set[str] = set()
    parameter_counts: set[float] = set()
    for index, trace in enumerate(trace_rows, start=1):
        model = trace.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"reasoned trace {index} must record its generating model"
            )
        models.add(model)
        parameter_counts.add(
            _parameter_count(
                trace.get("model_parameters_billion"),
                label=(
                    f"reasoned trace {index} model_parameters_billion"
                ),
            )
        )
    if not models:
        raise ValueError("SyntheticCoT construction needs at least one trace")
    if len(models) != 1:
        raise ValueError("SyntheticCoT traces must come from exactly one model")
    if len(parameter_counts) != 1:
        raise ValueError(
            "SyntheticCoT traces must use exactly one model parameter count"
        )
    return next(iter(models)), next(iter(parameter_counts))


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = value.strip()
        key = normalize(value)
        if value and key and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _score_prediction(
    subject: str,
    relation: str,
    predictions: list[str],
    gold_row: dict[str, Any],
) -> dict[str, Any]:
    pred_row = {
        "SubjectEntity": subject,
        "Relation": relation,
        "ObjectEntities": predictions,
    }
    scores = evaluate_per_sr_pair(
        [pred_row], [gold_row], RELATION_TYPE, tolerance=0.05
    )
    if len(scores) != 1:
        raise ValueError(
            f"could not score SyntheticCoT row for {(subject, relation)!r}"
        )
    return scores[0]


def answer_is_eligible(
    subject: str,
    relation: str,
    predictions: list[str],
    gold_row: dict[str, Any],
) -> bool:
    """Return whether an answer satisfies the strict SyntheticCoT filter."""

    score = _score_prediction(subject, relation, predictions, gold_row)
    if relation == "awardWonBy":
        # Exhaustive award lists are unrealistic in one reasoning path. Keep a
        # useful subset only when it has no false positives and at least one TP.
        return score["tp"] >= 1 and score["p"] == 1.0
    if relation in {"hasArea", "hasCapacity"}:
        # The official evaluator applies the task's 5% relative tolerance.
        return (
            len(predictions) == 1
            and score["tp"] == 1
            and score["p"] == 1.0
            and score["r"] == 1.0
        )
    # For death, exchange, and borders, require an exactly correct entity set,
    # including exact empty/non-empty cardinality under official alias matching.
    return score["p"] == 1.0 and score["r"] == 1.0 and score["f1"] == 1.0


def _reasoned_steps(trace: dict[str, Any]) -> Iterable[tuple[str, list[str]]]:
    steps = trace.get("workflow_steps")
    if not isinstance(steps, list):
        return
    for step in steps:
        if not isinstance(step, dict):
            continue
        rationale = step.get("rationale")
        parsed = step.get("parsed")
        if not isinstance(rationale, str) or not rationale.strip():
            continue
        if not isinstance(parsed, list) or any(not isinstance(v, str) for v in parsed):
            continue
        yield rationale.strip(), _deduplicate(parsed)


def build_synthetic_cot_rows(
    train_rows: Sequence[dict[str, Any]],
    trace_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter correct reasoning paths and return deterministic JSONL rows."""

    gold_index = _index_unique(train_rows, label="gold")
    generating_model, model_parameters_billion = _trace_provenance(trace_rows)
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...], str]] = set()
    for trace in trace_rows:
        subject, relation = _row_key(trace)
        key = (subject, relation)
        if key not in gold_index:
            raise ValueError(
                f"trace row {key!r} is not present in the supplied training gold"
            )
        for rationale, predictions in _reasoned_steps(trace):
            if not answer_is_eligible(
                subject, relation, predictions, gold_index[key]
            ):
                continue
            identity = (
                subject,
                relation,
                tuple(normalize(value) for value in predictions),
                rationale,
            )
            if identity in seen:
                continue
            seen.add(identity)
            output.append(
                {
                    "SubjectEntity": subject,
                    "Relation": relation,
                    "ObjectEntities": predictions,
                    "Rationale": rationale,
                    "Model": generating_model,
                    "ModelParametersBillion": model_parameters_billion,
                }
            )
    validate_synthetic_cot_rows(
        output,
        train_rows,
        expected_model=generating_model,
        expected_model_parameters_billion=model_parameters_billion,
    )
    return output


def validate_synthetic_cot_rows(
    rows: Sequence[dict[str, Any]],
    train_rows: Sequence[dict[str, Any]],
    *,
    expected_model: str | None = None,
    expected_model_parameters_billion: float | None = None,
) -> None:
    """Validate schema, train provenance, and evaluator-correct answers.

    This function is intentionally public so configuration loading can reject a
    malformed, leaked, or incorrectly filtered SyntheticCoT file before it is
    placed in a prompt.
    """

    gold_index = _index_unique(train_rows, label="gold")
    expected_parameters = (
        None
        if expected_model_parameters_billion is None
        else _parameter_count(
            expected_model_parameters_billion,
            label="expected_model_parameters_billion",
        )
    )
    observed_models: set[str] = set()
    observed_parameter_counts: set[float] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"SyntheticCoT row {index} must be an object")
        missing = REQUIRED_SYNTHETIC_KEYS - set(row)
        if missing:
            raise ValueError(
                f"SyntheticCoT row {index} is missing keys: {sorted(missing)!r}"
            )
        subject, relation = _row_key(row)
        key = (subject, relation)
        if key not in gold_index:
            raise ValueError(
                f"SyntheticCoT row {index} key {key!r} is not in training data"
            )
        rationale = row["Rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(
                f"SyntheticCoT row {index} must have a non-empty string Rationale"
            )
        model = row["Model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"SyntheticCoT row {index} must have a non-empty string Model"
            )
        observed_models.add(model)
        if expected_model is not None and model != expected_model:
            raise ValueError(
                f"SyntheticCoT row {index} was generated by {model!r}, not the "
                f"active model {expected_model!r}"
            )
        model_parameters = _parameter_count(
            row["ModelParametersBillion"],
            label=f"SyntheticCoT row {index} ModelParametersBillion",
        )
        observed_parameter_counts.add(model_parameters)
        if (
            expected_parameters is not None
            and model_parameters != expected_parameters
        ):
            raise ValueError(
                f"SyntheticCoT row {index} was generated with "
                f"{model_parameters:g}B parameters, not the active model's "
                f"{expected_parameters:g}B parameters"
            )
        answers = row["ObjectEntities"]
        if not isinstance(answers, list) or any(
            not isinstance(value, str) for value in answers
        ):
            raise ValueError(
                f"SyntheticCoT row {index} ObjectEntities must be a flat string list"
            )
        if answers != _deduplicate(answers):
            raise ValueError(
                f"SyntheticCoT row {index} ObjectEntities must be stripped "
                "and deduplicated"
            )
        if not answer_is_eligible(subject, relation, answers, gold_index[key]):
            criterion = (
                "a non-empty precision-one award subset"
                if relation == "awardWonBy"
                else "an evaluator-correct complete answer"
            )
            raise ValueError(
                f"SyntheticCoT row {index} does not satisfy {criterion}"
            )
    if len(observed_models) > 1:
        raise ValueError("SyntheticCoT rows must come from exactly one model")
    if len(observed_parameter_counts) > 1:
        raise ValueError(
            "SyntheticCoT rows must use exactly one model parameter count"
        )


_REQUIRED_MANIFEST_KEYS = {
    "artifact_type",
    "manifest_version",
    "model",
    "model_parameters_billion",
    "train_canonical_sha256",
    "traces_sha256",
    "synthetic_cot_sha256",
    "row_count",
}


def _sha256_value(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"SyntheticCoT manifest {label} must be a SHA-256 hex digest"
        )
    return value


def _read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = synthetic_cot_manifest_path(path)
    if not manifest_path.is_file():
        raise ValueError(
            f"SyntheticCoT provenance manifest is missing: {manifest_path}"
        )
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"SyntheticCoT provenance manifest is invalid JSON: {manifest_path}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("SyntheticCoT provenance manifest must be a JSON object")
    missing = _REQUIRED_MANIFEST_KEYS - set(value)
    extra = set(value) - _REQUIRED_MANIFEST_KEYS
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)!r}")
        if extra:
            details.append(f"unexpected {sorted(extra)!r}")
        raise ValueError(
            "SyntheticCoT provenance manifest has invalid keys: "
            + "; ".join(details)
        )
    if value["artifact_type"] != SYNTHETIC_COT_ARTIFACT_TYPE:
        raise ValueError(
            "SyntheticCoT provenance manifest has the wrong artifact type"
        )
    if (
        isinstance(value["manifest_version"], bool)
        or value["manifest_version"] != SYNTHETIC_COT_MANIFEST_VERSION
    ):
        raise ValueError(
            "SyntheticCoT provenance manifest version is unsupported: "
            f"{value['manifest_version']!r}"
        )
    if not isinstance(value["model"], str) or not value["model"].strip():
        raise ValueError("SyntheticCoT provenance manifest model must be non-empty")
    value["model_parameters_billion"] = _parameter_count(
        value["model_parameters_billion"],
        label="SyntheticCoT manifest model_parameters_billion",
    )
    if (
        isinstance(value["row_count"], bool)
        or not isinstance(value["row_count"], int)
        or value["row_count"] < 0
    ):
        raise ValueError(
            "SyntheticCoT provenance manifest row_count must be non-negative"
        )
    for key in (
        "train_canonical_sha256",
        "traces_sha256",
        "synthetic_cot_sha256",
    ):
        _sha256_value(value[key], label=key)
    return value


def load_synthetic_cot_file(
    path: str | Path,
    train_rows: Sequence[dict[str, Any]],
    *,
    expected_model: str,
    expected_model_parameters_billion: float,
    expected_traces_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load a SyntheticCoT store only when its bound provenance still matches.

    ``expected_traces_path`` is optional because inference need not retain the
    original trace source.  Supplying it re-hashes that source and verifies the
    sidecar's trace binding; artifact, train, model, and parameter bindings are
    always checked.
    """

    if not isinstance(expected_model, str) or not expected_model.strip():
        raise ValueError("expected_model must be a non-empty string")
    expected_parameters = _parameter_count(
        expected_model_parameters_billion,
        label="expected_model_parameters_billion",
    )
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    manifest = _read_manifest(target)
    if manifest["model"] != expected_model:
        raise ValueError(
            f"SyntheticCoT store was generated by {manifest['model']!r}, not the "
            f"active model {expected_model!r}"
        )
    if manifest["model_parameters_billion"] != expected_parameters:
        raise ValueError(
            "SyntheticCoT store was generated with "
            f"{manifest['model_parameters_billion']:g}B parameters, not the "
            f"active model's {expected_parameters:g}B parameters"
        )
    train_hash = _canonical_rows_sha256(train_rows)
    if manifest["train_canonical_sha256"] != train_hash:
        raise ValueError(
            "SyntheticCoT store is bound to different training content"
        )
    artifact_hash = _file_sha256(target)
    if manifest["synthetic_cot_sha256"] != artifact_hash:
        raise ValueError(
            "SyntheticCoT JSONL does not match its provenance manifest"
        )
    if expected_traces_path is not None:
        trace_hash = _trace_source_sha256(expected_traces_path)
        if manifest["traces_sha256"] != trace_hash:
            raise ValueError(
                "SyntheticCoT store is bound to a different trace source"
            )

    rows = read_jsonl(target)
    if _file_sha256(target) != artifact_hash:
        raise ValueError("SyntheticCoT JSONL changed while it was being loaded")
    if manifest["row_count"] != len(rows):
        raise ValueError(
            "SyntheticCoT row count does not match its provenance manifest"
        )
    validate_synthetic_cot_rows(
        rows,
        train_rows,
        expected_model=expected_model,
        expected_model_parameters_billion=expected_parameters,
    )
    return rows


def build_synthetic_cot_file(
    train_path: str | Path,
    traces_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    reject_obvious_nontrain_path(train_path, label="gold")
    reject_obvious_nontrain_path(traces_path, label="trace")
    output = Path(output_path)
    manifest_path = synthetic_cot_manifest_path(output)
    train_source = Path(train_path).resolve()
    trace_source = Path(traces_path).resolve()
    output_target = output.resolve()
    manifest_target = manifest_path.resolve()
    if output_target == train_source or manifest_target == train_source:
        raise ValueError("SyntheticCoT output must not overwrite its training source")
    if trace_source.is_file() and (
        output_target == trace_source or manifest_target == trace_source
    ):
        raise ValueError("SyntheticCoT output must not overwrite its trace source")
    if trace_source.is_dir() and (
        trace_source in output_target.parents
        or trace_source in manifest_target.parents
    ):
        raise ValueError(
            "SyntheticCoT output must be outside its trace-source directory"
        )
    train_rows = load_verified_train_rows(train_path)
    traces_sha256 = _trace_source_sha256(traces_path)
    trace_rows = load_raw_traces(traces_path)
    if _trace_source_sha256(traces_path) != traces_sha256:
        raise ValueError("trace source changed while it was being loaded")
    generating_model, model_parameters_billion = _trace_provenance(trace_rows)
    rows = build_synthetic_cot_rows(train_rows, trace_rows)
    atomic_write_jsonl(rows, output)
    manifest = {
        "artifact_type": SYNTHETIC_COT_ARTIFACT_TYPE,
        "manifest_version": SYNTHETIC_COT_MANIFEST_VERSION,
        "model": generating_model,
        "model_parameters_billion": model_parameters_billion,
        "train_canonical_sha256": _canonical_rows_sha256(train_rows),
        "traces_sha256": traces_sha256,
        "synthetic_cot_sha256": _file_sha256(output),
        "row_count": len(rows),
    }
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return load_synthetic_cot_file(
        output,
        train_rows,
        expected_model=generating_model,
        expected_model_parameters_billion=model_parameters_billion,
        expected_traces_path=traces_path,
    )

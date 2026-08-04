from __future__ import annotations

import json
import os
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Sequence

from .io import atomic_write_jsonl


SUBJECT_FIELD = "SubjectEntity"
RELATION_FIELD = "Relation"
OBJECTS_FIELD = "ObjectEntities"
PREDICTION_MEMBER = "predictions.jsonl"

Row = dict[str, Any]
RowKey = tuple[str, str]


def read_strict_jsonl(path: str | Path) -> list[Row]:
    """Read JSONL with useful line errors and no silently ignored blank records."""
    input_path = Path(path)
    rows: list[Row] = []
    try:
        handle = input_path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read {input_path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(
                    f"{input_path}:{line_number}: blank JSONL lines are not allowed"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{input_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{input_path}:{line_number}: each JSONL row must be an object"
                )
            rows.append(row)
    return rows


def _row_key(row: Row, source: str, row_number: int) -> RowKey:
    subject = row.get(SUBJECT_FIELD)
    relation = row.get(RELATION_FIELD)
    for field, value in ((SUBJECT_FIELD, subject), (RELATION_FIELD, relation)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{source}:{row_number}: {field} must be a non-empty string"
            )
    return subject, relation


def _validate_identities(rows: Sequence[Row], source: str) -> dict[RowKey, int]:
    positions: dict[RowKey, int] = {}
    for row_number, row in enumerate(rows, 1):
        key = _row_key(row, source, row_number)
        if key in positions:
            first = positions[key] + 1
            raise ValueError(
                f"{source}:{row_number}: duplicate SubjectEntity+Relation key {key!r} "
                f"(first seen on row {first})"
            )
        positions[key] = row_number - 1
    return positions


def validate_prediction_rows(
    rows: Sequence[Row],
    *,
    source: str = "predictions",
    expected_rows: int | None = None,
) -> dict[RowKey, int]:
    """Validate submission structure and return each row key's position."""
    if expected_rows is not None:
        if isinstance(expected_rows, bool) or expected_rows < 0:
            raise ValueError("expected row count must be a non-negative integer")
        if len(rows) != expected_rows:
            raise ValueError(
                f"{source}: expected {expected_rows} rows, found {len(rows)}"
            )
    positions = _validate_identities(rows, source)
    for row_number, row in enumerate(rows, 1):
        objects = row.get(OBJECTS_FIELD)
        if not isinstance(objects, list):
            raise ValueError(
                f"{source}:{row_number}: {OBJECTS_FIELD} must be a list of strings"
            )
        for object_number, value in enumerate(objects, 1):
            if not isinstance(value, str):
                raise ValueError(
                    f"{source}:{row_number}: {OBJECTS_FIELD}[{object_number}] "
                    "must be a string (nested arrays and other values are invalid)"
                )
    return positions


def validate_prediction_file(
    path: str | Path, *, expected_rows: int | None = None
) -> list[Row]:
    input_path = Path(path)
    rows = read_strict_jsonl(input_path)
    validate_prediction_rows(
        rows, source=str(input_path), expected_rows=expected_rows
    )
    return rows


def _require_distinct_output(output: Path, inputs: Sequence[Path]) -> None:
    resolved_output = output.resolve()
    for input_path in inputs:
        if input_path.resolve() == resolved_output:
            raise ValueError(f"Output path must differ from input path: {output}")


def select_relation_file(
    input_path: str | Path,
    output_path: str | Path,
    relations: Sequence[str],
) -> list[Row]:
    """Select input relations in original order and remove any exposed answers."""
    source = Path(input_path)
    output = Path(output_path)
    _require_distinct_output(output, [source])
    if not relations:
        raise ValueError("At least one relation must be selected")
    requested = set(relations)
    if any(not isinstance(relation, str) or not relation.strip() for relation in requested):
        raise ValueError("Relation names must be non-empty strings")

    rows = read_strict_jsonl(source)
    _validate_identities(rows, str(source))
    available = {row[RELATION_FIELD] for row in rows}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(
            f"Relations not found in {source}: {', '.join(unknown)}"
        )

    selected = []
    for row in rows:
        if row[RELATION_FIELD] in requested:
            inference_row = dict(row)
            inference_row[OBJECTS_FIELD] = []
            selected.append(inference_row)
    validate_prediction_rows(selected, source=f"selected rows from {source}")
    atomic_write_jsonl(selected, output)
    return selected


def merge_prediction_files(
    base_path: str | Path,
    override_paths: Sequence[str | Path],
    output_path: str | Path,
) -> list[Row]:
    """Replace base ObjectEntities by keyed partial overrides, preserving base order."""
    base = Path(base_path)
    overrides = [Path(path) for path in override_paths]
    output = Path(output_path)
    if not overrides:
        raise ValueError("At least one partial override file is required")
    _require_distinct_output(output, [base, *overrides])

    base_rows = validate_prediction_file(base)
    base_positions = validate_prediction_rows(base_rows, source=str(base))
    merged = [dict(row) for row in base_rows]
    overridden_by: dict[RowKey, Path] = {}

    for override in overrides:
        override_rows = validate_prediction_file(override)
        for row_number, row in enumerate(override_rows, 1):
            key = _row_key(row, str(override), row_number)
            if key not in base_positions:
                raise ValueError(
                    f"{override}:{row_number}: override key {key!r} is not in {base}"
                )
            if key in overridden_by:
                raise ValueError(
                    f"{override}:{row_number}: duplicate override key {key!r}; "
                    f"already supplied by {overridden_by[key]}"
                )
            overridden_by[key] = override
            merged[base_positions[key]][OBJECTS_FIELD] = list(row[OBJECTS_FIELD])

    validate_prediction_rows(merged, source=f"merged predictions for {output}")
    atomic_write_jsonl(merged, output)
    return merged


def _replace_with_retry(temporary: Path, output: Path) -> None:
    for attempt in range(30):
        try:
            os.replace(temporary, output)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(min(0.25 * (attempt + 1), 1.0))


def package_prediction_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    expected_rows: int | None = None,
) -> int:
    """Validate predictions and atomically build the exact Codabench ZIP layout."""
    source = Path(input_path)
    output = Path(output_path)
    _require_distinct_output(output, [source])
    if output.suffix.lower() != ".zip":
        raise ValueError(f"Package output must have a .zip extension: {output}")
    rows = validate_prediction_file(source, expected_rows=expected_rows)
    content = source.read_bytes()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr(PREDICTION_MEMBER, content)
        with zipfile.ZipFile(temporary, "r") as archive:
            if archive.namelist() != [PREDICTION_MEMBER]:
                raise RuntimeError("Internal error: package has an invalid member layout")
            if archive.testzip() is not None:
                raise RuntimeError("Internal error: package failed its CRC check")
        _replace_with_retry(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(rows)



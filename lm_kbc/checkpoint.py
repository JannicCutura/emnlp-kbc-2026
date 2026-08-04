from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from .io import atomic_write_text


CHECKPOINT_VERSION = 3


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class RowCheckpointStore:
    """Immutable one-file-per-row recovery state for long inference runs."""

    def __init__(
        self,
        root: str | Path,
        config: dict[str, Any],
        input_rows: list[dict[str, Any]],
    ) -> None:
        self.root = Path(root)
        self.rows_dir = self.root / "rows"
        self.manifest_path = self.root / "manifest.json"
        self.config = config
        self.input_rows = input_rows
        self.manifest = {
            "version": CHECKPOINT_VERSION,
            "config": config,
            "config_sha256": _stable_digest(config),
            "input_sha256": _stable_digest(
                [
                    [row.get("SubjectEntity"), row.get("Relation")]
                    for row in input_rows
                ]
            ),
            "total_rows": len(input_rows),
        }

    def initialize(self) -> None:
        if self.manifest_path.exists():
            saved = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if saved != self.manifest:
                raise ValueError(
                    "Row-checkpoint manifest does not match this configuration "
                    "and input; use the original command or a different runtime directory"
                )
            return
        self.rows_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.manifest_path,
            json.dumps(self.manifest, ensure_ascii=False, indent=2),
        )

    def row_path(self, index: int) -> Path:
        return self.rows_dir / f"{index:06d}.json"

    def load(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.manifest_path.exists():
            return [], []
        self.initialize()
        predictions: list[dict[str, Any]] = []
        raw_records: list[dict[str, Any]] = []
        existing = sorted(self.rows_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"))
        if len(existing) > len(self.input_rows):
            raise ValueError("Row checkpoint contains more rows than the current input")
        for index, path in enumerate(existing):
            expected_path = self.row_path(index)
            if path != expected_path:
                raise ValueError(
                    f"Row checkpoints are not a contiguous prefix; expected {expected_path}"
                )
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError(f"Row checkpoint {path} must contain a JSON object")
            if record.get("version") != CHECKPOINT_VERSION:
                raise ValueError(
                    f"Row checkpoint {path} has an unsupported shard version"
                )
            expected = self.input_rows[index]
            if (
                type(record.get("row_index")) is not int
                or record.get("row_index") != index
                or record.get("SubjectEntity") != expected.get("SubjectEntity")
                or record.get("Relation") != expected.get("Relation")
            ):
                raise ValueError(f"Row checkpoint {path} does not match its input row")
            prediction = record.get("prediction")
            raw = record.get("raw")
            if not isinstance(prediction, dict) or not isinstance(raw, dict):
                raise ValueError(f"Row checkpoint {path} is incomplete")
            if (
                prediction.get("SubjectEntity") != expected.get("SubjectEntity")
                or prediction.get("Relation") != expected.get("Relation")
            ):
                raise ValueError(
                    f"Row checkpoint {path} prediction does not match its input row"
                )
            entities = prediction.get("ObjectEntities")
            if not isinstance(entities, list) or any(
                not isinstance(entity, str) for entity in entities
            ):
                raise ValueError(
                    f"Row checkpoint {path} prediction must have a flat string "
                    "ObjectEntities array"
                )
            if (
                type(raw.get("row_index")) is not int
                or raw.get("row_index") != index
                or raw.get("SubjectEntity") != expected.get("SubjectEntity")
                or raw.get("Relation") != expected.get("Relation")
            ):
                raise ValueError(
                    f"Row checkpoint {path} raw trace does not match its input row"
                )
            predictions.append(prediction)
            raw_records.append(raw)
        return predictions, raw_records

    def write(
        self,
        index: int,
        input_row: dict[str, Any],
        prediction: dict[str, Any],
        raw: dict[str, Any],
    ) -> None:
        path = self.row_path(index)
        record = {
            "version": CHECKPOINT_VERSION,
            "row_index": index,
            "SubjectEntity": input_row.get("SubjectEntity"),
            "Relation": input_row.get("Relation"),
            "prediction": prediction,
            "raw": raw,
        }
        content = json.dumps(record, ensure_ascii=False, indent=2)
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved != record:
                raise ValueError(f"Refusing to overwrite different row checkpoint {path}")
            return
        atomic_write_text(path, content)

import json
import tempfile
import unittest
from pathlib import Path

from lm_kbc.checkpoint import CHECKPOINT_VERSION, RowCheckpointStore


class RowCheckpointValidationTests(unittest.TestCase):
    def _store_with_one_row(self, root: Path) -> RowCheckpointStore:
        input_row = {"SubjectEntity": "Island", "Relation": "hasArea"}
        store = RowCheckpointStore(root, {"model": "fake"}, [input_row])
        store.initialize()
        store.write(
            0,
            input_row,
            {
                "SubjectEntity": "Island",
                "Relation": "hasArea",
                "ObjectEntities": ["100"],
            },
            {
                "row_index": 0,
                "SubjectEntity": "Island",
                "Relation": "hasArea",
                "final": ["100"],
            },
        )
        return store

    @staticmethod
    def _read_shard(store: RowCheckpointStore) -> dict:
        return json.loads(store.row_path(0).read_text(encoding="utf-8"))

    @staticmethod
    def _write_shard(store: RowCheckpointStore, record: object) -> None:
        store.row_path(0).write_text(json.dumps(record), encoding="utf-8")

    def test_valid_shard_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._store_with_one_row(Path(directory))

            predictions, traces = store.load()

            self.assertEqual(predictions[0]["ObjectEntities"], ["100"])
            self.assertEqual(traces[0]["row_index"], 0)

    def test_rejects_wrong_or_missing_shard_version(self):
        for version in (CHECKPOINT_VERSION - 1, None):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                store = self._store_with_one_row(Path(directory))
                record = self._read_shard(store)
                if version is None:
                    record.pop("version")
                else:
                    record["version"] = version
                self._write_shard(store, record)

                with self.assertRaisesRegex(ValueError, "shard version"):
                    store.load()

    def test_rejects_nested_prediction_identity_mismatch(self):
        for field, value in (
            ("SubjectEntity", "Different Island"),
            ("Relation", "hasCapacity"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                store = self._store_with_one_row(Path(directory))
                record = self._read_shard(store)
                record["prediction"][field] = value
                self._write_shard(store, record)

                with self.assertRaisesRegex(ValueError, "prediction does not match"):
                    store.load()

    def test_rejects_non_flat_or_non_string_object_entities(self):
        for entities in ("100", [["100"]], [100], ["100", None]):
            with self.subTest(entities=entities), tempfile.TemporaryDirectory() as directory:
                store = self._store_with_one_row(Path(directory))
                record = self._read_shard(store)
                record["prediction"]["ObjectEntities"] = entities
                self._write_shard(store, record)

                with self.assertRaisesRegex(ValueError, "flat string"):
                    store.load()

    def test_rejects_raw_trace_identity_or_index_mismatch(self):
        for field, value in (
            ("row_index", 1),
            ("SubjectEntity", "Different Island"),
            ("Relation", "hasCapacity"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                store = self._store_with_one_row(Path(directory))
                record = self._read_shard(store)
                record["raw"][field] = value
                self._write_shard(store, record)

                with self.assertRaisesRegex(ValueError, "raw trace does not match"):
                    store.load()


if __name__ == "__main__":
    unittest.main()

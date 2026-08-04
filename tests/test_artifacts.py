import tempfile
import unittest
import zipfile
from pathlib import Path

from lm_kbc.artifacts import (
    merge_prediction_files,
    package_prediction_file,
    select_relation_file,
    validate_prediction_file,
)
from lm_kbc.io import read_jsonl, write_jsonl


def prediction(subject, relation, objects, **extra):
    return {
        "SubjectEntity": subject,
        "Relation": relation,
        "ObjectEntities": objects,
        **extra,
    }


class RelationSelectionTests(unittest.TestCase):
    def test_selects_in_input_order_and_blanks_gold_answers(self):
        rows = [
            prediction("A", "hasArea", [["10"]], note="keep"),
            prediction("B", "awardWonBy", [["Winner"]]),
            prediction("C", "hasArea", []),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gold.jsonl"
            output = root / "area-input.jsonl"
            write_jsonl(rows, source)

            selected = select_relation_file(source, output, ["hasArea"])

            self.assertEqual([row["SubjectEntity"] for row in selected], ["A", "C"])
            self.assertEqual([row["ObjectEntities"] for row in selected], [[], []])
            self.assertEqual(selected[0]["note"], "keep")
            self.assertEqual(read_jsonl(output), selected)

    def test_rejects_unknown_relation_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rows.jsonl"
            output = root / "subset.jsonl"
            write_jsonl([prediction("A", "hasArea", [])], source)

            with self.assertRaisesRegex(ValueError, "not found"):
                select_relation_file(source, output, ["typoRelation"])

            self.assertFalse(output.exists())


class MergeTests(unittest.TestCase):
    def test_merges_multiple_partials_by_key_and_preserves_base_order(self):
        base_rows = [
            prediction("A", "hasArea", ["1"], provenance="base-a"),
            prediction("B", "awardWonBy", ["Old"], provenance="base-b"),
            prediction("C", "hasCapacity", ["3"]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.jsonl"
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output = root / "merged.jsonl"
            write_jsonl(base_rows, base)
            write_jsonl(
                [prediction("B", "awardWonBy", ["New"], provenance="override")],
                first,
            )
            write_jsonl([prediction("A", "hasArea", ["2"])], second)

            merged = merge_prediction_files(base, [first, second], output)

            self.assertEqual([row["SubjectEntity"] for row in merged], ["A", "B", "C"])
            self.assertEqual(
                [row["ObjectEntities"] for row in merged], [["2"], ["New"], ["3"]]
            )
            self.assertEqual(merged[1]["provenance"], "base-b")
            self.assertEqual(read_jsonl(output), merged)

    def test_rejects_unknown_and_cross_file_duplicate_override_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.jsonl"
            unknown = root / "unknown.jsonl"
            duplicate = root / "duplicate.jsonl"
            output = root / "merged.jsonl"
            write_jsonl([prediction("A", "hasArea", ["1"])], base)
            write_jsonl([prediction("B", "hasArea", ["2"])], unknown)
            with self.assertRaisesRegex(ValueError, "is not in"):
                merge_prediction_files(base, [unknown], output)
            self.assertFalse(output.exists())

            write_jsonl([prediction("A", "hasArea", ["2"])], unknown)
            write_jsonl([prediction("A", "hasArea", ["3"])], duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate override key"):
                merge_prediction_files(base, [unknown, duplicate], output)
            self.assertFalse(output.exists())

    def test_rejects_duplicate_base_keys_and_non_flat_object_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.jsonl"
            partial = root / "partial.jsonl"
            output = root / "merged.jsonl"
            write_jsonl(
                [
                    prediction("A", "hasArea", ["1"]),
                    prediction("A", "hasArea", ["2"]),
                ],
                base,
            )
            write_jsonl([prediction("A", "hasArea", ["3"])], partial)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                merge_prediction_files(base, [partial], output)

            write_jsonl([prediction("A", "hasArea", [["3"]])], base)
            with self.assertRaisesRegex(ValueError, "nested arrays"):
                merge_prediction_files(base, [partial], output)
            self.assertFalse(output.exists())


class PackagingTests(unittest.TestCase):
    def test_packages_exact_archive_member_after_validation(self):
        rows = [prediction("A", "hasArea", ["10"]), prediction("B", "awardWonBy", [])]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidate.jsonl"
            output = root / "submission.zip"
            write_jsonl(rows, source)

            count = package_prediction_file(source, output, expected_rows=2)

            self.assertEqual(count, 2)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.namelist(), ["predictions.jsonl"])
                self.assertEqual(
                    archive.read("predictions.jsonl"), source.read_bytes()
                )

    def test_row_count_or_structure_failure_does_not_replace_existing_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidate.jsonl"
            output = root / "submission.zip"
            write_jsonl([prediction("A", "hasArea", ["10"])], source)
            output.write_bytes(b"existing")

            with self.assertRaisesRegex(ValueError, "expected 2 rows"):
                package_prediction_file(source, output, expected_rows=2)
            self.assertEqual(output.read_bytes(), b"existing")

            write_jsonl([prediction("A", "hasArea", [10])], source)
            with self.assertRaisesRegex(ValueError, "must be a string"):
                validate_prediction_file(source)
            with self.assertRaisesRegex(ValueError, "must be a string"):
                package_prediction_file(source, output)
            self.assertEqual(output.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()

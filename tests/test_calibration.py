import json
import tempfile
import unittest
from pathlib import Path

import yaml
from unittest.mock import patch

import lm_kbc.calibration as calibration
from lm_kbc.calibration import (
    aggregate_passes,
    calibrate_files,
    calibrate_thresholds,
    load_raw_traces,
    load_verified_train_rows,
    reject_obvious_nontrain_path,
)


def gold(subject, relation, entities):
    return {
        "SubjectEntity": subject,
        "Relation": relation,
        "ObjectEntities": entities,
    }


def trace(subject, relation, passes):
    return {
        "SubjectEntity": subject,
        "Relation": relation,
        "self_consistency": {"passes": passes},
    }


class AggregationTests(unittest.TestCase):
    def test_death_uses_majority_candidate_and_none_threshold(self):
        passes = [[], [], ["Paris"]]
        self.assertEqual(
            aggregate_passes(
                "personHasCityOfDeath",
                passes,
                candidate_threshold=0.0,
                none_threshold=0.5,
            ),
            [],
        )
        self.assertEqual(
            aggregate_passes(
                "personHasCityOfDeath",
                passes,
                candidate_threshold=0.0,
                none_threshold=0.9,
            ),
            ["Paris"],
        )

    def test_exchange_none_plurality_can_abstain(self):
        passes = [[], [], ["Nasdaq"]]
        self.assertEqual(
            aggregate_passes(
                "companyTradesAtStockExchange",
                passes,
                candidate_threshold=0.2,
                none_threshold=0.5,
            ),
            [],
        )
        self.assertEqual(
            aggregate_passes(
                "companyTradesAtStockExchange",
                passes,
                candidate_threshold=0.2,
                none_threshold=0.9,
            ),
            ["Nasdaq"],
        )


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            gold("Alice", "personHasCityOfDeath", [["Paris"]]),
            gold("Bob", "personHasCityOfDeath", []),
            gold("Acme", "companyTradesAtStockExchange", [["NYSE"]]),
            gold("PrivateCo", "companyTradesAtStockExchange", []),
            gold("Prize", "awardWonBy", [["Ada"], ["Bob"]]),
        ]
        self.traces = [
            trace("Alice", "personHasCityOfDeath", [["Paris"], ["Paris"], []]),
            trace("Bob", "personHasCityOfDeath", [[], [], ["London"]]),
            trace(
                "Acme",
                "companyTradesAtStockExchange",
                [["NYSE"], ["NYSE"], []],
            ),
            trace(
                "PrivateCo",
                "companyTradesAtStockExchange",
                [[], [], ["Nasdaq"]],
            ),
            trace("Prize", "awardWonBy", [["Ada"], ["Bob"], ["Wrong"]]),
        ]

    def test_sweep_uses_official_relation_macro_f1(self):
        result = calibrate_thresholds(self.train, self.traces)
        relations = result["relations"]
        self.assertEqual(relations["personHasCityOfDeath"]["train_macro_f1"], 1.0)
        self.assertEqual(
            relations["companyTradesAtStockExchange"]["train_macro_f1"], 1.0
        )
        self.assertGreater(relations["awardWonBy"]["train_macro_f1"], 0.0)
        self.assertIn("none_threshold", relations["personHasCityOfDeath"])
        self.assertIn("candidate_threshold", relations["awardWonBy"])

    def test_trace_key_must_come_from_train(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            calibrate_thresholds(
                self.train,
                self.traces + [trace("Leak", "awardWonBy", [["Someone"]])],
            )

    def test_calibrate_files_writes_pasteable_yaml_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train-gold.jsonl"
            trace_path = root / "train-traces.json"
            output_path = root / "thresholds.yaml"
            train_path.write_text(
                "".join(json.dumps(row) + "\n" for row in self.train),
                encoding="utf-8",
            )
            trace_path.write_text(json.dumps(self.traces), encoding="utf-8")
            with patch.object(calibration, "CANONICAL_TRAIN_PATH", train_path):
                calibrate_files(train_path, trace_path, output_path)
            content = yaml.safe_load(output_path.read_text(encoding="utf-8"))
            generation = content["generation"]
            self.assertIn("candidate_thresholds", generation)
            self.assertIn("abstention_thresholds", generation)
            self.assertEqual(
                len(content["calibration_provenance"]["train_gold_sha256"]), 64
            )

    def test_obvious_validation_and_test_paths_are_rejected(self):
        paths = (
            "val.jsonl",
            "validation-traces.json",
            "test_gold.jsonl",
            Path("runs") / "val-run.state" / "rows",
            Path("archive") / "validation" / "rows" / "train-traces.json",
        )
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "train only"):
                    reject_obvious_nontrain_path(path)

    def test_requested_relation_requires_complete_train_coverage_by_default(self):
        partial = [self.traces[0]]
        with self.assertRaisesRegex(ValueError, "incomplete train trace coverage"):
            calibrate_thresholds(
                self.train,
                partial,
                relations=["personHasCityOfDeath"],
            )

        result = calibrate_thresholds(
            self.train,
            partial,
            relations=["personHasCityOfDeath"],
            allow_partial=True,
        )
        metrics = result["relations"]["personHasCityOfDeath"]
        self.assertEqual(metrics["rows"], 1)
        self.assertEqual(metrics["train_rows_available"], 2)
        self.assertFalse(metrics["coverage_complete"])
        self.assertTrue(result["allow_partial"])

    def test_allow_partial_is_propagated_through_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train-gold.jsonl"
            trace_path = root / "train-traces.json"
            output_path = root / "partial-thresholds.yaml"
            train_path.write_text(
                "".join(json.dumps(row) + "\n" for row in self.train),
                encoding="utf-8",
            )
            trace_path.write_text(json.dumps([self.traces[0]]), encoding="utf-8")

            with patch.object(calibration, "CANONICAL_TRAIN_PATH", train_path):
                snippet = calibrate_files(
                    train_path,
                    trace_path,
                    output_path,
                    relations=["personHasCityOfDeath"],
                    allow_partial=True,
                )

            self.assertTrue(snippet["calibration_provenance"]["allow_partial"])

    def test_raw_trace_contract_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train-traces.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "SubjectEntity": "Alice",
                            "Relation": "personHasCityOfDeath",
                            "self_consistency": {"passes": "not-a-list"},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            loaded = load_raw_traces(path)
            with self.assertRaisesRegex(ValueError, "self_consistency.passes"):
                calibrate_thresholds(self.train, loaded)

    def test_row_checkpoint_directory_unwraps_raw_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = Path(directory) / "train-run.state" / "rows"
            rows.mkdir(parents=True)
            wrapped = {
                "version": 3,
                "row_index": 0,
                "SubjectEntity": "wrapper identity",
                "Relation": "wrapper relation",
                "raw": self.traces[0],
            }
            (rows / "000000.json").write_text(
                json.dumps(wrapped), encoding="utf-8"
            )

            loaded = load_raw_traces(rows)

            self.assertEqual(loaded, [self.traces[0]])

    def test_training_rows_must_be_exact_bundled_train_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical-train.jsonl"
            supplied = root / "train-subset.jsonl"
            canonical.write_text(
                json.dumps(self.train[0]) + "\n", encoding="utf-8"
            )
            changed = dict(self.train[0])
            changed["ObjectEntities"] = [["validation fact"]]
            supplied.write_text(json.dumps(changed) + "\n", encoding="utf-8")

            with patch.object(calibration, "CANONICAL_TRAIN_PATH", canonical):
                with self.assertRaisesRegex(ValueError, "not copied exactly"):
                    load_verified_train_rows(supplied)


if __name__ == "__main__":
    unittest.main()

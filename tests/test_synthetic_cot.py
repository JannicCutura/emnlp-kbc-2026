import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lm_kbc.io import read_jsonl
from lm_kbc.synthetic_cot import (
    build_synthetic_cot_file,
    build_synthetic_cot_rows,
    load_synthetic_cot_file,
    synthetic_cot_manifest_path,
    validate_synthetic_cot_rows,
)


def gold(subject, relation, entities):
    return {
        "SubjectEntity": subject,
        "Relation": relation,
        "ObjectEntities": entities,
    }


def reasoned_trace(
    subject,
    relation,
    steps,
    model="fake-model",
    model_parameters_billion=20,
):
    return {
        "SubjectEntity": subject,
        "Relation": relation,
        "model": model,
        "model_parameters_billion": model_parameters_billion,
        "workflow_steps": steps,
    }


def step(rationale, parsed):
    return {"rationale": rationale, "parsed": parsed}


class SyntheticCoTTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            gold("Island", "hasArea", [["100"]]),
            gold(
                "Country",
                "countryLandBordersCountry",
                [["Northland"], ["Southland"]],
            ),
            gold("Prize", "awardWonBy", [["Ada"], ["Bob"]]),
            gold("Living Person", "personHasCityOfDeath", []),
        ]
        self.traces = [
            reasoned_trace(
                "Island",
                "hasArea",
                [
                    step("The island is roughly one hundred km2.", ["104"]),
                    step("A loose recollection exceeds tolerance.", ["106"]),
                ],
            ),
            reasoned_trace(
                "Country",
                "countryLandBordersCountry",
                [
                    step("It has two land neighbours.", ["Northland", "Southland"]),
                    step("This path omitted one neighbour.", ["Northland"]),
                ],
            ),
            reasoned_trace(
                "Prize",
                "awardWonBy",
                [
                    step("Ada is one documented recipient.", ["Ada"]),
                    step("This adds a hallucinated recipient.", ["Ada", "Wrong"]),
                    step("No recipient recalled.", []),
                ],
            ),
            reasoned_trace(
                "Living Person",
                "personHasCityOfDeath",
                [step("The person is living, so no death city exists.", [])],
            ),
        ]

    def build_file(self, train_path, trace_path, output_path):
        with patch(
            "lm_kbc.calibration.CANONICAL_TRAIN_PATH", Path(train_path)
        ):
            return build_synthetic_cot_file(
                train_path, trace_path, output_path
            )

    def test_strict_filters_keep_correct_paths_and_award_subsets(self):
        rows = build_synthetic_cot_rows(self.train, self.traces)
        self.assertEqual(len(rows), 4)
        by_key = {(row["SubjectEntity"], row["Relation"]): row for row in rows}
        self.assertEqual(by_key[("Island", "hasArea")]["ObjectEntities"], ["104"])
        self.assertEqual(
            by_key[("Country", "countryLandBordersCountry")]["ObjectEntities"],
            ["Northland", "Southland"],
        )
        self.assertEqual(by_key[("Prize", "awardWonBy")]["ObjectEntities"], ["Ada"])
        self.assertEqual(
            by_key[("Living Person", "personHasCityOfDeath")]["ObjectEntities"],
            [],
        )

    def test_validator_rejects_nontrain_key(self):
        rows = [
            {
                "SubjectEntity": "Validation Subject",
                "Relation": "hasArea",
                "ObjectEntities": ["100"],
                "Rationale": "Looks correct.",
                "Model": "fake-model",
                "ModelParametersBillion": 20,
            }
        ]
        with self.assertRaisesRegex(ValueError, "not in training"):
            validate_synthetic_cot_rows(rows, self.train)

    def test_validator_rejects_empty_rationale_and_wrong_answer(self):
        empty_rationale = {
            "SubjectEntity": "Island",
            "Relation": "hasArea",
            "ObjectEntities": ["100"],
            "Rationale": " ",
            "Model": "fake-model",
            "ModelParametersBillion": 20,
        }
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_synthetic_cot_rows([empty_rationale], self.train)
        wrong = {**empty_rationale, "ObjectEntities": ["200"], "Rationale": "Guess."}
        with self.assertRaisesRegex(ValueError, "evaluator-correct"):
            validate_synthetic_cot_rows([wrong], self.train)

    def test_validator_rejects_imprecise_award_subset(self):
        row = {
            "SubjectEntity": "Prize",
            "Relation": "awardWonBy",
            "ObjectEntities": ["Ada", "Wrong"],
            "Rationale": "One fact and one hallucination.",
            "Model": "fake-model",
            "ModelParametersBillion": 20,
        }
        with self.assertRaisesRegex(ValueError, "precision-one"):
            validate_synthetic_cot_rows([row], self.train)

    def test_file_builder_outputs_jsonl_and_rejects_val_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train-gold.jsonl"
            trace_path = root / "train-reasoned.json"
            output_path = root / "synthetic-cot.jsonl"
            train_path.write_text(
                "".join(json.dumps(row) + "\n" for row in self.train),
                encoding="utf-8",
            )
            trace_path.write_text(json.dumps(self.traces), encoding="utf-8")
            rows = self.build_file(train_path, trace_path, output_path)
            self.assertEqual(read_jsonl(output_path), rows)
            manifest_path = synthetic_cot_manifest_path(output_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model"], "fake-model")
            self.assertEqual(manifest["model_parameters_billion"], 20.0)
            self.assertEqual(manifest["row_count"], len(rows))
            self.assertEqual(len(manifest["train_canonical_sha256"]), 64)
            self.assertEqual(len(manifest["traces_sha256"]), 64)
            self.assertEqual(len(manifest["synthetic_cot_sha256"]), 64)
            self.assertEqual(
                load_synthetic_cot_file(
                    output_path,
                    self.train,
                    expected_model="fake-model",
                    expected_model_parameters_billion=20,
                    expected_traces_path=trace_path,
                ),
                rows,
            )

            leaked = root / "val.jsonl"
            leaked.write_text(train_path.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train only"):
                self.build_file(leaked, trace_path, output_path)

    def test_store_is_bound_to_one_active_model(self):
        rows = build_synthetic_cot_rows(self.train, self.traces)
        validate_synthetic_cot_rows(
            rows,
            self.train,
            expected_model="fake-model",
            expected_model_parameters_billion=20,
        )
        with self.assertRaisesRegex(ValueError, "active model"):
            validate_synthetic_cot_rows(
                rows, self.train, expected_model="different-model"
            )
        with self.assertRaisesRegex(ValueError, "active model's"):
            validate_synthetic_cot_rows(
                rows,
                self.train,
                expected_model="fake-model",
                expected_model_parameters_billion=19,
            )

        mixed = list(self.traces)
        mixed[0] = {**mixed[0], "model": "different-model"}
        with self.assertRaisesRegex(ValueError, "exactly one model"):
            build_synthetic_cot_rows(self.train, mixed)

        mixed_parameters = list(self.traces)
        mixed_parameters[0] = {
            **mixed_parameters[0],
            "model_parameters_billion": 19,
        }
        with self.assertRaisesRegex(ValueError, "parameter count"):
            build_synthetic_cot_rows(self.train, mixed_parameters)

    def test_loader_rejects_every_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            trace_path = root / "train-traces.json"
            output_path = root / "synthetic.jsonl"
            train_path.write_text(
                "".join(json.dumps(row) + "\n" for row in self.train),
                encoding="utf-8",
            )
            trace_path.write_text(json.dumps(self.traces), encoding="utf-8")
            self.build_file(train_path, trace_path, output_path)

            with self.assertRaisesRegex(ValueError, "active model"):
                load_synthetic_cot_file(
                    output_path,
                    self.train,
                    expected_model="another-model",
                    expected_model_parameters_billion=20,
                )
            with self.assertRaisesRegex(ValueError, "active model's"):
                load_synthetic_cot_file(
                    output_path,
                    self.train,
                    expected_model="fake-model",
                    expected_model_parameters_billion=19,
                )
            changed_train = [dict(row) for row in self.train]
            changed_train[0] = {
                **changed_train[0],
                "ObjectEntities": [["101"]],
            }
            with self.assertRaisesRegex(ValueError, "different training content"):
                load_synthetic_cot_file(
                    output_path,
                    changed_train,
                    expected_model="fake-model",
                    expected_model_parameters_billion=20,
                )

            trace_path.write_text(
                trace_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "different trace source"):
                load_synthetic_cot_file(
                    output_path,
                    self.train,
                    expected_model="fake-model",
                    expected_model_parameters_billion=20,
                    expected_traces_path=trace_path,
                )

            output_path.write_text(
                output_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_synthetic_cot_file(
                    output_path,
                    self.train,
                    expected_model="fake-model",
                    expected_model_parameters_billion=20,
                )

    def test_loader_requires_manifest_and_rows_require_parameter_provenance(self):
        rows = build_synthetic_cot_rows(self.train, self.traces)
        missing_parameters = dict(rows[0])
        del missing_parameters["ModelParametersBillion"]
        with self.assertRaisesRegex(ValueError, "missing keys"):
            validate_synthetic_cot_rows([missing_parameters], self.train)

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "unbound.jsonl"
            output_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "manifest is missing"):
                load_synthetic_cot_file(
                    output_path,
                    self.train,
                    expected_model="fake-model",
                    expected_model_parameters_billion=20,
                )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import akbc
from lm_kbc.io import read_jsonl


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def write_config(
    path: Path,
    *,
    train_path: str,
    model: str = "configured-model",
    parameters: float = 1.0,
) -> None:
    normalized_train_path = train_path.replace("\\", "/")
    path.write_text(
        f"""
lm_studio:
  model: "{model}"
  model_parameters_billion: {parameters}
train_data_file: "{normalized_train_path}"
""".lstrip(),
        encoding="utf-8",
    )


class CapturingPipeline:
    initializations = []
    runs = []

    def __init__(self, config, client, examples):
        self.config = config
        self.client = client
        self.examples = examples
        type(self).initializations.append(self)

    def run(self, rows, output, *, resume=False, runtime_dir=None):
        type(self).runs.append(
            {
                "rows": rows,
                "output": output,
                "resume": resume,
                "runtime_dir": runtime_dir,
            }
        )
        return []


class PredictCliTests(unittest.TestCase):
    def setUp(self):
        CapturingPipeline.initializations.clear()
        CapturingPipeline.runs.clear()

    def test_smoke_schema_and_model_overrides_reuse_the_same_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            inputs = root / "input.jsonl"
            config = root / "config.yaml"
            output = root / "predictions.jsonl"
            write_jsonl(
                train,
                [
                    {
                        "SubjectEntity": "Training Place",
                        "Relation": "hasArea",
                        "ObjectEntities": [["10"]],
                    }
                ],
            )
            write_jsonl(
                inputs,
                [{"SubjectEntity": "Target Place", "Relation": "hasArea"}],
            )
            write_config(config, train_path=str(train))
            args = akbc.build_parser().parse_args(
                [
                    "predict",
                    "--config",
                    str(config),
                    "--input",
                    str(inputs),
                    "--output",
                    str(output),
                    "--smoke",
                    "--output-mode",
                    "json_schema",
                    "--model",
                    "override-model",
                    "--model-parameters-billion",
                    "24",
                ]
            )

            with patch.object(
                akbc, "load_verified_train_rows", return_value=read_jsonl(train)
            ), patch.object(
                akbc, "LMStudioClient", return_value=object()
            ), patch.object(akbc, "PredictionPipeline", CapturingPipeline):
                self.assertEqual(akbc._predict(args), 0)

            self.assertEqual(len(CapturingPipeline.initializations), 1)
            effective = CapturingPipeline.initializations[0].config
            self.assertEqual(effective.lm_studio.model, "override-model")
            self.assertEqual(effective.lm_studio.model_parameters_billion, 24)
            self.assertEqual(effective.generation.output_mode, "json_schema")
            self.assertEqual(set(effective.generation.samples.values()), {1})
            self.assertEqual(effective.generation.temperature, 0.2)
            self.assertEqual(effective.generation.max_tokens, 2048)
            self.assertEqual(
                CapturingPipeline.initializations[0].examples[0]["SubjectEntity"],
                "Training Place",
            )

    def test_dry_run_writes_blank_flat_predictions_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            inputs = root / "input.jsonl"
            config = root / "config.yaml"
            output = root / "predictions.jsonl"
            write_jsonl(train, [])
            write_jsonl(
                inputs,
                [
                    {
                        "SubjectEntity": "A",
                        "Relation": "hasArea",
                        "ObjectEntities": [["exposed gold must not survive"]],
                    },
                    {
                        "SubjectEntity": "B",
                        "Relation": "hasCapacity",
                    },
                ],
            )
            write_config(config, train_path=str(train))

            with patch.object(
                akbc,
                "LMStudioClient",
                side_effect=AssertionError("dry-run contacted LM Studio"),
            ), patch.object(
                akbc,
                "PredictionPipeline",
                side_effect=AssertionError("dry-run constructed the pipeline"),
            ):
                result = akbc.main(
                    [
                        "predict",
                        "--config",
                        str(config),
                        "--input",
                        str(inputs),
                        "--output",
                        str(output),
                        "--limit",
                        "1",
                        "--dry-run",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(
                read_jsonl(output),
                [
                    {
                        "SubjectEntity": "A",
                        "Relation": "hasArea",
                        "ObjectEntities": [],
                    }
                ],
            )

    def test_rejects_validation_named_few_shot_path_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input.jsonl"
            config = root / "config.yaml"
            output = root / "predictions.jsonl"
            suspicious_train = root / "validation-run" / "train-copy.jsonl"
            write_jsonl(
                inputs,
                [{"SubjectEntity": "Target", "Relation": "hasArea"}],
            )
            write_config(config, train_path=str(suspicious_train))
            args = akbc.build_parser().parse_args(
                [
                    "predict",
                    "--config",
                    str(config),
                    "--input",
                    str(inputs),
                    "--output",
                    str(output),
                ]
            )

            with patch.object(
                akbc,
                "LMStudioClient",
                side_effect=AssertionError("guard ran after model construction"),
            ):
                with self.assertRaisesRegex(ValueError, "train only"):
                    akbc._predict(args)

    def test_model_override_requires_identifier_and_parameter_count_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input.jsonl"
            config = root / "config.yaml"
            output = root / "predictions.jsonl"
            write_jsonl(inputs, [])
            write_config(config, train_path=str(root / "train.jsonl"))
            args = akbc.build_parser().parse_args(
                [
                    "predict",
                    "--config",
                    str(config),
                    "--input",
                    str(inputs),
                    "--output",
                    str(output),
                    "--model",
                    "unaccounted-model",
                ]
            )

            with self.assertRaisesRegex(ValueError, "supplied together"):
                akbc._predict(args)

    def test_prediction_output_cannot_overwrite_input_or_train_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input.jsonl"
            train = root / "train.jsonl"
            config = root / "config.yaml"
            write_jsonl(
                inputs, [{"SubjectEntity": "A", "Relation": "hasArea"}]
            )
            write_jsonl(train, [])
            write_config(config, train_path=str(train))
            for protected in (inputs, train):
                args = akbc.build_parser().parse_args(
                    [
                        "predict",
                        "--config",
                        str(config),
                        "--input",
                        str(inputs),
                        "--output",
                        str(protected),
                        "--dry-run",
                    ]
                )
                with self.subTest(protected=protected):
                    with self.assertRaisesRegex(ValueError, "must not overwrite"):
                        akbc._predict(args)

    def test_dry_run_rejects_invalid_or_duplicate_input_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input.jsonl"
            train = root / "train.jsonl"
            config = root / "config.yaml"
            output = root / "predictions.jsonl"
            write_jsonl(train, [])
            write_config(config, train_path=str(train))
            write_jsonl(
                inputs,
                [
                    {"SubjectEntity": "A", "Relation": "hasArea"},
                    {"SubjectEntity": "A", "Relation": "hasArea"},
                ],
            )
            args = akbc.build_parser().parse_args(
                [
                    "predict",
                    "--config",
                    str(config),
                    "--input",
                    str(inputs),
                    "--output",
                    str(output),
                    "--dry-run",
                ]
            )
            with self.assertRaisesRegex(ValueError, "duplicates key"):
                akbc._predict(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

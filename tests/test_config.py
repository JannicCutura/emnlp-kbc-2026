import tempfile
import unittest
from pathlib import Path

from lm_kbc.config import load_config


def write_config(path: Path, extra: str = "", *, lm_extra: str = "") -> None:
    path.write_text(
        (
            "lm_studio:\n"
            "  model: fake-model\n"
            "  model_parameters_billion: 1\n"
            f"{lm_extra}"
            f"{extra}"
        ),
        encoding="utf-8",
    )


class RootConfigValidationTests(unittest.TestCase):
    def test_rejects_unknown_top_level_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            write_config(path, "train_data_flie: data/train.jsonl\n")

            with self.assertRaisesRegex(ValueError, "unknown top-level key"):
                load_config(path)

    def test_accepts_documented_calibration_provenance_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            write_config(
                path,
                "calibration_provenance:\n"
                "  method: train-only observed-fraction grid search\n"
                "  train_gold_sha256: abc123\n",
            )

            config = load_config(path)

            self.assertEqual(config.lm_studio.model, "fake-model")

    def test_calibration_provenance_must_be_a_map(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            write_config(path, "calibration_provenance: ignored-string\n")

            with self.assertRaisesRegex(ValueError, "calibration_provenance"):
                load_config(path)

    def test_malformed_scalar_types_raise_clean_value_errors(self):
        malformed = {
            "temperature": "generation:\n  temperature: bad\n",
            "top_p": "generation:\n  top_p: bad\n",
            "max_tokens": "generation:\n  max_tokens: 1.5\n",
            "few_shot": "generation:\n  few_shot: 1.5\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, extra in malformed.items():
                path = Path(directory) / f"{name}.yaml"
                write_config(path, extra)
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, name):
                        load_config(path)

    def test_lm_studio_string_fields_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("base_url", "api_key"):
                path = Path(directory) / f"{name}.yaml"
                write_config(path, lm_extra=f"  {name}: []\n")
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, name):
                        load_config(path)


if __name__ == "__main__":
    unittest.main()

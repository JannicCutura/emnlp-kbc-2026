import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from lm_kbc.config import (
    AppConfig,
    GenerationConfig,
    LMStudioConfig,
    RELATIONS,
    validate_config,
)
from lm_kbc.io import read_jsonl
from lm_kbc.pipeline import PredictionPipeline


def anchor(description: str = "the exact subject") -> str:
    return json.dumps(
        {
            "description": description,
            "identity_checks": ["not a namesake"],
        }
    )


def answers(*values: str) -> str:
    return json.dumps({"rationale": "closed-book recall", "answers": list(values)})


def plain_answers(*values: str) -> str:
    return json.dumps({"answers": list(values)})


def award_answers(*values: str, reasoned: bool = True) -> str:
    payload = {
        "candidates": [
            {"recipient": value, "work": None, "year": None} for value in values
        ]
    }
    if reasoned:
        payload["rationale"] = "closed-book recipient recall"
    return json.dumps(payload)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response, {"usage": {"total_tokens": 1}}


def pipeline_config(
    *,
    samples: dict[str, int] | None = None,
    few_shot: int = 0,
    output_mode: str = "prompt_json",
    temperature: float = 0.6,
    candidate_thresholds: dict[str, float] | None = None,
    abstention_thresholds: dict[str, float] | None = None,
) -> AppConfig:
    relation_samples = {relation: 1 for relation in RELATIONS}
    if samples:
        relation_samples.update(samples)
    return AppConfig(
        lm_studio=LMStudioConfig(
            model="fake-model", model_parameters_billion=1
        ),
        generation=GenerationConfig(
            output_mode=output_mode,
            temperature=temperature,
            seed=42,
            samples=relation_samples,
            few_shot=few_shot,
            few_shot_by_relation={"awardWonBy": min(few_shot, 1)},
            candidate_thresholds=(
                candidate_thresholds
                or {
                    "companyTradesAtStockExchange": 0.30,
                    "awardWonBy": 0.05,
                }
            ),
            abstention_thresholds=(
                abstention_thresholds
                or {
                    "personHasCityOfDeath": 0.50,
                    "companyTradesAtStockExchange": 0.50,
                }
            ),
        ),
    )


class ConfigTests(unittest.TestCase):
    def test_accepts_exactly_32b_and_rejects_any_larger_model(self):
        allowed = replace(
            pipeline_config(),
            lm_studio=LMStudioConfig(
                model="eligible", model_parameters_billion=32.0
            ),
        )
        validate_config(allowed)

        over_budget = replace(
            allowed,
            lm_studio=replace(
                allowed.lm_studio, model_parameters_billion=32.01
            ),
        )
        with self.assertRaisesRegex(ValueError, "32B"):
            validate_config(over_budget)


class FixedRelationWorkflowTests(unittest.TestCase):
    def test_area_anchors_then_samples_with_distinct_seeds_and_true_median(self):
        client = FakeClient(
            [
                anchor("a small island, not the surrounding archipelago"),
                answers("100"),
                answers("110"),
                answers("120"),
                answers("130"),
            ]
        )
        config = pipeline_config(samples={"hasArea": 4})

        prediction, trace = PredictionPipeline(config, client, []).predict_row(
            {"SubjectEntity": "Exact Island", "Relation": "hasArea"}, 0
        )

        self.assertEqual(prediction["ObjectEntities"], ["115"])
        self.assertEqual(trace["self_consistency"]["method"], "median")
        self.assertEqual(
            [step["stage"] for step in trace["workflow_steps"]],
            ["sample:1", "sample:2", "sample:3", "sample:4"],
        )
        self.assertEqual([call["seed"] for call in client.calls], [42, 52, 53, 54, 55])
        self.assertTrue(
            all(
                "a small island" in call["messages"][0]["content"]
                for call in client.calls[1:]
            )
        )

    def test_capacity_uses_numeric_median(self):
        client = FakeClient(
            [anchor("the exact stadium"), answers("1000"), answers("1200"), answers("1100")]
        )
        config = pipeline_config(samples={"hasCapacity": 3})

        prediction, trace = PredictionPipeline(config, client, []).predict_row(
            {"SubjectEntity": "Example Stadium", "Relation": "hasCapacity"}, 0
        )

        self.assertEqual(prediction["ObjectEntities"], ["1100"])
        self.assertEqual(trace["self_consistency"]["method"], "median")

    def test_death_uses_majority_with_empty_as_none(self):
        client = FakeClient(
            [
                anchor("the exact person"),
                answers(),
                answers("Paris"),
                answers(),
                answers("Paris"),
                answers(),
            ]
        )
        config = pipeline_config(
            samples={"personHasCityOfDeath": 5},
            abstention_thresholds={
                "personHasCityOfDeath": 0.60,
                "companyTradesAtStockExchange": 0.50,
            },
        )

        prediction, trace = PredictionPipeline(config, client, []).predict_row(
            {
                "SubjectEntity": "Uncertain Person",
                "Relation": "personHasCityOfDeath",
            },
            0,
        )

        self.assertEqual(prediction["ObjectEntities"], [])
        self.assertEqual(trace["self_consistency"]["method"], "majority-with-none")
        self.assertEqual(trace["self_consistency"]["empty_samples"], 3)
        self.assertEqual(trace["self_consistency"]["none_threshold"], 0.60)

    def test_exchange_canonicalizes_train_aliases_before_support_threshold(self):
        train_rows = [
            {
                "SubjectEntity": "Training Company",
                "Relation": "companyTradesAtStockExchange",
                "ObjectEntities": [["New York Stock Exchange", "NYSE"]],
            }
        ]
        client = FakeClient(
            [
                anchor("the independently listed legal entity"),
                answers("NYSE"),
                answers("New York Stock Exchange"),
                answers("NYSE"),
                answers(),
            ]
        )
        config = pipeline_config(
            samples={"companyTradesAtStockExchange": 4},
            candidate_thresholds={
                "companyTradesAtStockExchange": 0.75,
                "awardWonBy": 0.05,
            },
            abstention_thresholds={
                "personHasCityOfDeath": 0.50,
                "companyTradesAtStockExchange": 1.00,
            },
        )

        prediction, trace = PredictionPipeline(
            config, client, train_rows
        ).predict_row(
            {
                "SubjectEntity": "Target Company",
                "Relation": "companyTradesAtStockExchange",
            },
            0,
        )

        self.assertEqual(
            prediction["ObjectEntities"], ["New York Stock Exchange"]
        )
        support = trace["self_consistency"]
        self.assertEqual(support["method"], "threshold-with-none")
        self.assertEqual(
            support["passes"],
            [
                ["New York Stock Exchange"],
                ["New York Stock Exchange"],
                ["New York Stock Exchange"],
                [],
            ],
        )
        self.assertIn("raw_passes", support)
        self.assertTrue(support["alias_rewrites"])

    def test_border_unions_samples_completes_omissions_then_verifies(self):
        client = FakeClient(
            [
                anchor("the exact sovereign territory"),
                answers("France"),
                answers("Spain"),
                plain_answers("Portugal"),
                plain_answers("France", "Portugal"),
            ]
        )
        config = pipeline_config(samples={"countryLandBordersCountry": 2})

        prediction, trace = PredictionPipeline(config, client, []).predict_row(
            {
                "SubjectEntity": "Example Country",
                "Relation": "countryLandBordersCountry",
            },
            0,
        )

        self.assertEqual(prediction["ObjectEntities"], ["France", "Portugal"])
        self.assertEqual(trace["self_consistency"]["method"], "union")
        self.assertEqual(trace["completion"]["missing"], ["Portugal"])
        self.assertEqual(trace["verification"], {"accepted": ["France", "Portugal"]})
        self.assertEqual(
            [step["stage"] for step in trace["workflow_steps"]],
            ["sample:1", "sample:2", "completion", "border-verification"],
        )

    def test_malformed_border_verifier_cannot_erase_valid_candidates(self):
        client = FakeClient(
            [
                anchor("the exact sovereign territory"),
                answers("France"),
                plain_answers(),
                "truncated verifier output",
            ]
        )
        config = pipeline_config(samples={"countryLandBordersCountry": 1})

        prediction, trace = PredictionPipeline(config, client, []).predict_row(
            {
                "SubjectEntity": "Example Country",
                "Relation": "countryLandBordersCountry",
            },
            0,
        )

        self.assertEqual(prediction["ObjectEntities"], ["France"])
        self.assertEqual(
            trace["verification"]["fallback"], "malformed-verifier-response"
        )

    def test_award_uses_low_threshold_union_and_completion_without_verifier(self):
        client = FakeClient(
            [
                anchor("the exact award"),
                award_answers("Recipient A"),
                award_answers(),
                award_answers("Recipient B"),
                award_answers("Recipient B"),
                award_answers("Recipient C", reasoned=False),
            ]
        )
        config = pipeline_config(
            samples={"awardWonBy": 4},
            candidate_thresholds={
                "companyTradesAtStockExchange": 0.30,
                "awardWonBy": 0.25,
            },
        )

        prediction, trace = PredictionPipeline(config, client, []).predict_row(
            {"SubjectEntity": "Example Award", "Relation": "awardWonBy"}, 0
        )

        self.assertEqual(
            prediction["ObjectEntities"],
            ["Recipient A", "Recipient B", "Recipient C"],
        )
        self.assertEqual(trace["self_consistency"]["method"], "threshold-union")
        self.assertEqual(trace["completion"]["missing"], ["Recipient C"])
        self.assertIsNone(trace["verification"])
        self.assertNotIn(
            "border-verification",
            [step["stage"] for step in trace["workflow_steps"]],
        )
        self.assertEqual(len(client.calls), 6)

    def test_target_training_row_cannot_supply_its_own_aliases(self):
        target_train_row = {
            "SubjectEntity": "Target Company",
            "Relation": "companyTradesAtStockExchange",
            "ObjectEntities": [["New York Stock Exchange", "NYSE"]],
        }
        client = FakeClient(
            [anchor("the exact company"), answers("NYSE")]
        )

        prediction, trace = PredictionPipeline(
            pipeline_config(), client, [target_train_row]
        ).predict_row(
            {
                "SubjectEntity": "Target Company",
                "Relation": "companyTradesAtStockExchange",
            },
            0,
        )

        self.assertEqual(prediction["ObjectEntities"], ["NYSE"])
        self.assertEqual(trace["self_consistency"]["alias_rewrites"], [])

    def test_unparseable_sample_drops_out_instead_of_ending_the_row(self):
        client = FakeClient(
            [
                anchor("the exact person"),
                "not JSON",
                answers("Paris"),
                answers("Paris"),
            ]
        )
        config = pipeline_config(samples={"personHasCityOfDeath": 3})
        prediction, trace = PredictionPipeline(config, client, []).predict_row(
            {"SubjectEntity": "Example Person", "Relation": "personHasCityOfDeath"},
            0,
        )

        self.assertEqual(prediction["ObjectEntities"], ["Paris"])
        self.assertEqual(trace["unusable_samples"], 1)
        self.assertEqual(trace["self_consistency"]["samples"], 2)
        self.assertEqual(trace["self_consistency"]["unusable_samples"], 1)
        # The failed sample must not be counted as a vote for "no objects".
        self.assertEqual(trace["self_consistency"]["empty_samples"], 0)
        self.assertFalse(trace["workflow_steps"][0]["usable"])

    def test_row_without_any_usable_sample_completes_and_records_the_failure(self):
        client = FakeClient([anchor("the exact person"), "not JSON", "still not JSON"])
        config = pipeline_config(samples={"personHasCityOfDeath": 2})
        prediction, trace = PredictionPipeline(config, client, []).predict_row(
            {"SubjectEntity": "Example Person", "Relation": "personHasCityOfDeath"},
            0,
        )

        self.assertEqual(prediction["ObjectEntities"], [])
        self.assertEqual(trace["unusable_samples"], 2)
        self.assertEqual(trace["self_consistency"]["samples"], 0)

    def test_concurrent_samples_match_the_serial_trace_exactly(self):
        import random
        import time

        class SlowUnorderedClient(FakeClient):
            """Completes samples out of request order to expose any reordering."""

            def __init__(self, responses):
                super().__init__(responses)
                self.lock = __import__("threading").Lock()

            def chat(self, messages, **kwargs):
                with self.lock:
                    response = next(self.responses)
                time.sleep(random.uniform(0, 0.02))
                return response, {"usage": {"total_tokens": 1}}

        responses = [anchor("a small island")] + [
            answers(str(100 + index)) for index in range(9)
        ]
        outcomes = []
        for concurrency in (1, 5):
            config = replace(
                pipeline_config(samples={"hasArea": 9}),
                generation=replace(
                    pipeline_config(samples={"hasArea": 9}).generation,
                    concurrency=concurrency,
                ),
            )
            validate_config(config)
            prediction, trace = PredictionPipeline(
                config, SlowUnorderedClient(list(responses)), []
            ).predict_row({"SubjectEntity": "Isle", "Relation": "hasArea"}, 0)
            outcomes.append(
                (
                    prediction["ObjectEntities"],
                    [step["stage"] for step in trace["workflow_steps"]],
                    trace["self_consistency"]["passes"],
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][0], ["104"])
        self.assertEqual(outcomes[0][1], [f"sample:{i + 1}" for i in range(9)])

    def test_unanimous_empty_border_aggregate_skips_the_omission_pass(self):
        # Only an anchor and three empty samples: any further call would be the
        # completion pass inventing neighbours for a country that has none.
        client = FakeClient(
            [anchor("an island country"), answers(), answers(), answers()]
        )
        config = pipeline_config(samples={"countryLandBordersCountry": 3})
        prediction, trace = PredictionPipeline(config, client, []).predict_row(
            {"SubjectEntity": "Malta", "Relation": "countryLandBordersCountry"}, 0
        )

        self.assertEqual(prediction["ObjectEntities"], [])
        self.assertEqual(trace["completion"], {"skipped": "unanimous-empty-aggregate"})
        self.assertEqual(len(client.calls), 4)

    def test_non_empty_border_aggregate_still_runs_the_omission_pass(self):
        client = FakeClient(
            [
                anchor("a country with land neighbours"),
                answers("France"),
                answers("France"),
                answers("France"),
                plain_answers("Germany"),
                plain_answers("France", "Germany"),
            ]
        )
        config = pipeline_config(samples={"countryLandBordersCountry": 3})
        prediction, trace = PredictionPipeline(config, client, []).predict_row(
            {"SubjectEntity": "Belgium", "Relation": "countryLandBordersCountry"}, 0
        )

        self.assertEqual(prediction["ObjectEntities"], ["France", "Germany"])
        self.assertEqual(trace["completion"]["missing"], ["Germany"])

    def test_unparseable_completion_keeps_the_aggregated_candidates(self):
        client = FakeClient(
            [
                anchor("a country with land neighbours"),
                answers("France"),
                "not JSON",
                plain_answers("France"),
            ]
        )
        prediction, trace = PredictionPipeline(
            pipeline_config(), client, []
        ).predict_row(
            {"SubjectEntity": "Belgium", "Relation": "countryLandBordersCountry"},
            0,
        )

        self.assertEqual(prediction["ObjectEntities"], ["France"])
        self.assertFalse(trace["completion"]["usable"])


class CheckpointTests(unittest.TestCase):
    def test_interrupted_run_resumes_from_immutable_row_shard(self):
        rows = [
            {"SubjectEntity": "A", "Relation": "hasArea"},
            {"SubjectEntity": "B", "Relation": "hasArea"},
        ]
        config = pipeline_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "final" / "predictions.jsonl"
            runtime = root / "runtime"

            first = FakeClient([anchor("A"), answers("10")])
            with self.assertRaises(StopIteration):
                PredictionPipeline(config, first, []).run(
                    rows, output, runtime_dir=runtime
                )

            states = list(runtime.glob("*.state"))
            self.assertEqual(len(states), 1)
            first_shard = states[0] / "rows" / "000000.json"
            self.assertTrue(first_shard.exists())
            immutable_bytes = first_shard.read_bytes()

            second = FakeClient([anchor("B"), answers("20")])
            predictions = PredictionPipeline(config, second, []).run(
                rows, output, runtime_dir=runtime, resume=True
            )

            self.assertEqual(len(second.calls), 2)
            self.assertEqual(
                [row["ObjectEntities"] for row in predictions], [["10"], ["20"]]
            )
            self.assertEqual(read_jsonl(output), predictions)
            self.assertEqual(first_shard.read_bytes(), immutable_bytes)

    def test_resume_rejects_changed_effective_configuration(self):
        row = {"SubjectEntity": "A", "Relation": "hasArea"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions.jsonl"
            runtime = root / "runtime"
            config = pipeline_config()
            PredictionPipeline(
                config, FakeClient([anchor("A"), answers("10")]), []
            ).run([row], output, runtime_dir=runtime)

            changed = replace(
                config,
                generation=replace(config.generation, temperature=0.9),
            )
            with self.assertRaisesRegex(ValueError, "manifest"):
                PredictionPipeline(changed, FakeClient([]), []).run(
                    [row], output, runtime_dir=runtime, resume=True
                )

    def test_resume_rejects_changed_prompt_example_contents(self):
        row = {"SubjectEntity": "A", "Relation": "hasArea"}
        first_examples = [
            {
                "SubjectEntity": "Example",
                "Relation": "hasArea",
                "ObjectEntities": [["10"]],
            }
        ]
        changed_examples = [
            {
                "SubjectEntity": "Example",
                "Relation": "hasArea",
                "ObjectEntities": [["20"]],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions.jsonl"
            runtime = root / "runtime"
            config = pipeline_config()
            PredictionPipeline(
                config,
                FakeClient([anchor("A"), answers("10")]),
                first_examples,
            ).run([row], output, runtime_dir=runtime)

            with self.assertRaisesRegex(ValueError, "manifest"):
                PredictionPipeline(config, FakeClient([]), changed_examples).run(
                    [row], output, runtime_dir=runtime, resume=True
                )


if __name__ == "__main__":
    unittest.main()

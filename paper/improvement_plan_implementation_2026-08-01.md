# Improvement-plan implementation checkpoint - 2026-08-01

This note records the hard cutover to the improvement-plan system. No LM Studio
inference was launched during the refactor, so it does not introduce a new
validation score.

## Hard cutover

The initial prototype, its ablation-specific workflows, and the separate model
sweep are no longer supported. The project now has one system entry point,
`akbc.py`, and one run configuration, `configs/improvement-plan.yaml`.
There is no compatibility layer for the discarded gates, legacy verifiers,
numeric selector, or older prompt policy.

The same command implements all operational variants:

- `akbc.py predict` runs the model pipeline;
- `--smoke` changes only the sample count and completion cap for qualification;
- `--output-mode json_schema` runs the structured-output comparison;
- `--model` together with `--model-parameters-billion` compares another
  downloaded model without creating another pipeline;
- `calibrate`, `build-cot`, `select`, `merge`, and `package` expose the
  offline support operations through the same entry point.

New checkpoints are labelled `improvement-plan-v2`. They intentionally do not
resume the discarded prototype method.

## Fixed inference method

Each subject-relation row follows one opinionated path:

1. The configured local model produces a short identity anchor.
2. The same model produces relation-focused, rationale-bearing samples using
   rotating examples sourced only from challenge training data.
3. The pipeline parses and normalizes every sample, canonicalizing only
   unambiguous aliases observed in nested `train.jsonl` alias sets.
4. It applies the fixed relation-specific aggregation policy.
5. Borders and awards receive one omission-completion pass; borders alone
   receive conservative candidate verification.

| Relation | Samples | Fixed aggregation |
| --- | ---: | --- |
| `hasArea` | 20 | Numeric median |
| `hasCapacity` | 15 | Numeric median |
| `personHasCityOfDeath` | 20 | Majority vote including empty/`None` |
| `companyTradesAtStockExchange` | 10 | Candidate threshold plus empty voting |
| `countryLandBordersCountry` | 3 | Union, completion, and verification |
| `awardWonBy` | 20 | Low-threshold union and one completion |

The system supports tolerant prompt-JSON and Pydantic-derived JSON Schema, but
both modes feed the same cleaners and aggregation logic. There are no hard
pre-generation gates or destructive award verification.

## Persistence and outputs

Every completed input row is committed to its own immutable JSON file. With the
documented `--runtime-dir data/model_outputs` setting, shards live under
`data/model_outputs/<run>-<hash>.improvement-plan-v2.state/rows/`.
`--resume` continues at the first missing row. The manifest binds the input,
effective configuration, prompt examples, and the exact prompt/parser/schema
policy code, preventing mixed runs after a model, threshold, implementation, or
SyntheticCoT change.

A completed run assembles a flat prediction JSONL file and a raw JSON trace.
The trace records the anchor, independent samples, rationales, parsing status,
support, aggregation decision, completions, verification, and token usage.

## Train-only calibration and SyntheticCoT

Threshold calibration is offline and uses the repository's official evaluator.
It requires complete requested-relation training traces by default, rejects
validation/test-looking trace paths and any gold row not copied exactly from the
bundled training file, and writes provenance and input hashes. It never mutates
the run configuration.

The optional SyntheticCoT builder is also offline. It keeps numeric paths only
within the official 5% tolerance, exact complete sets for ordinary string
relations, and non-empty precision-one award subsets. Every retained key must
come from `train.jsonl`. A sidecar manifest binds the store to its model ID,
published parameter count, training content, source traces, and artifact bytes.
A store is generated and consumed by the same chosen model; it is not shared
across model comparisons.

## Challenge-safety decisions

- All neural calls in a run use one configured model whose published total
  parameter count must be at most 32B.
- Few-shot examples, calibration gold, aliases, and SyntheticCoT rows come only
  from `train.jsonl`.
- The system performs no web search, retrieval, external fact lookup,
  fine-tuning, or hand-written validation-entity correction.
- Validation gold is used only by the official evaluator after predictions are
  complete, never for prompt construction or threshold fitting.

## Next empirical step

Replace the placeholder model identifier in `configs/improvement-plan.yaml`
and run the fixed 12-row smoke through `akbc.py predict --smoke`. Inspect its
raw trace and latency before starting relation-only or full validation runs.
The complete commands and artifact locations are documented in the README
section "Local improvement-plan pipeline".

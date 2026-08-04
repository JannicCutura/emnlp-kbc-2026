# Qwen 3.6 smoke experiments — 2026-08-01

## Model and runtime

- Model: `lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q8_0.gguf`
- API identifier: `qwen3.6-27b`
- Total parameters: 27B (within the 32B challenge limit)
- Quantization: Q8_0
- LM Studio context: 8192 tokens
- Flash attention: enabled
- Speculative decoding: disabled
- Direct configuration: reasoning disabled, prompt-JSON output, otherwise the
  same relaxed relation workflows used for the strongest GPT-OSS run

All durable predictions are stored as immutable one-row JSON shards below
`data/model_outputs/<run>.state/rows/`. A combined JSONL is written only when a
run completes.

## Direct non-award smoke comparison

The stratified smoke contains two validation subjects from each relation. The
direct Qwen run completed the first ten rows (all relations except awards).
Mean per-row F1 on these exact subjects was 0.30 for Qwen and 0.50 for the
relaxed GPT-OSS system.

| Relation | Qwen direct mean row F1 | GPT-OSS relaxed mean row F1 |
|---|---:|---:|
| `hasArea` | 0.00 | 0.00 |
| `hasCapacity` | 0.00 | 0.50 |
| `personHasCityOfDeath` | 0.50 | 0.50 |
| `companyTradesAtStockExchange` | 0.00 | 0.50 |
| `countryLandBordersCountry` | 1.00 | 1.00 |

Qwen's four direct numeric predictions were all incorrect. It also missed the
Saudi Stock Exchange for United Wire Factories and hallucinated a London Stock
Exchange listing for RPS Group. Its two border predictions were correct, but
the existing GPT-OSS border workflow was already correct on both rows.

## Bounded numeric reasoning probe

Qwen defaults to reasoning mode. Unbounded reasoning was impractical because
it consumed the whole completion budget without producing a final answer. An
optional two-step probe was therefore implemented:

1. Run a 192-token reasoning pass and retain LM Studio's separate
   `reasoning_content`.
2. Run a reasoning-off extraction pass that must emit exactly one numeric JSON
   value; fall back to the original workflow if extraction is invalid.

This workflow is disabled by default and uses no knowledge beyond the same
model weights and permitted training examples.

| Subject | Qwen direct | Qwen reasoned | Gold | Reasoned F1 |
|---|---:|---:|---:|---:|
| Lošinj | 129.5 | 128.49 | 74.36 | 0.0 |
| Ilha da Queimada Grande | 4.3 | 0.43 | 0.43 | 1.0 |
| Estádio Gileno de Carli | 20000 | 20000 | 5000 | 0.0 |
| University Stadium in Georgia | 32000 | 32000 | 9500 | 0.0 |

The probe improves Qwen from 0/4 to 1/4 exact rows, tying GPT-OSS at 1/4 on
this subset. Runtime was 634 seconds, or approximately 159 seconds per row.
This cost and the tied result do not justify running the probe on all 200
numeric validation rows.

## Award runtime ablations

The original award workflow uses five disjoint time slices plus one omission
pass. With a 4096-token ceiling, Qwen did not finish the first award row within
five minutes. A compact workflow using one direct pass plus one omission pass
at 512 tokens produced 18 candidates for `honorary doctor of Stockholm
University`; only one was correct (precision 0.0556, recall 0.0250, F1 0.0345).
GPT-OSS scored 0 on that same row, but the absolute Qwen result remains too low
to justify an all-award run. A 128-token cap truncated both JSON responses and
was rejected as invalid.

## Decision

Qwen 3.6 27B Q8_0 was the strongest candidate on model specifications, but it
is not a stronger empirical replacement under this closed-book pipeline. A
full 478-row Qwen validation run was not launched because the controlled smoke
was worse than GPT-OSS, while the only successful reasoning correction merely
tied GPT-OSS on the four-row numeric subset at much higher runtime. Future use
of Qwen should be limited to a new, independently justified relation-specific
method rather than wholesale replacement.

## Artifacts

- Direct ten-row shards:
  `data/model_outputs/qwen3.6-27b-relaxed-direct-smoke-*.state/rows/`
- Completed numeric-reasoning predictions:
  `data/model_outputs/qwen3.6-27b-numeric-reasoning-smoke.jsonl`
- Numeric-reasoning shards:
  `data/model_outputs/qwen3.6-27b-numeric-reasoning-smoke-*.state/rows/`
- Award compact shards:
  `data/model_outputs/qwen3.6-27b-awards-compact-smoke-*.state/rows/`

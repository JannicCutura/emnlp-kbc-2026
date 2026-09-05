# Gemma 3 27B Q8 smoke experiment — 2026-08-02

## Model and configuration

- LM Studio identifier: `gemma-3-27b-it`
- Model: Gemma 3 27B Instruct
- Total parameters used for the challenge declaration: 27.4B
- Quantization: Q8_0
- Loaded context: 8192 tokens
- Pipeline configuration: `configs/improvement-plan.yaml`
- Output mode: prompt-enforced JSON (`prompt_json`)
- Input: the fixed 12-row stratified validation smoke set, with two rows per
  relation

The model is within the challenge's 32B total-parameter limit. Quantization
does not reduce the declared parameter count.

The run used the following command:

```powershell
.\.venv\Scripts\python.exe -u akbc.py predict `
  -c configs\improvement-plan.yaml `
  -i data\smoke-stratified.jsonl `
  -o data\model_outputs\gemma-3-27b-it-q8-smoke-v2.jsonl `
  --runtime-dir data\model_outputs `
  --smoke `
  --model gemma-3-27b-it `
  --model-parameters-billion 27.4 `
  --resume `
  --max-restarts 20 `
  --restart-delay 5
```

## Result

The official evaluator gives an overall smoke macro F1 of **0.4500**. This is
a small diagnostic subset and must not be compared directly with the 0.6501
full-validation leaderboard reference.

| Relation | Macro P | Macro R | Macro F1 |
|---|---:|---:|---:|
| `hasArea` | 1.0000 | 1.0000 | 1.0000 |
| `hasCapacity` | 0.0000 | 0.0000 | 0.0000 |
| `personHasCityOfDeath` | 1.0000 | 0.5000 | 0.5000 |
| `companyTradesAtStockExchange` | 1.0000 | 0.5000 | 0.5000 |
| `countryLandBordersCountry` | 0.5000 | 1.0000 | 0.5000 |
| `awardWonBy` | 0.2692 | 0.1591 | 0.2000 |
| **All 12 rows** | **0.6282** | **0.5265** | **0.4500** |

The ten non-award rows have mean row F1 **0.5000**, exactly tying the existing
GPT-OSS smoke baseline rather than improving it. The three rows with empty
gold answers have mean F1 0.6667. Gemma correctly abstained for Joseph L.
Goldstein and RPS Group, but the completion/verifier stages changed an initially
correct empty answer for Antigua and Barbuda into the false prediction
`Montserrat (UK)`.

The two area probes were both correct (`76` for a gold value of `74.36`, within
the official five-percent tolerance, and exact `0.43`). Both capacity probes
were wrong because the model resolved the venue identity incorrectly. For the
two award rows, the Grammy prediction achieved F1 0.4000 (7 of 13 predictions
matched 22 gold entities), while the Stockholm honorary-doctor prediction
achieved F1 0.0000 (0 of 15 matched 40 gold entities).

## Runtime and robustness

The complete run took approximately **51 minutes 35 seconds**, or 258 seconds
per row. The first ten non-award rows took about 21 minutes 51 seconds, or 131
seconds per row. This is well above the improvement plan's 120-second rejection
threshold even before the especially slow award rows are considered.

Immutable row shards preserved all completed work. The launcher recovered
twice with `--resume`: once after LM Studio returned no usable JSON at 10/12
rows, and once after a 300-second read timeout at 11/12 rows. Only the unfinished
row was retried. The final artifacts contain exactly 12 ordered rows with flat
string `ObjectEntities` arrays.

Successful saved traces contain 30 model calls and 18,569 tokens (16,056 prompt
and 2,513 completion tokens). These totals exclude calls that failed before a
row shard could be written. Eleven of twelve anchor responses were schema-valid;
all 18 saved workflow-stage responses were schema-valid. The errors are thus
primarily factual and identity-resolution errors, not JSON parsing failures.

## Decision

Do not launch a full 478-row Gemma run with this configuration. It ties the
non-award GPT-OSS smoke quality while being too slow, fails both capacity
probes, and performs poorly on awards. The perfect two-row area result is the
only positive signal and could justify a bounded, train-derived area-only probe
if a relation-specific experiment is needed. A safer pipeline change suggested
by this run is to preserve a unanimous empty border result instead of allowing
the completion stage to introduce a new country.

## Artifacts

- Predictions: `data/model_outputs/gemma-3-27b-it-q8-smoke-v2.jsonl`
- Raw successful traces:
  `data/model_outputs/raw/gemma-3-27b-it-q8-smoke-v2.raw.json`
- Immutable row state:
  `data/model_outputs/gemma-3-27b-it-q8-smoke-v2-02fc2f914dfff92b.improvement-plan-v2.state/`


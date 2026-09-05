# GPT-OSS-20B validation error analysis

> **Historical result.** This run predates the hard cutover. Its configuration
> and executable pipeline were deliberately removed, so the old paths below are
> provenance labels rather than runnable current commands. The scores and error
> categories remain evidence for the new method.

Run completed on 31 July 2026 using `configs/smoke-gpt-oss-20b.yaml` on all
478 validation rows. The cleaned predictions are in
`output/gpt-oss-20b-val.jsonl` and the complete traces are in
`output/raw/gpt-oss-20b-val.raw.json`.

## Results

| Relation | Macro P | Macro R | Macro F1 | Reference F1 |
|---|---:|---:|---:|---:|
| `awardWonBy` | 0.298 | 0.044 | 0.074 | 0.2338 |
| `companyTradesAtStockExchange` | 0.945 | 0.615 | 0.595 | 0.7702 |
| `countryLandBordersCountry` | 0.978 | 0.970 | 0.970 | 0.9909 |
| `hasArea` | 0.280 | 0.280 | 0.280 | 0.7300 |
| `hasCapacity` | 0.220 | 0.220 | 0.220 | 0.2900 |
| `personHasCityOfDeath` | 0.700 | 0.480 | 0.380 | 0.6200 |
| **All relations** | **0.594** | **0.473** | **0.448** | **0.6501** |

The reference is the user-observed public validation-leaderboard snapshot in
`paper/leaderboard_reference.md`, not a result from this repository.

## Failure analysis

### Classification gates suppress valid answers

The exchange gate passed only 32 rows. It correctly gated 34 empty rows but
also gated 34 of the 64 rows with non-empty gold answers. Even a `no/no` gate
decision was only weakly informative: 30 such rows were truly empty and 26
were non-empty. The model returned `unknown/unknown` for eleven rows, eight of
which had non-empty gold. Treating every `unknown` as an empty prediction is
therefore especially damaging.

The death-city gate has the same problem. It correctly gated 26 empty cases
but incorrectly gated 28 of the 61 non-empty cases. Among the 46 `yes/yes`
decisions, 13 were actually gold-empty. The gate is neither sufficiently
sensitive nor sufficiently specific to make a hard decision before factual
generation.

Recommended ablation: disable both hard gates. Always attempt factual recall,
then combine the recalled candidate and the gate assessment. At minimum,
`unknown` must pass through to generation. Compare no gate, permissive gate,
and current hard gate on the same model.

### Award verification removes correct recipients

Before verification, the union of the chronological, categorical, and
omission passes contained 96 correct recipients. The verifier retained only 56
of them. Estimated award macro precision/recall/F1 before verification were
0.221/0.107/0.121; after verification they became 0.298/0.044/0.074. The
precision gain does not compensate for the recall collapse.

The most severe case was the Turing Award: the generated union contained 36
correct recipients, while verification retained only 17. Across the ten award
rows, mean prediction size was 15.2 against a mean of 145.8 gold recipients.
Three awards had no correct final recipient at all.

Recommended ablation: do not apply binary verification to award unions. Keep
the full union, or retain candidates unless the verifier expresses strong
contradictory evidence. Increase recall through smaller time windows and
continuation passes that see the already-generated names.

### Numeric errors are mainly knowledge errors, not formatting errors

Every numeric row produced one parseable prediction. Area accuracy was 28/100
and capacity accuracy 22/100 under the official five-percent tolerance. The
median relative error was 50.5% for area and 37.5% for capacity. Thus, parsing
and output cardinality are functioning; the model often recalls the wrong
fact or confuses the entity with a similarly named or containing entity.

An oracle that selected a correct value whenever any of the three independent
passes was within tolerance would score 34/100 on area and 30/100 on capacity.
This indicates that median aggregation discards some useful minority answers,
but aggregation changes alone cannot approach the 0.73 area reference.

Examples of entity confusion include `Mainland` (predicted 9,596,961 instead
of 540 km²), `Grande Terre` (587,041 instead of 16,648 km²), `Hong Kong Island`
(1,104 instead of 80.4 km²), and `Diamond Sports Stadium in South Australia`
(30,000 instead of capacity 3,000).

Recommended changes: make every pass restate the exact subject and reject
figures belonging to a country, region, city, similarly named venue, or former
configuration. Add a final scale-consistency comparison across candidates.
For experiments, compare median selection with an LLM selector that sees all
three values and their short justifications.

### Borders are already strong

Borders achieved 0.970 macro F1. There were 37 perfect non-empty rows, 18
correct empty rows, and 13 partial rows; there were no wholly wrong non-empty
rows. Remaining errors are mostly one missing or extra neighbor. Examples
include Eswatini (one of two missing), Burkina Faso (four of six), and Namibia
(three of four). This workflow should remain stable while higher-impact
relations are changed.

### Death-city generation still needs factual precision

Beyond the 32 false-empty predictions, 20 non-empty predictions named the
wrong city, only nine non-empty cases were exactly correct, and ten gold-empty
rows received a city. Disabling the gate will increase recall but can worsen
empty-case precision, so candidate generation needs an explicit option to
return no city when the person is living or the locality is genuinely
unknown. The gate should become evidence supplied to a final joint decision,
not a hard early exit.

## Prioritized next experiments

1. Disable or relax the exchange and death gates.
2. Disable award verification while retaining the complete candidate union.
3. Run those two changes on validation and measure overall and zero-object
   scores; these are low-cost changes with direct trace evidence.
4. Add justification-aware numeric selection and entity/scale checks.
5. Expand award recall through narrower chronological continuation chunks.
6. Leave the border workflow unchanged as a control.

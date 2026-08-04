# AKBC Shared Task 2026: Knowledge Base Construction from Language Models (5th Edition)

This repository hosts data for the [AKBC Shared Task](https://lm-kbc.github.io/challenge2026/) at [AKBC](https://www.akbc.ws/2026/) / [EMNLP 2026](https://2026.emnlp.org/) in Budapest.

This repository contains:

- The [dataset](data/) for the shared task
- [Evaluation script](evaluate.py)
- The single local [improvement-plan pipeline](akbc.py) and
  [configuration](configs/improvement-plan.yaml)
- Instructions for submitting your predictions

## Table of contents

1. [News](#news)
2. [Challenge overview](#challenge-overview)
3. [Dataset](#dataset)
4. [Relation definitions](#relation-definitions)
5. [Evaluation metrics](#evaluation-metrics)
6. [Getting started](#getting-started)
    - [Setup](#setup)
    - [Local improvement-plan pipeline](#local-improvement-plan-pipeline)
    - [How to structure your prediction file](#how-to-structure-your-prediction-file)
    - [Submit your predictions](#submit-your-predictions)

## News

- **July 2026**: Ground-truth quality release + evaluator update. **Splits and `test.jsonl` are unchanged** — only gold answer sets were corrected (val: 21 rows, train: 25 rows, plus the private test key; ~160 further rows per split gained additional aliases only). Details:
    - The ground truth now reflects the state of the world **as of 1 July 2026**: city-of-death entries for people who died 2022–2026, current stock-exchange listings (delistings such as RPS Group and Daimler/Mercedes-Benz removed; missing listings such as Bharti Airtel's NSE listing added), and award winners through the most recent editions.
    - Award rows cleaned and completed: winning *works* (books, albums) were replaced by their authors/artists; recipients of similarly-named but distinct awards were removed (e.g. the 1945 Medal of Freedom vs the *Presidential* Medal of Freedom); rescinded awards are excluded; most award rows are now verified-complete winner lists.
    - Border corrections (e.g. Djibouti +Somalia, Denmark +Canada) and further hectare→km² unit fixes for `hasArea`.
    - Alias sets additionally include Wikipedia sitelink titles and Latin-script labels from all languages, making matching robust to name-order variants (e.g. Hungarian "Family-name First-name" ≡ Western "First-name Family-name", "Soong Mei-ling" ≡ "Soong May-ling").
    - `evaluate.py` normalization improved: apostrophe-like marks (`'`, `’`, `ʻ`) are dropped, **all** Unicode punctuation acts as a separator (previously ASCII-only), and case-folding is applied after Unicode decomposition (`ß` → `ss`). `O'Brien` / `O’Brien` / `OBrien` and `Kaua'i` / `Kauaʻi` / `Kauai` now all match. **Please re-download `evaluate.py`** — the change only converts former misses into matches.
- **May 2026**: Dataset cleanup release. Changes participants should be aware of:
    - Object alias sets are now richer per entity (full Latin-language label + aliases pulled from Wikidata), giving the evaluator more matching surface area. Non-Latin scripts, social-media handles (`@...`), ordinal abbreviations (`POTUS 45`), Wikipedia list articles, and orphaned name suffixes (`Jr.`/`Sr.`) have been filtered out.
    - Subjects that previously contained raw Q-IDs (`"Q5847811 in Lima"`) are replaced with the resolved Wikidata label (`"Estadio Caballeros del Deporte in Lima"`).
    - The val row for `"Ireland" (hasArea)` is now labeled `"Island of Ireland"` to disambiguate it from the Republic of Ireland used elsewhere in the dataset.
    - Five `hasArea` values were stored in the wrong unit (hectares or square miles) and have been corrected to km²: Isle of Bute, South Uist, Nantucket, Molokai, Bequia.
    - One duplicate row (`United Kingdom`, `countryLandBordersCountry`) was removed from `test.jsonl`; the test set now has 477 rows instead of 478.
    - Six relations now have explicit definitions ([§ Relation definitions](#relation-definitions)) clarifying scope (e.g. land-only borders, total country area, city-granularity death location, distinct predecessor/successor awards).
- **April 2026**: Release of dataset, baseline, and evaluation script.

## Challenge overview

Pretrained language models (LMs) contain a substantial amount of factual knowledge. Turning that knowledge into reliable knowledge base entries, however, is much harder than answering a single factual question. In this shared task, we invite participants to build knowledge bases from LMs for given subjects and relations. In crucial difference to existing probing benchmarks like LAMA ([Petroni et al., 2019](https://arxiv.org/pdf/1909.01066.pdf)), we make no simplifying assumptions on relation cardinalities, i.e., a subject-entity can stand in relation with zero, one, or many object-entities.

Unlike earlier editions, this version does **not require entity disambiguation**. Predictions are evaluated as **strings** using normalization and alias matching.

> Formally, given the input subject-entity (s) and relation (r), the task is to
> predict all the correct
> object-entities ({o<sub>1</sub>, o<sub>2</sub>, ..., o<sub>k</sub>}) using LM
> probing.

## Dataset

Number of unique subject-entities in the data splits.

<table>
<thead>
    <tr>
        <th>Relation</th>
        <th>Train</th>
        <th>Val</th>
        <th>Test</th>
        <th>Special features</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td>countryLandBordersCountry</td>
        <td>67</td>
        <td>68</td>
        <td>67</td>
        <td>Null values possible; <em>land</em> borders only</td>
    </tr>
    <tr>
        <td>personHasCityOfDeath</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Null values possible</td>
    </tr>
    <tr>
        <td>hasCapacity</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Object is numeric</td>
    </tr>
    <tr>
        <td>awardWonBy</td>
        <td>10</td>
        <td>10</td>
        <td>10</td>
        <td>Many objects per subject</td>
    </tr>
    <tr>
        <td>companyTradesAtStockExchange</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Null values possible</td>
    </tr>
        <tr>
        <td>hasArea</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Object is numeric (square km)</td>
    </tr>
</tbody>
</table>

## Relation definitions

These are the precise scopes used when constructing the ground truth. Models will be evaluated against these definitions, so participants should target them rather than a generic interpretation. The ground truth reflects the state of the world **as of 1 July 2026** (deaths, stock-exchange listings, award winners, and borders up to that date).

- **`countryLandBordersCountry`** — countries (or comparable territories) that share a **land** border with the subject. Maritime borders (e.g. Russia–Japan, Samoa–USA) are **excluded**. Island countries without a land border have an empty answer set. Includes only currently-recognised states; deprecated/disputed border statements on Wikidata are not considered. Borders through a country's integral overseas territory count (e.g. Suriname–France via French Guiana, Spain–Morocco via Ceuta and Melilla), while borders via non-integral dependencies do not (e.g. Cyprus–United Kingdom via the Sovereign Base Areas). Enclave borders count (e.g. Vatican City–Italy).

- **`personHasCityOfDeath`** — the **city** where the person died. Granularity is the city (or the most specific publicly known locality), not the country or region. If the person is still living or no locality is known, the answer is empty.

- **`hasCapacity`** — the **maximum spectator capacity** of the venue, expressed as an integer **number of people**. For stadiums and arenas this corresponds to Wikidata's `P1083` (maximum capacity). When multiple capacities exist (seated vs total, before/after renovation), the **highest published capacity** is used.

- **`awardWonBy`** — entities that have received the specific award identified by the subject. Winners are recorded as the **recipient entities** (people, groups, organizations, projects) — not the winning works. Predecessor or successor awards (e.g. *Medal of Freedom* vs *Presidential Medal of Freedom*) are **distinct** and not bundled, and rescinded awards are excluded. Some awards have hundreds of recipients; participants should expect large object sets. For a small number of awards with very large or open-ended recipient sets (e.g. product-design awards, honorary doctorates), the gold set is necessarily partial.

- **`companyTradesAtStockExchange`** — the stock exchange(s) on which the company's shares are publicly traded. Multiple listings are possible. Subsidiaries that are not separately listed have an empty answer set.

- **`hasArea`** — the surface area of the subject geographic entity, in **square kilometres** (km²). For countries, the **total area** (land + inland water) is used, matching the Wikidata preferred-rank value for `P2046`. Areas reported on Wikidata in hectares, square miles, etc. are converted to km².

## Evaluation metrics

We evaluate predictions using **macro precision, recall, and F1-score**.

For **string relations**, predicted strings are normalized (case-folded, diacritics removed, apostrophe-like marks dropped, punctuation of any script treated as whitespace) and matched against the ground-truth label and its known aliases via maximum bipartite matching — each gold entity credits at most one prediction and vice versa, independent of prediction order. Predictions are deduplicated by normalized string; note that predicting several surface forms of the *same* entity (e.g. `["NYC", "New York City"]`) counts as separate predictions and lowers precision.
For **numeric relations** (`hasCapacity`, `hasArea`), a prediction is correct if it falls within **5% relative tolerance** of the ground-truth value.

See the evaluation script ([evaluate.py](evaluate.py)) for details.

```bash
python evaluate.py \
  -g data/val.jsonl \
  -p your_predictions.jsonl
```

Parameters: ``-g`` (the ground truth file), ``-p`` (the prediction file).

## Getting started

### Setup

1. Clone this repository:

    ```bash
    mkdir lm-kbc-2026
    cd lm-kbc-2026
    git clone https://github.com/lm-kbc/dataset2026.git
    cd dataset2026
    ```

2. Create a virtual environment and install the requirements:

    ```bash
    conda create -n lm-kbc-2026 python=3.11
    ```

    ```bash
    conda activate lm-kbc-2026
    pip install -r requirements.txt
    ```

3. Write your own solution and generate predictions (format described
   in [How to structure your prediction file](#how-to-structure-your-prediction-file)).
4. Evaluate your predictions using the evaluation script
   (see [Evaluation metrics](#evaluation-metrics)).
5. Submit your predictions
   (see [Submit your predictions](#submit-your-predictions)).

### Local improvement-plan pipeline

`akbc.py` is the only supported local solution entry point. It implements the
improvement-plan method with one eligible model served by LM Studio. The system
is closed-book: it does not browse, retrieve external facts, fine-tune a model,
or combine several neural models.

#### Setup and configuration

Create the environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Load one model in LM Studio and start its OpenAI-compatible local server. Edit
`configs/improvement-plan.yaml` once:

- set `lm_studio.model` to the exact identifier shown by LM Studio;
- set `lm_studio.model_parameters_billion` to the published **total** parameter
  count;
- keep the value at or below the challenge limit of 32B.

Quantization does not reduce the counted parameter total, and a
mixture-of-experts model is counted by total rather than active parameters.
Every neural call in a run uses this one configured model.

Validate the configuration and input without contacting LM Studio:

```powershell
python akbc.py predict `
  -c configs/improvement-plan.yaml `
  -i data/smoke-stratified.jsonl `
  -o data/model_outputs/dry-run.jsonl `
  --dry-run
```

#### Method

Every row first receives a short identity description from the same model.
The pipeline then rotates train-only few-shot examples and asks for independent,
rationale-bearing candidate sets. It normalizes strings like the official
evaluator, collapses only unambiguous aliases observed in `train.jsonl`, and
uses this fixed relation-specific policy:

| Relation | Samples | Aggregation and post-processing |
| --- | ---: | --- |
| `hasArea` | 20 | Median of parseable values in km² |
| `hasCapacity` | 15 | Median of parseable integer-capacity values |
| `personHasCityOfDeath` | 20 | Majority vote with empty/`None` as a first-class candidate |
| `companyTradesAtStockExchange` | 10 | Candidate-support threshold plus train-calibrated empty voting |
| `countryLandBordersCountry` | 3 | Union, one omission-completion pass, then conservative verification |
| `awardWonBy` | 20 | Low-support-threshold union plus one non-destructive omission pass |

The configured thresholds are starting values. Replace them only with results
from the train-only calibration procedure below. There are no hard pre-gates,
cross-model ensembles, or validation-entity rules.

#### Predicting and resuming

Run the fixed 12-row, relation-stratified smoke test first. `--smoke` keeps the
same method but uses one sample per relation and caps each completion at 512
tokens:

```powershell
python akbc.py predict `
  -c configs/improvement-plan.yaml `
  -i data/smoke-stratified.jsonl `
  -o data/model_outputs/smoke.jsonl `
  --runtime-dir data/model_outputs `
  --smoke `
  --resume
```

After checking quality, latency, and the raw responses, run all 478 validation
rows:

```powershell
python akbc.py predict `
  -c configs/improvement-plan.yaml `
  -i data/val.jsonl `
  -o data/model_outputs/val-prompt-json.jsonl `
  --runtime-dir data/model_outputs `
  --resume
```

Prediction reads only `SubjectEntity` and `Relation` from each input row; the
validation answers are never placed in prompts. Use a different output name for
each controlled experiment. To compare constrained decoding with tolerant
prompt-JSON using the same pipeline:

```powershell
python akbc.py predict `
  -c configs/improvement-plan.yaml `
  -i data/val.jsonl `
  -o data/model_outputs/val-json-schema.jsonl `
  --runtime-dir data/model_outputs `
  --output-mode json_schema `
  --resume
```

To compare another downloaded model without creating another configuration,
override both its exact LM Studio identifier and published total size:

```powershell
python akbc.py predict `
  -c configs/improvement-plan.yaml `
  -i data/smoke-stratified.jsonl `
  -o data/model_outputs/smoke-mistral-24b.jsonl `
  --runtime-dir data/model_outputs `
  --smoke `
  --model "exact-lm-studio-model-id" `
  --model-parameters-billion 24 `
  --resume
```

Keep a separate output name for every model and response mode. This prevents
their checkpoints and measurements from being mixed.

For the final 477-row test split:

```powershell
python akbc.py predict `
  -c configs/improvement-plan.yaml `
  -i data/test.jsonl `
  -o data/model_outputs/test.jsonl `
  --runtime-dir data/model_outputs `
  --resume
```

Each completed row is immediately committed as a separate immutable JSON shard
under:

```text
data/model_outputs/<output-name>-<hash>.improvement-plan-v2.state/rows/
```

Repeating the identical command with `--resume` continues at the first missing
row. The manifest rejects changed inputs, model settings, generation settings,
prompt-example content, or prompt/parser/schema policy code. A finished run
also materializes:

- `data/model_outputs/<output-name>.jsonl` — cleaned submission rows;
- `data/model_outputs/raw/<output-name>.raw.json` — prompts, responses,
  rationales, aggregation decisions, and token usage;
- `...state/complete.json` — completed and expected row counts.

Keeping the shards under `data/` avoids the previous synced single-file
checkpoint bottleneck while preserving crash recovery.

#### Evaluation, merging, and submission packaging

Evaluate validation predictions locally:

```powershell
python evaluate.py `
  -g data/val.jsonl `
  -p data/model_outputs/val-prompt-json.jsonl
```

Relation-only experiments can be created without copying gold answers and then
merged into a complete base prediction:

```powershell
python akbc.py select `
  -i data/val.jsonl `
  -o data/model_outputs/award-input.jsonl `
  -r awardWonBy

python akbc.py merge `
  -b data/model_outputs/val-prompt-json.jsonl `
  -p data/model_outputs/award-predictions.jsonl `
  -o data/model_outputs/val-hybrid.jsonl
```

Package validation output with 478 rows:

```powershell
python akbc.py package `
  -i data/model_outputs/val-hybrid.jsonl `
  -o data/model_outputs/validation-submission.zip `
  --expected-rows 478
```

Package final-test output with 477 rows:

```powershell
python akbc.py package `
  -i data/model_outputs/test.jsonl `
  -o data/model_outputs/test-submission.zip `
  --expected-rows 477
```

The package command validates row structure and creates a ZIP containing exactly
one file named `predictions.jsonl` at its root.

#### Train-only threshold calibration

Calibration is offline and never calls a model. First generate complete
leave-one-out train traces for the relations whose thresholds should be
calibrated. The pipeline automatically excludes the target training subject
from its own demonstrations:

```powershell
python akbc.py select `
  -i data/train.jsonl `
  -o data/model_outputs/train-calibration-input.jsonl `
  -r personHasCityOfDeath companyTradesAtStockExchange awardWonBy

python akbc.py predict `
  -c configs/improvement-plan.yaml `
  -i data/model_outputs/train-calibration-input.jsonl `
  -o data/model_outputs/train-calibration.jsonl `
  --runtime-dir data/model_outputs `
  --resume

python akbc.py calibrate `
  --train data/train.jsonl `
  --traces data/model_outputs/raw/train-calibration.raw.json `
  --output data/model_outputs/train-thresholds.yaml `
  --relations personHasCityOfDeath companyTradesAtStockExchange awardWonBy
```

The calibrator maximizes relation macro-F1 with the official evaluator and
writes thresholds plus hashes and provenance. It requires complete train
coverage by default, proves that supplied gold rows are exact subsets of the
bundled `data/train.jsonl`, and never edits the run configuration automatically.
Copy the selected values into
`configs/improvement-plan.yaml` before starting a new locked validation run.
The `--traces` argument can also point to a completed run's `.state/rows/`
directory.

#### Train-derived SyntheticCoT

SyntheticCoT is optional and model-specific. Generate an answer-blanked
all-relation training input, run it with the same chosen model, and retain only
reasoning paths that the offline filter verifies against training gold:

```powershell
python akbc.py select `
  -i data/train.jsonl `
  -o data/model_outputs/train-all-input.jsonl `
  -r hasArea hasCapacity personHasCityOfDeath companyTradesAtStockExchange countryLandBordersCountry awardWonBy

python akbc.py predict `
  -c configs/improvement-plan.yaml `
  -i data/model_outputs/train-all-input.jsonl `
  -o data/model_outputs/train-all-reasoned.jsonl `
  --runtime-dir data/model_outputs `
  --resume

python akbc.py build-cot `
  --train data/train.jsonl `
  --traces data/model_outputs/raw/train-all-reasoned.raw.json `
  --output data/model_outputs/synthetic-cot.jsonl
```

The builder keeps numeric paths within the official 5% tolerance, exact
complete sets for ordinary string relations, and non-empty precision-one award
subsets. Every source row must be copied exactly from `train.jsonl`. It writes
`synthetic-cot.jsonl.manifest.json`, binding the store to the model identifier,
published parameter count, training content, trace source, and artifact bytes.
Uncomment
`synthetic_cot_file` in the configuration to prioritize retained examples while
keeping ordinary training examples as fallback. Start a new output/checkpoint
after changing examples, and never reuse one model's SyntheticCoT store for a
different model.

### How to structure your prediction file

Your prediction file should be in the jsonl format.
Each line of a valid prediction file contains a JSON object which must
contain at least 3 fields to be used by the evaluation script:

- ``SubjectEntity``: the subject entity (string)
- ``Relation``: the relation (string)
- ``ObjectEntities``: the predicted object entity strings (list of strings)

This is an example of how to write a prediction file:

```python
import json

# Dummy predictions
predictions = [
    {
        "SubjectEntity": "Dominican republic",
        "Relation": "countryLandBordersCountry",
        "ObjectEntities": ["Haiti"]
    },
    {
        "SubjectEntity": "Jiaxing Stadium in Jiaxing",
        "Relation": "hasCapacity",
        "ObjectEntities": ["35000"]
    },
    {
        "SubjectEntity": "Mauritius",
        "Relation": "countryLandBordersCountry",
        "ObjectEntities": []
    }

]

fp = "./path/to/your/prediction/file.jsonl"

with open(fp, "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred) + "\n")
```

### Submit your predictions

Codabench submissions must be uploaded as a ZIP archive containing a file
named exactly `predictions.jsonl` at the archive root. Do not merely rename the
ZIP file: the JSONL member inside it must have that exact name. Before upload,
copy the desired generated output to `predictions.jsonl` and ZIP that file.

Submit your system paper via [OpenReview](https://openreview.net/group?id=EMNLP/2026/Workshop/LM-KBC_Shared_Task).

For the validation leaderboard, submit your predictions to [Codabench (validation)](https://www.codabench.org/competitions/16267/).

Upload the packaged 477-row test predictions when the organizers open the
final test leaderboard.

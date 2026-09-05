# Validation leaderboard reference

Snapshot noted on 31 July 2026 from the public AKBC Shared Task 2026
validation leaderboard, as reported by the project author. This is another
team's submission, not a result produced by this repository. Treat it as an
empirical reference for what was demonstrably achievable at that point, and
recheck the live leaderboard before using it in the paper.

| Relation | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|
| `awardWonBy` | 0.3169 | 0.2314 | 0.2338 |
| `companyTradesAtStockExchange` | 0.8466 | 0.8625 | 0.7702 |
| `countryLandBordersCountry` | 0.9839 | 1.0000 | 0.9909 |
| `hasArea` | 0.7300 | 0.7300 | 0.7300 |
| `hasCapacity` | 0.2900 | 0.2900 | 0.2900 |
| `personHasCityOfDeath` | 0.8100 | 0.6600 | 0.6200 |
| **All relations** | **0.7066** | **0.6790** | **0.6501** |
| **Zero-object cases** | **0.7615** | **0.8925** | **0.8218** |

## Targets suggested by the snapshot

- Overall validation macro F1 reference: **0.6501**.
- Zero-object macro F1 reference: **0.8218**; abstention and relation gates
  therefore warrant explicit evaluation.
- Border prediction is close to saturated at **0.9909 F1**.
- Award completion remains the hardest relation at **0.2338 F1**.
- Capacity prediction is another clear improvement opportunity at **0.2900
  F1**, despite being single-valued and numeric.


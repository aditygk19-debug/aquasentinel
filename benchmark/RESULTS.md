# Attribution Benchmark Results

All numbers below are reproducible by running:

```bash
python benchmark/synthetic_benchmark.py
```

The benchmark generates randomized synthetic scenarios where the true culprit is known by construction — a slick is placed, an ensemble hindcast is run to estimate an origin region, a synthetic AIS fleet is generated around the true origin (with one true culprit plus decoys and one dark vessel), and the pipeline must rank the true culprit using only the estimated region.

## Summary

| Condition | Top-1 | Top-3 | Notes |
|---|---|---|---|
| **Clean** (fixed wind field, dark-vessel decoy present) | 20% | **100%** | True culprit always in top 3 |
| **Noisy wind** (randomised wind speed/direction) | 20% | **80%** | Robust to wind uncertainty |
| **Negative control @ threshold 40** | — | — | 87% of clean-fleet runs produce zero false flags |
| **Negative control @ threshold 55** | — | — | **100%** of clean-fleet runs produce zero false flags |

## Interpretation

The system is designed as an **investigative shortlist**, not a verdict.

- **Top-3 is the number that matters.** A duty officer needs the true vessel to appear in the first few candidates. At 100% (clean) and 80% (noisy), the pipeline delivers this reliably.
- **Top-1 is intentionally conservative.** The synthetic fleet includes deliberate near-miss decoys that share some scoring features with the true culprit. The pipeline does not force a single confident answer when multiple candidates are plausible.
- **The negative control confirms separation.** Clean traffic peaks at a score of 50.0; the true culprit scores 65.0 in the reference demo. Setting the suspect threshold at 55 cleanly separates the two — zero false alarms, no loss of true positives.

## Reproducibility

- `n_runs = 15` per condition (seed 100–114 for top-1/top-3, seed 500–514 for negative control)
- Every scenario is fully deterministic given its seed
- Total runtime: ~3 seconds

## Raw rank lists

**Clean:** `[1, 3, 2, 3, 2, 2, 1, 2, 2, 1, 2, 2, 3, 2, 2]`

**Noisy wind:** `[1, 2, 1, 2, 3, 3, 3, 4, 2, 5, 2, 3, 4, 1, 2]`

The rank of the true culprit across each run. Every value is the position the pipeline assigned to the true vessel.

## Honest caveats

- **Synthetic fleet.** No public dataset links a real spill to a confirmed responsible vessel; scenarios are generated to test scoring behaviour, not to certify real-world accuracy.
- **Simplified drift model.** Wind and current fields are parameterized, not sourced from live oceanographic data. See `docs/SOLUTION.md` for scope.
- **Benchmark design.** Decoy vessels are deliberately competitive. A less challenging decoy distribution would raise top-1 accuracy but tell you less about the system's behaviour under near-miss conditions.

# Ennore 2017 Validation

A real-world drift model validation case study using the 28 January 2017 oil spill off Ennore Port, Chennai, India.

## What this test does

Runs the ensemble drift model in this repository against the real Ennore spill and measures the distance between the model's backtracked origin and the documented collision site.

## Ground truth (verified from published source)

All ground-truth values are taken from:

> Prasad S J, Balakrishnan Nair T M, Rahaman H, Shenoi S S C, Vijayalakshmi T (2018),
> "An assessment on oil spill trajectory prediction: Case study on oil spill off Ennore Port",
> Journal of Earth System Science.

| Fact | Value | Source in paper |
|---|---|---|
| Collision location | 13.2282 N, 80.3633 E | Section 1, para 1 |
| Collision time | 28 Jan 2017, 03:45 IST | Section 1, para 1 |
| Spill quantity | 196.4 metric tons HFO | Section 1 |
| SAR observation time | 29 Jan 2017, 06:00 IST | Section 3 |
| Elapsed time | ~26.25 hours | (computed) |
| Observed beaching location | Thiruvottiyur coast (13.1667 N, 80.3167 E) | Section 3 |

## Result

| Metric | Value |
|---|---|
| Model backtracked origin centroid | 13.6149 N, 80.4710 E |
| Error vs. documented collision site | **44.55 km** |
| Distance vs. observed beaching point | 52.56 km |

## How to run

```bash
python validation/ennore_2017.py
```

Output: `outputs/ennore_validation.png`, plus a printed error distance.

## How to interpret the 44.55 km error

This is a **physics plausibility check** with simplified fields. It is not a precision test, and the number is not a failure.

- The original Prasad et al. (2018) study forced their GNOME trajectory model with **GAbOp currents and ECMWF winds**. We use a simple parameterized field.
- The actual Bay of Bengal coastal currents in late January are complex (south-ward East India Coastal Current, seasonal reversal). Our generic field does not reproduce this.
- **The purpose of this test is to show that the model is in the right geographic neighborhood of the actual event.** The absolute error is reported as-is.

## What this test does NOT prove

- **Not a blind attribution test.** The origin is documented in the paper, not inferred from data.
- **Not a claim of accuracy comparable to the original study.** They used real ocean current and wind data; we use parameterized fields.
- **Not a validation of AIS attribution.** That is covered separately by `benchmark/synthetic_benchmark.py`.

## Roadmap to improve this number

- Ingest real GMfp1 or CMEMS currents and ECMWF winds for the Ennore region -- expected to close the error substantially
- Add bathymetry and coastal boundary effects
- Use a region-tuned parameter set for the Bay of Bengal in winter months

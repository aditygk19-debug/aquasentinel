# AquaSentinel — Gaps Addressed & Innovations

This document states plainly what AquaSentinel covers, what it partially covers, and what it deliberately scoped out for this round. It exists so that evaluators, mentors, and reviewers can hold the project to its own claims.

---

## Gaps in Existing Solutions — and How AquaSentinel Addresses Them

| # | Gap in existing tools | AquaSentinel's answer | Coverage |
|---|---|---|---|
| 1 | **Slick geometry assumed rather than measured.** Many pipelines match detected slicks against idealized shapes rather than modeling how oil actually moves through water. | Our attribution does not depend on slick shape at all. Suspect scoring is based on physical drift constraints (ensemble hindcast from the detected slick) plus vessel spatio-temporal behaviour. | **Covered** |
| 2 | **Matching is geometric, not physics-based.** The origin is often the closest point on a straight line, not a location consistent with real advection. | 25-member perturbed drift ensemble integrated backwards in time. Wind, current, and their uncertainty are all modeled explicitly. Produces an origin **probability region**, not a single pin. | **Covered** |
| 3 | **Uncertainty is not quantified for the user.** The output is a coordinate; no confidence is attached. | Every origin estimate comes with a 68% confidence radius derived from the ensemble spread. The user sees a circle, not a point. | **Covered** |
| 4 | **Dark vessels are an afterthought.** Vessels that switch off AIS are usually dropped from the candidate list, which is exactly when they matter most. | Vessels with an AIS gap spanning the release window are explicitly **flagged as dark-vessel candidates**, not silently excluded. They remain in the ranked list with a visible marker. | **Partial** — AIS-gap flagging is built and working. SAR-based detection of vessels with no AIS match at all is roadmap. |
| 5 | **Evidence is not usable in an investigation.** Ranking output is often a name and a number, with no explanation of *why*. | Every ranked vessel carries a list of human-readable **reason codes** — "within 2.6 km of origin region," "AIS gap spanning the release window," "trajectory consistent with moving away from origin." Output is a ranked CSV an investigator can audit. | **Partial** — reason codes and CSV are built. A formatted evidence dossier is roadmap. |

---

## Innovations

| # | Innovation | What it does | Status |
|---|---|---|---|
| 1 | **Probabilistic origin estimation** | Instead of one backtracked trajectory, we run a 25-member perturbed ensemble and report the origin as a confidence region. This is the project's core differentiator. | **Fully built and verified** |
| 2 | **Dark-vessel-aware attribution** | Vessels with AIS gaps during the release window are flagged, not dropped, and carry an explicit dark-vessel marker through the ranking pipeline. | **Partial** — AIS-gap flagging built; SAR-based detection of fully-dark vessels is roadmap |
| 3 | **Explainable evidence per suspect** | Every ranked vessel carries reason codes that state exactly which scoring factors fired. No black-box ranking. | **Partial** — reason codes and CSV built; formatted dossier is roadmap |
| 4 | **Honest uncertainty and sensitivity reporting** | Confidence radius on every origin estimate; multi-condition benchmark (clean, noisy-wind, fully-dark vessel, negative control) reported side-by-side rather than cherry-picked. | **In progress** — will be complete once benchmark script is executed |
| 5 | **Synthetic scenario benchmark** | Repeatable top-1 / top-3 attribution accuracy across many randomized scenarios, so accuracy claims are testable rather than anecdotal. | **Built** — will be reported once executed |
| 6 | **Full pipeline transparency** | Detection weights, drift parameters, scoring weights, and benchmark harness are all in the repository. Nothing is hidden behind an API or a hosted service. | **Fully built** |

---

## Scope Decisions

AquaSentinel is a focused prototype. We built the pipeline end-to-end and made deliberate choices about what to include in this round. The following items were scoped out on purpose, not overlooked:

- **Real-time alerting** — out of scope for round 2. The pipeline is designed to run against whatever AIS and imagery is available, rather than optimizing time-to-alert.
- **Human-in-the-loop operator workflow** — no formal review process is built. The ranked CSV and reason codes are the interface.
- **Formal India-specific incident certification** — the system runs on a documented real incident for demonstration; formal certification against a specific historical case is future work.
- **Look-alike filtering (algae, natural seeps)** — roadmap.
- **Segmentation-grade slick geometry** — the current detector is a tile classifier producing an approximate bounding region. A segmentation head is on the roadmap.

Every choice above is stated so that reviewers can hold the project to its own claims.
- **Grad-CAM as a shipped feature** — out of scope. Attention heatmaps were generated during model development to sanity-check the detector, but they are not part of the end-to-end pipeline (detection → drift → attribution). Adding a Grad-CAM overlay to the dashboard is roadmap.

---

## Roadmap (future work)

- Continuous-learning look-alike filter
- Multi-pass slick tracking across satellite revisit cycles
- Slick age estimation from SAR imagery
- SAR-based ship detection for fully-dark vessels
- Live AIS ingest (AISStream, MarineTraffic)
- GeoTIFF-native geolocation (currently linear pixel → lat/lon)
- Segmentation head for pixel-precise slick geometry

---

## One-Line Summary

> AquaSentinel covers the physics-vs-geometry gap and the confidence-calibration gap fully, the dark-vessel and evidence-usability gaps partially, and states its scope decisions openly rather than hiding them.

---

## Repository Discipline

Every claim above is backed by code or a number that a reviewer can verify in this repository:

- Detection: `detect/detect_model.py`, checkpoint in `detect/checkpoints/`
- Drift ensemble: `drift/ensemble_drift.py`
- Attribution and reason codes: `ais/ais_attribution.py`
- Benchmark: `benchmark/synthetic_benchmark.py`
- End-to-end demo: `run_demo.py`
- Outputs: `outputs/`

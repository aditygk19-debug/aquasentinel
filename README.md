# AquaSentinel — SIH26143

Satellite-based marine oil spill detection with AIS-based vessel attribution.
Built for Smart India Hackathon 2026, problem statement **SIH26143** (NTRO — Disaster Management).

## What this repo contains

A working prototype of the full detect → hindcast → attribute pipeline:

1. **Detection** (`detect/`) — `detect_model.py` rebuilds the exact trained
   architecture (EfficientNet-B0, first conv adapted to 1-channel input,
   96×96 tiles), matching the model already trained on the Zenodo Deep-SAR
   dataset (99.01% accuracy, with a documented 34.42% cross-dataset result
   as an honest domain-shift finding). It adds a **sliding-window scan**
   over a larger scene to turn whole-image classification into an
   approximate bounding region (polygon + area), without needing a full
   segmentation model under a 2-day deadline.

   To use your real trained weights:
   - Copy your checkpoint to `detect/checkpoints/efficientnet_zenodo_best.pt`
   - Copy a sample SAR scene to `data/sample_scene.png` and set its real
     geographic bounds in `SCENE_BOUNDS` at the top of `run_demo.py`
   - Run `python3 run_demo.py` — it auto-detects both files and switches
     from the fallback location to the real detector output. If either is
     missing, it clearly logs that and falls back so the rest of the
     pipeline can still be demonstrated.
2. **Ensemble drift hindcast** (`drift/`) — instead of a single deterministic
   backtrack, we run an ensemble of Lagrangian particle simulations (RK2
   advection) with randomly perturbed wind/current conditions. The spread of
   backtracked origins becomes a **probability region**, not a single guessed
   point. See `drift/ensemble_drift.py` and `drift/plot_drift.py`.
3. **AIS attribution with dark-vessel flagging** (`ais/`) — candidate vessels
   are scored against the origin probability region (proximity, temporal
   overlap, trajectory alignment, vessel type), and vessels with an AIS gap
   spanning the release window are explicitly flagged as dark-vessel
   candidates rather than silently dropped. Every score comes with
   human-readable reason codes. See `ais/ais_attribution.py`.
4. **Synthetic attribution benchmark** (`benchmark/`) — since no public
   dataset links real slicks to real polluting vessels, we measure
   performance on synthetic scenarios with a known planted culprit, and
   report top-1 / top-3 accuracy under varying conditions. See
   `benchmark/synthetic_benchmark.py`.

## Why this design

Operational and published systems (EMSA CleanSeaNet, SkyTruth Cerulean, KSAT)
already do satellite-based oil spill detection and AIS correlation. We are not
claiming to reinvent detection. Our focus is on gaps those systems document
about themselves:

- Cerulean states its attributions are only as good as the AIS supplied, and
  that the true polluter may be a "dark" vessel not broadcasting — our
  pipeline explicitly detects and flags this case instead of ignoring it.
- Existing public methods report point/geometric matches; we report a
  **calibrated probability region** from ensemble hindcasting, with honest
  uncertainty.
- We could not find a public attribution accuracy benchmark; ours is a small,
  reproducible one built for this purpose.

Wind/current fields in the drift model are currently simplified and clearly
labelled as simulated (not a live Copernicus Marine / ERA5 pull) — this was a
deliberate scope decision under a hard deadline. The interface
(`sample_current()` / `sample_wind()` in `drift/ensemble_drift.py`) is built
so real gridded data can be substituted without changing the rest of the
pipeline.

## Quickstart

```bash
pip install -r requirements.txt
python3 run_demo.py
```

This runs the full pipeline end-to-end on a representative scenario and
writes to `outputs/`:
- `hindcast_ensemble.png` — ensemble backtrack paths + origin probability region
- `suspect_ranking.png` — ranked vessel suspects with scores
- `suspect_ranking.csv` — full ranked table with reason codes

To run the benchmark:
```bash
python3 benchmark/synthetic_benchmark.py
```

## Repo structure

```
aquasentinel/
├── detect/          # CNN detection model + training notebook
├── drift/           # ensemble hindcast/forecast (Innovation 1)
├── ais/             # AIS attribution + dark-vessel flagging (Innovation 2)
├── benchmark/        # synthetic top-k accuracy benchmark (Innovation 3)
├── outputs/          # generated plots/tables from run_demo.py
├── run_demo.py        # end-to-end pipeline entry point
└── README.md
```

## Team

[Add team name / members here]

## References

- EMSA CleanSeaNet — https://www.emsa.europa.eu/csn-menu.html
- SkyTruth Cerulean — https://skytruth.org/cerulean
- INCOIS Online Oil Spill Advisory System — https://incois.gov.in/
- Zenodo Deep-SAR Oil Spill dataset — DOI 10.5281/zenodo.15298010

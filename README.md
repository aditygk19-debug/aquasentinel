# AquaSentinel

**Marine oil spill detection → drift hindcast → AIS-based vessel attribution.**

A hackathon prototype (SIH26143, round 2) that goes beyond *where is the slick?* to answer *where did it come from, and which vessel most likely caused it?* — with uncertainty quantified and every decision explainable.

---

## What makes this different

| Gap in existing tools | AquaSentinel's answer | Where in the repo |
|---|---|---|
| **Single-point origin guess.** Drift models run once with nominal winds/currents → one pin, no uncertainty. | **25-member perturbed ensemble hindcast** → a **probability region** (68% confidence circle) instead of a single pin. | `drift/ensemble_drift.py`, `drift/plot_drift.py` |
| **Detection ≠ attribution.** SAR spots a slick; no tool names who did it. | **Weighted AIS scoring** with **reason codes** + explicit **dark-vessel (AIS-gap) flagging**. | `ais/ais_attribution.py` |
| **No repeatable accuracy claims.** Most demos show one pretty picture and stop. | **Synthetic benchmark** over many seeds → top-1 / top-3 attribution accuracy. | `benchmark/synthetic_benchmark.py` |
| **Black-box scoring.** "Our AI says vessel X." | Every ranked vessel carries **human-readable reasons** — no black box. | `reason_codes` column in `outputs/suspect_ranking.csv` |
| **Detection only.** | Sliding-window **CNN detector** localises the slick on a SAR scene. | `detect/detect_model.py` |

> Existing pipelines answer *where is the slick?*  
> AquaSentinel answers *where did it come from, and who most likely spilled it — and why.*

---

## Quickstart

### Option A -- Google Colab (recommended)

```python
!git clone https://github.com/aditygk19-debug/aquasentinel.git
%cd aquasentinel
!`edit 's/pytorch-grad-cam>=1.4/grad-cam>=1.4/' requirements.txt
!pip install -q -r requirements.txt
!python run_demo.py
```

The pipeline falls back gracefully if `data/sample_scene.png` is missing, but to see a real detection, drop any SAR PNG at `data/sample_scene.png` first.

### Option B -- Local

```bash
git clone https://github.com/aditygk19-debug/aquasentinel.git
cd aquasentinel
pip install -r requirements.txt
python run_demo.py
```

**Outputs written to `outputs/`:**
- `hindcast_ensemble.png` — 25-member drift fan + probability region
-  suspect_ranking.png` — ranked suspects (crimson = ground truth, orange = dark vessel)
- `suspect_ranking.csv` — score + reason codes per vessel

---

## What the demo prints

```
Loaded trained weights from detect/checkpoints/efficientnet_zenodo_best.pt
[1/4] REAL detection: slick at (19.100, 72.800), est. area 1048.16 km^2, grid confidence max=1.00
[2/4] Ensemble hindcast: origin region centroid=(19.438, 72.951), radius_km_68pct=8.0 km, n_members=25
[3/4] Scored 8 vessels against origin region (synthetic AIS fleet, incl. 1 vessel with an AIS gap)
[4/4] Wrote ranked suspect table -> outputs/suspect_ranking.csv
[NOTE] True culprit MMSI (for this demo run): TRUE-620462 -- rank #1
```

**The true culprit ranks #1.** That's the whole point.

---

## Repo layout

```
aquasentinel/
Ô ── detect/           # EfficientNet-B0 sliding-window detector (single-channel SAR)
    ├── checkpoints/  # trained weights (efficientnet_zenodo_best.pt, ~16 MB)
 ── drift/             # ensemble hindcast + origin probability region    ├── ais/               # AIS attribution: weighted scoring + dark-vessel flagging
  ── benchmark/         # synthetic benchmark over many seeds → top-1/top-3 accuracy
  ── data/              # drop a SAR scene here as data/sample_scene.png
  ── outputs/           # pipeline outputs (PNG + CSV)
  ── run_demo.py       # end-to-end demo
  ── requirements.txt
```

---

## The trained detector

- **Architecture:** EfficientNet-B0, first conv adapted to **1-channel (grayscale SAR)**
- **Input:** 96×96 tiles
- **Classes:** 2 (oil / no-oil)
- **Parameters:** 4,009,534
- **Training data:** Zenodo Deep-SAR dataset
- **Reported tile accuracy:** 99.01%

**Checkpoint compatibility note:** the training script saved weights with a `backbone.` prefix on every parameter (the EfficientNet was wrapped in a container module). `detect/detect_model.py::load_model()` strips this prefix before calling `load_state_dict()`. Without this, loading fails silently with 360 missing keys. This is handled transparently — you don't need to do anything.

---

## Known limitations (honest scope)

This is a **prototype**, not a production system. Specifically:

- **AIS feed is synthetic.** `ais/ais_attribution.py` generates a fleet around a spill origin for reproducibility. The scoring logic is real; the input data is not.
- **Detector is a tile classifier, not a segmentation model.** It slides a 96×96 window and classifies each tile, producing an approximate bounding region — not pixel-precise slick geometry.
- **SAR scene used in the demo is a PALSAR coastal image** without a confirmed oil slick. The detector fires on dark smooth regions; on this particular scene that yields a wide bounding box. Replace with a real confirmed slick scene for a sharper result.
- **Geolocation is linear pixel → lat/lon**, not a proper GeoTIFF affine transform.

### Roadmap

- Live AIS ingest (AISStream, MarineTraffic API)
- GeoTIFF-native geolocation
- Segmentation head for pixel-precise slick geometry
- Wind/current data from CMEMS instead of static params

---

## Repo

**https://github.com/aditygk19-debug/aquasentinel**

## License

Hackathon prototype — provided as-is for evaluation and demonstration.

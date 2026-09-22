"""
AquaSentinel — end-to-end demo pipeline (SIH26143, round 2)

Detected slick (from the trained CNN, see detect/) -> ensemble hindcast
(drift/) -> AIS attribution with dark-vessel flagging (ais/) -> ranked
suspect table + evidence reasoning.

Run: python3 run_demo.py
Outputs: outputs/hindcast_ensemble.png, outputs/suspect_ranking.csv,
         outputs/suspect_ranking.png
"""

import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "drift"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ais"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detect"))

from ensemble_drift import FieldParams, run_ensemble_hindcast, origin_probability_region  # noqa: E402
from plot_drift import plot_hindcast_ensemble  # noqa: E402
from ais_attribution import generate_synthetic_fleet, rank_vessels  # noqa: E402

import matplotlib.pyplot as plt


# --- Config: point these at your real files once available ---
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "detect", "checkpoints",
                                "efficientnet_zenodo_best.pt")
SCENE_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_scene.png")
# Geographic bounds of the sample scene (lat_min, lat_max, lon_min, lon_max) --
# replace with the real SAR product's footprint when using an actual scene.
SCENE_BOUNDS = (18.85, 19.25, 72.65, 73.05)


def run_detection_stage():
    """Runs the real CNN detector on a scene if a checkpoint + scene image are
    available; otherwise falls back to a representative detection so the rest
    of the pipeline (drift + AIS) can still be demonstrated end-to-end."""
    have_checkpoint = os.path.exists(CHECKPOINT_PATH)
    have_scene = os.path.exists(SCENE_PATH)

    if have_checkpoint and have_scene:
        from detect_model import load_model, detect_slick_region, bbox_px_to_geo, estimate_area_km2
        model = load_model(CHECKPOINT_PATH)
        confidence_grid, oil_mask, bbox_px, scene_size = detect_slick_region(
            model, SCENE_PATH, threshold=0.5)
        if bbox_px is None:
            print("[1/4] Detector found no oil-classified tiles above threshold in this scene.")
            return None
        geo = bbox_px_to_geo(bbox_px, scene_size, SCENE_BOUNDS)
        area_km2 = estimate_area_km2(bbox_px, scene_size, SCENE_BOUNDS)
        print(f"[1/4] REAL detection: slick at ({geo['centroid_lat']:.3f}, "
              f"{geo['centroid_lon']:.3f}), est. area {area_km2:.2f} km^2, "
              f"grid confidence max={confidence_grid.max():.2f}")
        return geo["centroid_lat"], geo["centroid_lon"]
    else:
        missing = []
        if not have_checkpoint:
            missing.append(f"checkpoint ({CHECKPOINT_PATH})")
        if not have_scene:
            missing.append(f"scene image ({SCENE_PATH})")
        print(f"[1/4] [FALLBACK] Missing: {', '.join(missing)}. "
              f"Using a representative detected-slick location so the drift+AIS "
              f"stages can still run end-to-end. Drop your trained .pt and a "
              f"sample scene at the paths above to use the real detector.")
        return 19.05, 72.85


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # --- Step 1: detection (real model if available, else representative fallback) ---
    detected = run_detection_stage()
    slick_lat, slick_lon = detected if detected else (19.05, 72.85)
    hours_since_release_est = 18  # from slick age heuristic

    # --- Step 2: ensemble hindcast ---
    base_params = FieldParams(current_speed=0.4, current_dir_deg=200,
                               wind_speed=6.0, wind_dir_deg=210)
    origins, paths = run_ensemble_hindcast(slick_lat, slick_lon, hours=hours_since_release_est,
                                            base_params=base_params, n_members=25, seed=42)
    region = origin_probability_region(origins)
    radius_key = [k for k in region if k.startswith("radius_km")][0]
    radius_km = region[radius_key]
    print(f"[2/4] Ensemble hindcast: origin region centroid=({region['centroid_lat']:.3f}, "
          f"{region['centroid_lon']:.3f}), {radius_key}={radius_km:.1f} km, "
          f"n_members={region['n_members']}")

    plot_hindcast_ensemble(slick_lat, slick_lon, origins, paths, region,
                            save_path=os.path.join(out_dir, "hindcast_ensemble.png"))

    # --- Step 3: AIS attribution (synthetic fleet around origin, incl. a dark vessel) ---
    rng = np.random.default_rng(7)
    fleet, true_mmsi = generate_synthetic_fleet(region["centroid_lat"], region["centroid_lon"],
                                                 origin_time_h=0.0, rng=rng, n_decoys=6,
                                                 gap_vessel=True)
    ranked = rank_vessels(fleet, region["centroid_lat"], region["centroid_lon"],
                           origin_time_h=0.0, radius_km=radius_km)
    print(f"[3/4] Scored {len(ranked)} vessels against origin region "
          f"(synthetic AIS fleet, incl. 1 vessel with an AIS gap)")

    # --- Step 4: outputs -- ranked table (CSV) + suspect bar chart ---
    csv_path = os.path.join(out_dir, "suspect_ranking.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "mmsi", "name", "vessel_type", "score",
                          "dark_vessel_flag", "reason_codes"])
        for i, r in enumerate(ranked, 1):
            writer.writerow([i, r["mmsi"], r["name"], r["vessel_type"], r["score"],
                              r["dark_vessel_flag"], " | ".join(r["reasons"])])
    print(f"[4/4] Wrote ranked suspect table -> {csv_path}")

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(figsize=(11, 7))
    names = [f"{r['mmsi']}\n({r['vessel_type']}{', DARK' if r['dark_vessel_flag'] else ''})"
              for r in ranked]
    scores = [r["score"] for r in ranked]
    colors = ["crimson" if r["mmsi"] == true_mmsi else ("darkorange" if r["dark_vessel_flag"] else "steelblue")
              for r in ranked]
    ax.barh(names[::-1], scores[::-1], color=colors[::-1], height=0.72)

    # annotate the true culprit
    for i, r in enumerate(ranked):
        if r["mmsi"] == true_mmsi:
            y_pos = len(ranked) - 1 - i
            ax.annotate("← matches ground truth", xy=(r["score"], y_pos),
                        xytext=(r["score"] + 2, y_pos),
                        fontsize=11, color="crimson", va="center", fontweight="bold")
            break

    # legend explaining the colors
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="crimson", label="Ground-truth culprit"),
        Patch(color="darkorange", label="Dark vessel (AIS gap)"),
        Patch(color="steelblue", label="Normal AIS traffic"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    ax.set_xlabel("Suspicion score (0–100)")
    ax.set_title("Ranked Vessel Suspects — Weighted Evidence Scoring\n"
                 "(proximity + timing + trajectory + AIS gap + vessel type)")
    ax.set_xlim(0, max(scores) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "suspect_ranking.png"), dpi=150)
    print(f"       Wrote suspect ranking chart -> {os.path.join(out_dir, 'suspect_ranking.png')}")

    print(f"\n[NOTE] True culprit MMSI (for this demo run): {true_mmsi} "
          f"-- rank #{[i for i,r in enumerate(ranked,1) if r['mmsi']==true_mmsi][0]}")


if __name__ == "__main__":
    main()

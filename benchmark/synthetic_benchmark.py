"""
Synthetic attribution benchmark for AquaSentinel.

Innovation 3: since there is no public ground-truth dataset linking real
slicks to real polluting vessels, we measure attribution performance on
synthetic scenarios where the true culprit is known by construction.
For each run: place a slick, run the ensemble hindcast to get an origin
region, generate a synthetic AIS fleet around that origin (with one true
culprit + decoys, and optionally a dark vessel), then check whether the
true culprit is ranked #1 / in the top 3.

This produces an honest, reproducible number for the deck instead of an
unverifiable accuracy claim.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "drift"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ais"))

from ensemble_drift import FieldParams, run_ensemble_hindcast, origin_probability_region  # noqa: E402
from ais_attribution import generate_synthetic_fleet, rank_vessels  # noqa: E402


def run_one_scenario(seed, hours=18, wind_noise=True, gap_vessel=True):
    rng = np.random.default_rng(seed)

    # Randomise "true" slick location and simple field params per scenario
    lat_start = 19.0 + rng.uniform(-0.3, 0.3)
    lon_start = 72.8 + rng.uniform(-0.3, 0.3)
    base = FieldParams(
        current_speed=rng.uniform(0.2, 0.6),
        current_dir_deg=rng.uniform(0, 360),
        wind_speed=rng.uniform(3, 10) if wind_noise else 6.0,
        wind_dir_deg=rng.uniform(0, 360),
    )

    # Step 1: ensemble hindcast -> origin probability region
    origins, _ = run_ensemble_hindcast(lat_start, lon_start, hours=hours, base_params=base,
                                        n_members=20, seed=int(rng.integers(0, 1e6)))
    region = origin_probability_region(origins)
    radius_key = [k for k in region if k.startswith("radius_km")][0]
    radius_km = max(region[radius_key], 3.0)  # floor so scoring isn't degenerate

    # Step 2: build synthetic fleet AROUND THE TRUE ORIGIN (lat_start/lon_start),
    # which the pipeline does not get to see directly -- it only sees `region`.
    fleet, true_mmsi = generate_synthetic_fleet(lat_start, lon_start, origin_time_h=0.0,
                                                 rng=rng, n_decoys=rng.integers(4, 9),
                                                 gap_vessel=gap_vessel)

    # Step 3: score fleet against the ESTIMATED region (centroid + radius), not the true origin
    ranked = rank_vessels(fleet, region["centroid_lat"], region["centroid_lon"],
                           origin_time_h=0.0, radius_km=radius_km)

    rank_of_true = next((i + 1 for i, r in enumerate(ranked) if r["mmsi"] == true_mmsi), None)
    return rank_of_true, len(fleet)


def run_benchmark(n_runs=15, seed0=100, **scenario_kwargs):
    top1, top3, ranks = 0, 0, []
    for i in range(n_runs):
        rank, fleet_size = run_one_scenario(seed=seed0 + i, **scenario_kwargs)
        ranks.append(rank)
        if rank == 1:
            top1 += 1
        if rank is not None and rank <= 3:
            top3 += 1
    return {
        "n_runs": n_runs,
        "top1_accuracy": top1 / n_runs,
        "top3_accuracy": top3 / n_runs,
        "ranks": ranks,
    }


if __name__ == "__main__":
    print("=== Clean conditions (moderate wind noise, with dark-vessel decoy) ===")
    res = run_benchmark(n_runs=15, wind_noise=False, gap_vessel=True)
    print(res)
    print(f"Top-1 accuracy: {res['top1_accuracy']*100:.0f}%  |  Top-3 accuracy: {res['top3_accuracy']*100:.0f}%\n")

    print("=== Noisy wind conditions (harder hindcast) ===")
    res_noisy = run_benchmark(n_runs=15, wind_noise=True, gap_vessel=True)
    print(res_noisy)
    print(f"Top-1 accuracy: {res_noisy['top1_accuracy']*100:.0f}%  |  Top-3 accuracy: {res_noisy['top3_accuracy']*100:.0f}%")

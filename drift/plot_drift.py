"""Visualise ensemble hindcast: slick location, backtracked paths, and the
resulting origin probability region (used in deck/video screenshots)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def plot_hindcast_ensemble(lat_start, lon_start, origins, paths, region, save_path=None,
                            title="Ensemble Hindcast — Origin Probability Region"):
    fig, ax = plt.subplots(figsize=(8, 7))

    # all backtracked paths, faint
    for path in paths:
        lats = [p[0] for p in path]
        lons = [p[1] for p in path]
        ax.plot(lons, lats, color="steelblue", alpha=0.25, linewidth=1)

    # ensemble origin scatter
    o_lats = [o[0] for o in origins]
    o_lons = [o[1] for o in origins]
    ax.scatter(o_lons, o_lats, color="crimson", s=25, zorder=5, label="Ensemble backtracked origins")

    # detected slick location
    ax.scatter([lon_start], [lat_start], color="black", marker="*", s=250, zorder=6,
               label="Detected slick (start point)")

    # probability region centroid + radius circle (approx, in degrees using rough km->deg)
    lat_c, lon_c = region["centroid_lat"], region["centroid_lon"]
    radius_key = [k for k in region if k.startswith("radius_km")][0]
    radius_km = region[radius_key]
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat_c))
    circle = Circle((lon_c, lat_c), radius_km / km_per_deg_lon, fill=True, alpha=0.15,
                     color="crimson", label=f"{radius_key.replace('radius_km_', '')} confidence region")
    ax.add_patch(circle)
    ax.scatter([lon_c], [lat_c], color="darkred", marker="x", s=100, zorder=6,
               label="Origin estimate (centroid)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ensemble_drift import FieldParams, run_ensemble_hindcast, origin_probability_region

    base = FieldParams()
    lat_start, lon_start = 19.05, 72.85  # off Mumbai, illustrative
    origins, paths = run_ensemble_hindcast(lat_start, lon_start, hours=18, base_params=base, n_members=25)
    region = origin_probability_region(origins)
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    plot_hindcast_ensemble(lat_start, lon_start, origins, paths, region,
                            save_path=os.path.join(out_dir, "hindcast_ensemble.png"))

"""
Ensemble drift (hindcast/forecast) module for AquaSentinel.

Innovation 1: instead of a single deterministic backtrack, we run an
ensemble of Lagrangian particle simulations with randomly perturbed
wind and current conditions. The spread of backtracked origins forms
a spatial probability region (not a single point), which is what the
AIS scoring stage consumes.

Wind/current fields here are simplified (constant + small spatial
gradient), clearly labelled as simulated. This keeps the method
demonstrable without depending on a live Copernicus Marine / ERA5
data pull, which is the failure-prone part under a tight deadline.
Swapping in real gridded fields later only requires changing
`sample_current()` / `sample_wind()`.
"""

import numpy as np
from dataclasses import dataclass


EARTH_R = 6371000.0  # m


def latlon_to_local_xy(lat0, lon0, lat, lon):
    """Small-area equirectangular projection, good enough for <100km scales."""
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = dlon * EARTH_R * np.cos(np.radians(lat0))
    y = dlat * EARTH_R
    return x, y


def local_xy_to_latlon(lat0, lon0, x, y):
    dlat = y / EARTH_R
    dlon = x / (EARTH_R * np.cos(np.radians(lat0)))
    lat = lat0 + np.degrees(dlat)
    lon = lon0 + np.degrees(dlon)
    return lat, lon


@dataclass
class FieldParams:
    """Simplified, simulated ocean current + wind field.
    current_speed/dir and wind_speed/dir are the base (unperturbed) values;
    a small spatial gradient is added so the field isn't perfectly uniform.
    """
    current_speed: float = 0.4      # m/s
    current_dir_deg: float = 200.0  # degrees, direction current flows TOWARD (met convention: from North=0, CW)
    wind_speed: float = 6.0         # m/s
    wind_dir_deg: float = 210.0     # degrees, direction wind blows TOWARD
    windage: float = 0.03           # fraction of wind speed added to drift (typical 1-4%)
    spatial_gradient: float = 0.05  # fractional change in speed per km from origin


def _dir_to_uv(speed, dir_deg):
    """Convert speed + compass direction (toward) into u (east), v (north) components."""
    theta = np.radians(dir_deg)
    u = speed * np.sin(theta)
    v = speed * np.cos(theta)
    return u, v


def sample_current(x, y, x0, y0, params: FieldParams):
    dist_km = np.hypot(x - x0, y - y0) / 1000.0
    speed = params.current_speed * (1.0 + params.spatial_gradient * dist_km * 0.1)
    return _dir_to_uv(speed, params.current_dir_deg)


def sample_wind(x, y, x0, y0, params: FieldParams):
    dist_km = np.hypot(x - x0, y - y0) / 1000.0
    speed = params.wind_speed * (1.0 + params.spatial_gradient * dist_km * 0.1)
    return _dir_to_uv(speed, params.wind_dir_deg)


def advect_rk2(x, y, dt_s, direction, params: FieldParams, x0, y0):
    """One RK2 (midpoint) step of Lagrangian advection.
    direction: +1 forward in time, -1 backward (hindcast)."""
    def velocity(px, py):
        cu, cv = sample_current(px, py, x0, y0, params)
        wu, wv = sample_wind(px, py, x0, y0, params)
        return cu + params.windage * wu, cv + params.windage * wv

    dt = dt_s * direction
    k1u, k1v = velocity(x, y)
    mx, my = x + 0.5 * dt * k1u, y + 0.5 * dt * k1v
    k2u, k2v = velocity(mx, my)
    return x + dt * k2u, y + dt * k2v


def run_single_trajectory(lat_start, lon_start, hours, direction, params: FieldParams,
                           dt_minutes=15):
    """Run one particle trajectory. direction=-1 for hindcast (backward), +1 for forecast."""
    lat0, lon0 = lat_start, lon_start
    x, y = 0.0, 0.0  # local coords, origin at start point
    dt_s = dt_minutes * 60
    n_steps = int((hours * 3600) / dt_s)
    path = [(lat_start, lon_start)]
    for _ in range(n_steps):
        x, y = advect_rk2(x, y, dt_s, direction, params, 0.0, 0.0)
        lat, lon = local_xy_to_latlon(lat0, lon0, x, y)
        path.append((lat, lon))
    return path


def perturb_params(base: FieldParams, rng, current_speed_frac=0.25, current_dir_deg=20,
                    wind_speed_frac=0.30, wind_dir_deg=25, windage_frac=0.3):
    """Return a randomly perturbed copy of base params for one ensemble member."""
    return FieldParams(
        current_speed=max(0.01, base.current_speed * (1 + rng.uniform(-current_speed_frac, current_speed_frac))),
        current_dir_deg=(base.current_dir_deg + rng.uniform(-current_dir_deg, current_dir_deg)) % 360,
        wind_speed=max(0.0, base.wind_speed * (1 + rng.uniform(-wind_speed_frac, wind_speed_frac))),
        wind_dir_deg=(base.wind_dir_deg + rng.uniform(-wind_dir_deg, wind_dir_deg)) % 360,
        windage=max(0.0, base.windage * (1 + rng.uniform(-windage_frac, windage_frac))),
        spatial_gradient=base.spatial_gradient,
    )


def run_ensemble_hindcast(lat_start, lon_start, hours, base_params: FieldParams,
                           n_members=20, seed=42, dt_minutes=15):
    """
    Run an ensemble of backward (hindcast) trajectories from the detected slick
    location, with perturbed wind/current for each member.

    Returns:
        origins: list of (lat, lon) — the backtracked origin estimate for each member
        paths: list of full paths (for plotting)
    """
    rng = np.random.default_rng(seed)
    origins, paths = [], []
    for _ in range(n_members):
        p = perturb_params(base_params, rng)
        path = run_single_trajectory(lat_start, lon_start, hours, direction=-1,
                                      params=p, dt_minutes=dt_minutes)
        origins.append(path[-1])
        paths.append(path)
    return origins, paths


def run_ensemble_forecast(lat_start, lon_start, hours, base_params: FieldParams,
                           n_members=20, seed=7, dt_minutes=15):
    """Ensemble forward drift forecast (where the slick is heading)."""
    rng = np.random.default_rng(seed)
    endpoints, paths = [], []
    for _ in range(n_members):
        p = perturb_params(base_params, rng)
        path = run_single_trajectory(lat_start, lon_start, hours, direction=+1,
                                      params=p, dt_minutes=dt_minutes)
        endpoints.append(path[-1])
        paths.append(path)
    return endpoints, paths


def origin_probability_region(origins, confidence=0.68):
    """
    Summarise the ensemble of backtracked origins as a probability region:
    centroid, covariance ellipse (1-sigma by default), and radius in km
    that contains `confidence` fraction of members (simple empirical estimate).

    This is what makes the hindcast "probabilistic" rather than a single point.
    """
    lats = np.array([o[0] for o in origins])
    lons = np.array([o[1] for o in origins])
    lat_c, lon_c = lats.mean(), lons.mean()
    x, y = latlon_to_local_xy(lat_c, lon_c, lats, lons)
    dists_km = np.hypot(x, y) / 1000.0
    radius_km = np.quantile(dists_km, confidence)
    cov = np.cov(np.vstack([x, y]))
    return {
        "centroid_lat": lat_c,
        "centroid_lon": lon_c,
        "radius_km_{}pct".format(int(confidence * 100)): radius_km,
        "cov_xy_m2": cov,
        "n_members": len(origins),
    }


if __name__ == "__main__":
    # Quick smoke test
    base = FieldParams()
    origins, paths = run_ensemble_hindcast(19.05, 72.85, hours=18, base_params=base, n_members=20)
    region = origin_probability_region(origins)
    print("Origin probability region:", region)

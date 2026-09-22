"""
AIS attribution module for AquaSentinel.

Innovation 2: vessels are scored against the ORIGIN PROBABILITY REGION
(from the ensemble hindcast), not a single point, and vessels with an
AIS gap during the release window are explicitly flagged as dark-vessel
candidates rather than silently excluded.

AIS here is synthetic (clearly labelled), built around a known "true"
culprit vessel plus decoys and one deliberately gapped vessel — this is
what the benchmark module (benchmark/synthetic_benchmark.py) exercises
repeatedly to produce a top-k accuracy number.
"""

import numpy as np
from dataclasses import dataclass, field


EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


@dataclass
class Vessel:
    mmsi: str
    name: str
    vessel_type: str  # "tanker", "cargo", "fishing", "other"
    track: list       # list of (lat, lon, timestamp_hours) — timestamp relative to spill window start
    has_ais_gap_near_origin: bool = False


def generate_synthetic_fleet(origin_lat, origin_lon, origin_time_h, rng, n_decoys=6,
                              true_culprit=True, gap_vessel=True):
    """
    Build a synthetic AIS fleet around a spill origin.
    - one 'true' culprit vessel whose track passes through the origin at origin_time_h
    - n_decoys other vessels scattered nearby, not matching well
    - optionally one vessel with an AIS gap exactly over the origin window (dark-vessel case)

    Returns a list[Vessel]. The identity of the true culprit's MMSI is returned separately
    for benchmarking (NOT used by the scoring function itself — scoring must find it blind).
    """
    vessel_types = ["tanker", "cargo", "fishing", "other"]
    fleet = []
    true_mmsi = None

    if true_culprit:
        # Track: passes close to origin at origin_time_h, moving in a straight line
        heading = rng.uniform(0, 360)
        speed_kmh = rng.uniform(15, 28)
        track = []
        for t in np.arange(-4, 6, 0.5):
            dist_km = speed_kmh * t
            dlat = (dist_km / 111.0) * np.cos(np.radians(heading))
            dlon = (dist_km / (111.0 * np.cos(np.radians(origin_lat)))) * np.sin(np.radians(heading))
            track.append((origin_lat + dlat, origin_lon + dlon, origin_time_h + t))
        true_mmsi = "TRUE-" + str(rng.integers(100000, 999999))
        fleet.append(Vessel(mmsi=true_mmsi, name="MV Culprit (synthetic)",
                             vessel_type="tanker", track=track))

    for i in range(n_decoys):
        # Decoys: scattered further away, random headings, may or may not overlap in time
        offset_km = rng.uniform(8, 60)
        bearing = rng.uniform(0, 360)
        start_lat = origin_lat + (offset_km / 111.0) * np.cos(np.radians(bearing))
        start_lon = origin_lon + (offset_km / (111.0 * np.cos(np.radians(origin_lat)))) * np.sin(np.radians(bearing))
        heading = rng.uniform(0, 360)
        speed_kmh = rng.uniform(10, 25)
        time_offset = rng.uniform(-8, 8)  # may not even overlap the window
        track = []
        for t in np.arange(-4, 6, 0.5):
            dist_km = speed_kmh * t
            dlat = (dist_km / 111.0) * np.cos(np.radians(heading))
            dlon = (dist_km / (111.0 * np.cos(np.radians(start_lat)))) * np.sin(np.radians(heading))
            track.append((start_lat + dlat, start_lon + dlon, origin_time_h + time_offset + t))
        fleet.append(Vessel(mmsi=f"DECOY-{rng.integers(100000, 999999)}",
                             name=f"Vessel {i+1}",
                             vessel_type=rng.choice(vessel_types), track=track))

    if gap_vessel:
        # A vessel whose AIS simply stops broadcasting right over the origin window —
        # dark-vessel candidate. Track has a gap: no points within +-1.5h of origin_time_h.
        offset_km = rng.uniform(5, 20)
        bearing = rng.uniform(0, 360)
        start_lat = origin_lat + (offset_km / 111.0) * np.cos(np.radians(bearing))
        start_lon = origin_lon + (offset_km / (111.0 * np.cos(np.radians(origin_lat)))) * np.sin(np.radians(bearing))
        heading = rng.uniform(0, 360)
        speed_kmh = rng.uniform(12, 20)
        track = []
        for t in np.arange(-4, 6, 0.5):
            if abs(t) <= 1.5:
                continue  # gap
            dist_km = speed_kmh * t
            dlat = (dist_km / 111.0) * np.cos(np.radians(heading))
            dlon = (dist_km / (111.0 * np.cos(np.radians(start_lat)))) * np.sin(np.radians(heading))
            track.append((start_lat + dlat, start_lon + dlon, origin_time_h + t))
        fleet.append(Vessel(mmsi=f"DARK-{rng.integers(100000, 999999)}",
                             name="Unnamed (AIS gap)", vessel_type="unknown",
                             track=track, has_ais_gap_near_origin=True))

    rng.shuffle(fleet)
    return fleet, true_mmsi


def score_vessel(vessel: Vessel, origin_lat, origin_lon, origin_time_h, radius_km,
                  weights=None):
    """
    Score one vessel against the origin probability region.
    Weighted formula (0-100), matching the original design:
      proximity        0-30
      temporal overlap  0-20
      trajectory align  0-15
      AIS gap near origin 0-20  (higher score = more suspicious, i.e. gap present)
      vessel type       0-15
    Returns (score, reason_codes: list[str]).
    """
    if weights is None:
        weights = dict(proximity=30, temporal=20, trajectory=15, gap=20, vtype=15)
    reasons = []

    if len(vessel.track) == 0:
        # entire track is the gap -> nothing to compare except the flag
        proximity_score = 0
        temporal_score = 0
        trajectory_score = 0
    else:
        # closest approach to origin, in space and time
        dists = [haversine_km(lat, lon, origin_lat, origin_lon) for lat, lon, t in vessel.track]
        time_diffs = [abs(t - origin_time_h) for lat, lon, t in vessel.track]
        min_dist_idx = int(np.argmin(dists))
        min_dist = dists[min_dist_idx]
        min_time_diff = time_diffs[min_dist_idx]

        proximity_score = weights["proximity"] * max(0, 1 - min_dist / max(radius_km, 1e-6))
        if min_dist <= radius_km:
            reasons.append(f"within {min_dist:.1f} km of origin region (radius {radius_km:.1f} km)")

        temporal_score = weights["temporal"] * max(0, 1 - min_time_diff / 3.0)
        if min_time_diff <= 1.0:
            reasons.append(f"track passes within {min_time_diff:.1f}h of release window")

        # trajectory alignment: does the vessel's heading around closest approach
        # point roughly AWAY from origin (consistent with having just left it)?
        if len(vessel.track) >= 2 and 0 < min_dist_idx < len(vessel.track) - 1:
            lat0, lon0, _ = vessel.track[max(0, min_dist_idx - 1)]
            lat1, lon1, _ = vessel.track[min(len(vessel.track) - 1, min_dist_idx + 1)]
            heading_vec = np.array([lat1 - lat0, lon1 - lon0])
            origin_vec = np.array([lat0 - origin_lat, lon0 - origin_lon])
            if np.linalg.norm(heading_vec) > 0 and np.linalg.norm(origin_vec) > 0:
                cos_sim = np.dot(heading_vec, origin_vec) / (
                    np.linalg.norm(heading_vec) * np.linalg.norm(origin_vec))
                trajectory_score = weights["trajectory"] * max(0, cos_sim)
                if cos_sim > 0.5:
                    reasons.append("trajectory consistent with moving away from origin")
            else:
                trajectory_score = 0
        else:
            trajectory_score = 0

    gap_score = weights["gap"] if vessel.has_ais_gap_near_origin else 0
    if vessel.has_ais_gap_near_origin:
        reasons.append("AIS signal gap spanning the release window (dark-vessel candidate)")

    vtype_weight_map = {"tanker": 1.0, "cargo": 0.7, "fishing": 0.3, "other": 0.4, "unknown": 0.6}
    vtype_score = weights["vtype"] * vtype_weight_map.get(vessel.vessel_type, 0.4)
    if vessel.vessel_type == "tanker":
        reasons.append("vessel type: tanker (higher prior for oil cargo)")

    total = proximity_score + temporal_score + trajectory_score + gap_score + vtype_score
    return round(float(total), 1), reasons


def rank_vessels(fleet, origin_lat, origin_lon, origin_time_h, radius_km):
    results = []
    for v in fleet:
        score, reasons = score_vessel(v, origin_lat, origin_lon, origin_time_h, radius_km)
        results.append({
            "mmsi": v.mmsi, "name": v.name, "vessel_type": v.vessel_type,
            "score": score, "reasons": reasons,
            "dark_vessel_flag": v.has_ais_gap_near_origin,
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    origin_lat, origin_lon, origin_time_h = 19.386, 73.001, 0.0
    fleet, true_mmsi = generate_synthetic_fleet(origin_lat, origin_lon, origin_time_h, rng)
    ranked = rank_vessels(fleet, origin_lat, origin_lon, origin_time_h, radius_km=7.4)
    print(f"True culprit MMSI: {true_mmsi}\n")
    for i, r in enumerate(ranked, 1):
        marker = " <-- TRUE CULPRIT" if r["mmsi"] == true_mmsi else ""
        print(f"#{i} {r['mmsi']:>16} score={r['score']:5.1f} type={r['vessel_type']:<8} "
              f"dark={r['dark_vessel_flag']}{marker}")
        for reason in r["reasons"]:
            print(f"      - {reason}")

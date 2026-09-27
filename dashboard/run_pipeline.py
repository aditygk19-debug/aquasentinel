"""
AquaSentinel — dashboard data pipeline.

Runs detection + ensemble hindcast + AIS attribution once and returns
everything the dashboard needs as a plain Python dict. No subprocess,
no file round-trips.
"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'drift'))
sys.path.insert(0, os.path.join(_ROOT, 'ais'))
sys.path.insert(0, os.path.join(_ROOT, 'detect'))

from ensemble_drift import FieldParams, run_ensemble_hindcast, origin_probability_region
from ais_attribution import generate_synthetic_fleet, rank_vessels


CHECKPOINT_PATH = os.path.join(_ROOT, 'detect', 'checkpoints', 'efficientnet_zenodo_best.pt')
SCENE_PATH = os.path.join(_ROOT, 'data', 'sample_scene.png')
SCENE_BOUNDS = (18.85, 19.25, 72.65, 73.05)


def _detect():
    """Run the CNN detector if checkpoint + scene exist; else fall back."""
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(SCENE_PATH):
        from detect_model import load_model, detect_slick_region, bbox_px_to_geo, estimate_area_km2
        model = load_model(CHECKPOINT_PATH)
        conf_grid, oil_mask, bbox_px, scene_size = detect_slick_region(
            model, SCENE_PATH, threshold=0.5)
        if bbox_px is None:
            return {'source': 'cnn-no-detection', 'lat': 19.05, 'lon': 72.85,
                    'area_km2': None, 'confidence_max': float(conf_grid.max()),
                    'bbox_px': None, 'scene_size': scene_size}
        geo = bbox_px_to_geo(bbox_px, scene_size, SCENE_BOUNDS)
        area_km2 = estimate_area_km2(bbox_px, scene_size, SCENE_BOUNDS)
        return {
            'source': 'cnn',
            'lat': geo['centroid_lat'], 'lon': geo['centroid_lon'],
            'area_km2': area_km2,
            'confidence_max': float(conf_grid.max()),
            'bbox_px': bbox_px, 'scene_size': scene_size,
        }
    return {'source': 'fallback', 'lat': 19.05, 'lon': 72.85,
            'area_km2': None, 'confidence_max': None,
            'bbox_px': None, 'scene_size': None}


def run_pipeline(hours_since_release=18, n_members=25, seed=42):
    """Run the full pipeline and return a dict with all data for the dashboard."""
    det = _detect()
    slick_lat, slick_lon = det['lat'], det['lon']

    # Direction convention: degrees are the compass bearing the current/wind
    # flows TOWARD. Setting ~100 deg (roughly ESE) means forward drift is
    # onshore, so backward hindcast runs offshore (west) -- physically
    # plausible for Arabian Sea monsoon-season flow and geographically
    # consistent with the demo region.
    base_params = FieldParams(current_speed=0.4, current_dir_deg=100,
                              wind_speed=6.0, wind_dir_deg=100)
    origins, paths = run_ensemble_hindcast(
        slick_lat, slick_lon, hours=hours_since_release,
        base_params=base_params, n_members=n_members, seed=seed)
    region = origin_probability_region(origins)
    radius_key = [k for k in region if k.startswith('radius_km')][0]
    radius_km = region[radius_key]

    rng = np.random.default_rng(7)
    fleet, true_mmsi = generate_synthetic_fleet(
        region['centroid_lat'], region['centroid_lon'],
        origin_time_h=0.0, rng=rng, n_decoys=6, gap_vessel=True)
    ranked = rank_vessels(fleet, region['centroid_lat'], region['centroid_lon'],
                          origin_time_h=0.0, radius_km=radius_km)

    # Simplify paths for JSON-ish transport (list of [lat, lon] per path)
    path_arrays = [[[float(p[0]), float(p[1])] for p in path] for path in paths]
    origin_points = [[float(o[0]), float(o[1])] for o in origins]

    # Fleet tracks for map overlay
    fleet_tracks = []
    for v in fleet:
        fleet_tracks.append({
            'mmsi': v.mmsi,
            'name': v.name,
            'vessel_type': v.vessel_type,
            'dark': bool(v.has_ais_gap_near_origin),
            'track': [[float(t[0]), float(t[1])] for t in v.track],
        })

    return {
        'detection': det,
        'slick': {'lat': float(slick_lat), 'lon': float(slick_lon)},
        'origin_points': origin_points,
        'paths': path_arrays,
        'region': {
            'centroid_lat': float(region['centroid_lat']),
            'centroid_lon': float(region['centroid_lon']),
            'radius_km': float(radius_km),
            'n_members': int(region['n_members']),
        },
        'ranked': ranked,
        'true_mmsi': true_mmsi,
        'fleet_tracks': fleet_tracks,
        'hours_since_release': hours_since_release,
    }


if __name__ == '__main__':
    import json
    data = run_pipeline()
    print('Slick:', data['slick'])
    print('Region:', data['region'])
    print('Top suspect:', data['ranked'][0]['mmsi'], '| score',
          data['ranked'][0]['score'], '| true?', data['ranked'][0]['mmsi'] == data['true_mmsi'])
    print('Fleet size:', len(data['ranked']), '| paths:', len(data['paths']))

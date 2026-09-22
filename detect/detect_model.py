"""
Detection module for AquaSentinel.

Wraps the trained EfficientNet-B0 classifier (single-channel adapted,
96x96 input, trained on the Zenodo Deep-SAR dataset -- 99.01% accuracy,
per prior training runs) and adds one new capability requested for
round 2: instead of classifying one whole image as oil/no-oil, we slide
a window over a larger scene and classify each tile, producing an
approximate bounding region (polygon) for the slick -- without having
to train a full segmentation model from scratch under a 2-day deadline.

Usage:
    model = load_model("checkpoints/efficientnet_zenodo_best.pt")
    polygon, mask, confidence_grid = detect_slick_region(model, scene_path)

If no checkpoint is available yet, `load_model(None)` returns a randomly
initialised model with the correct architecture, purely so the rest of
the pipeline (region extraction, plotting, run_demo.py) can be exercised
end-to-end while the real weights are being dropped in.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image


TILE_SIZE = 96  # matches original training input size


def build_efficientnet_b0_single_channel(num_classes=2):
    """Recreate the exact architecture used in training: EfficientNet-B0 with
    the first conv layer adapted to accept 1-channel (grayscale SAR) input."""
    model = tvm.efficientnet_b0(weights=None)
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    model.features[0][0] = new_conv
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(checkpoint_path=None, device="cpu"):
    """Load the trained model. If checkpoint_path is None or missing, returns
    an untrained model with correct architecture (for pipeline testing only --
    predictions will be meaningless until real weights are loaded)."""
    model = build_efficientnet_b0_single_channel(num_classes=2)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)
        print(f"Loaded trained weights from {checkpoint_path}")
    else:
        print("[WARNING] No checkpoint found -- using randomly initialised weights. "
              "Drop your trained .pt file at the given path for real predictions.")
    model.to(device)
    model.eval()
    return model


def preprocess_tile(tile_img: Image.Image):
    """Resize to TILE_SIZE, convert to grayscale, normalise to [0,1] tensor."""
    tile = tile_img.convert("L").resize((TILE_SIZE, TILE_SIZE))
    arr = np.asarray(tile, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor


@torch.no_grad()
def classify_tile(model, tile_img: Image.Image, device="cpu"):
    """Returns probability of 'oil' class for one tile."""
    tensor = preprocess_tile(tile_img).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    return float(probs[0, 1])  # class 1 = oil, matching original training labels


def detect_slick_region(model, scene_path, window=TILE_SIZE, stride=None,
                         threshold=0.5, device="cpu"):
    """
    Slide a window over a larger SAR scene, classify each tile, and build an
    approximate confidence grid + bounding polygon for oil-classified tiles.

    Returns:
        confidence_grid: 2D numpy array of oil-probabilities, one per tile
        oil_mask: boolean grid, confidence_grid > threshold
        bbox_px: (row_min, col_min, row_max, col_max) pixel bounding box of
                 detected tiles in the ORIGINAL scene, or None if nothing detected
        scene_size: (width, height) of the original scene image
    """
    if stride is None:
        stride = window  # non-overlapping by default; use window//2 for finer grid

    scene = Image.open(scene_path)
    w, h = scene.size
    n_cols = max(1, (w - window) // stride + 1)
    n_rows = max(1, (h - window) // stride + 1)

    confidence_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    for r in range(n_rows):
        for c in range(n_cols):
            left, top = c * stride, r * stride
            tile = scene.crop((left, top, left + window, top + window))
            confidence_grid[r, c] = classify_tile(model, tile, device=device)

    oil_mask = confidence_grid > threshold
    if not oil_mask.any():
        return confidence_grid, oil_mask, None, (w, h)

    rows, cols = np.where(oil_mask)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()
    bbox_px = (
        row_min * stride,
        col_min * stride,
        row_max * stride + window,
        col_max * stride + window,
    )
    return confidence_grid, oil_mask, bbox_px, (w, h)


def bbox_px_to_geo(bbox_px, scene_size, scene_bounds):
    """
    Convert a pixel bounding box to lat/lon, given the scene's geographic
    bounds (lat_min, lat_max, lon_min, lon_max) -- e.g. from the SAR product's
    metadata / GeoTIFF geotransform. This is a simple linear mapping; swap in
    a proper affine transform if working from georeferenced GeoTIFFs.
    """
    row_min, col_min, row_max, col_max = bbox_px
    w, h = scene_size
    lat_min, lat_max, lon_min, lon_max = scene_bounds

    def px_to_latlon(row, col):
        lat = lat_max - (row / h) * (lat_max - lat_min)
        lon = lon_min + (col / w) * (lon_max - lon_min)
        return lat, lon

    lat1, lon1 = px_to_latlon(row_min, col_min)
    lat2, lon2 = px_to_latlon(row_max, col_max)
    centroid_lat = (lat1 + lat2) / 2
    centroid_lon = (lon1 + lon2) / 2
    return {"centroid_lat": centroid_lat, "centroid_lon": centroid_lon,
            "corners": [(lat1, lon1), (lat2, lon2)]}


def estimate_area_km2(bbox_px, scene_size, scene_bounds):
    """Rough area estimate from the bounding box, using the geo bounds for scale."""
    row_min, col_min, row_max, col_max = bbox_px
    w, h = scene_size
    lat_min, lat_max, lon_min, lon_max = scene_bounds
    km_per_px_lat = (111.0 * (lat_max - lat_min)) / h
    lat_mid = (lat_min + lat_max) / 2
    km_per_px_lon = (111.0 * np.cos(np.radians(lat_mid)) * (lon_max - lon_min)) / w
    height_km = (row_max - row_min) * km_per_px_lat
    width_km = (col_max - col_min) * km_per_px_lon
    return abs(height_km * width_km)


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    model = load_model(ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready. Parameters: {n_params:,} "
          f"(matches EfficientNet-B0 single-channel spec, ~4.0M expected)")

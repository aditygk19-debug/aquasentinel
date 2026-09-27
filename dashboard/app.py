"""
AquaSentinel interactive dashboard.

Run:  streamlit run dashboard/app.py
"""

import os
import sys
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_pipeline import run_pipeline

st.set_page_config(page_title="AquaSentinel", page_icon="🛢️", layout="wide")

st.title("🛢️ AquaSentinel — Maritime Oil Spill Attribution")
st.caption("Detect slick → ensemble hindcast origin region → ranked vessel suspects with reason codes")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Run parameters")
    hours = st.slider("Hours since release (hindcast window)", 6, 24, 18)
    n_members = st.slider("Ensemble members", 10, 50, 25)
    seed = st.number_input("Random seed", value=42, step=1)
    uploaded_scene = st.file_uploader("Upload SAR scene (optional)",
                                       type=["png", "jpg", "jpeg"])
    if uploaded_scene is not None:
        st.caption("Custom upload overrides the bundled sample scene.")
    run_btn = st.button("▶ Run pipeline", type="primary")
    st.divider()
    st.caption("**Primary user:** Coast Guard / pollution-control duty officer")
    st.caption("**Output:** ranked suspect list — candidates for investigation, not verdicts.")

# --- Session state ---
if "data" not in st.session_state:
    st.session_state.data = None

if run_btn or st.session_state.data is None:
    if uploaded_scene is not None:
        # Save uploaded file as the scene the pipeline will read
        import os as _os
        _scene_path = _os.path.join(_HERE, '..', 'data', 'sample_scene.png')
        _os.makedirs(_os.path.dirname(_scene_path), exist_ok=True)
        with open(_scene_path, 'wb') as f:
            f.write(uploaded_scene.getbuffer())
        st.sidebar.success("Uploaded scene saved.")

    with st.spinner("Running detection → ensemble hindcast → AIS attribution ..."):
        st.session_state.data = run_pipeline(
            hours_since_release=hours, n_members=n_members, seed=int(seed))

data = st.session_state.data

# --- Top-line metrics ---
det = data["detection"]
col1, col2, col3, col4 = st.columns(4)
_src = det["source"]
if _src == "cnn":
    _src_label = "cnn (trained checkpoint loaded)"
elif _src == "cnn-no-detection":
    _src_label = "cnn (no slick above threshold)"
elif _src == "fallback":
    _src_label = "fallback (no checkpoint/scene)"
else:
    _src_label = _src
col1.metric("Detection source", _src_label)
col2.metric("Slick lat", f"{data['slick']['lat']:.3f}")
col2.caption(f"lon {data['slick']['lon']:.3f}")
col3.metric("Origin region radius", f"{data['region']['radius_km']:.1f} km")
col4.metric("Fleet size", len(data["ranked"]))

st.divider()

# --- Main layout: map full width ---
st.subheader("Map — detected slick, ensemble origins, vessel tracks")

# Build map centered on slick
center_lat = data["slick"]["lat"]
center_lon = data["slick"]["lon"]
m = folium.Map(location=[center_lat, center_lon], zoom_start=9,
               tiles="OpenStreetMap")

# Ensemble backtracked paths (faint blue lines)
for path in data["paths"]:
    folium.PolyLine(path, color="steelblue", weight=1, opacity=0.35).add_to(m)

# Ensemble origin points (small red dots)
for o in data["origin_points"]:
    folium.CircleMarker(location=o, radius=3, color="crimson",
                        fill=True, fill_opacity=0.8).add_to(m)

# Origin probability region (pink circle)
radius_m = data["region"]["radius_km"] * 1000
folium.Circle(
    location=[data["region"]["centroid_lat"], data["region"]["centroid_lon"]],
    radius=radius_m,
    color="crimson", fill=True, fill_opacity=0.15,
    popup=f"68% confidence region (~{data['region']['radius_km']:.0f} km)").add_to(m)

# Origin centroid marker (red X)
folium.Marker(
    location=[data["region"]["centroid_lat"], data["region"]["centroid_lon"]],
    icon=folium.Icon(color="red", icon="times", prefix="fa"),
    popup="Origin estimate (centroid)").add_to(m)

# Detected slick marker (black star)
folium.Marker(
    location=[data["slick"]["lat"], data["slick"]["lon"]],
    icon=folium.Icon(color="black", icon="star", prefix="fa"),
    popup=f"Detected slick<br>source: {det['source']}").add_to(m)

# Vessel tracks
for v in data["fleet_tracks"]:
    if not v["track"]:
        continue
    color = "crimson" if v["mmsi"] == data["true_mmsi"] else ("orange" if v["dark"] else "steelblue")
    folium.PolyLine(
        [[t[0], t[1]] for t in v["track"]],
        color=color, weight=2, opacity=0.75,
        popup=f"{v['mmsi']} ({v['vessel_type']}){' - DARK' if v['dark'] else ''}").add_to(m)

# Auto-fit bounds to include slick, origins, region, and all tracks
all_lats = [data["slick"]["lat"], data["region"]["centroid_lat"]]
all_lons = [data["slick"]["lon"], data["region"]["centroid_lon"]]
for o in data["origin_points"]:
    all_lats.append(o[0]); all_lons.append(o[1])
for v in data["fleet_tracks"]:
    for t in v["track"]:
        all_lats.append(t[0]); all_lons.append(t[1])
pad = 0.08
m.fit_bounds([[min(all_lats) - pad, min(all_lons) - pad],
              [max(all_lats) + pad, max(all_lons) + pad]])

st_folium(m, width=700, height=550, returned_objects=[])

st.caption("Geographic bounds shown are for the demo region. Custom uploads use "
           "approximate bounds for illustration — pixel-to-lat/lon is not GeoTIFF-accurate.")

st.subheader("Ranked suspects")

def _flag(r, true_mmsi):
    if r["mmsi"] == true_mmsi:
        return "🔴 TRUE CULPRIT"
    if r["dark_vessel_flag"]:
        return "🟠 DARK VESSEL"
    return "🔵 normal"

df = pd.DataFrame([
    {
        "rank": i + 1,
        "flag": _flag(r, data["true_mmsi"]),
        "MMSI": r["mmsi"],
        "type": r["vessel_type"],
        "score": r["score"],
    }
    for i, r in enumerate(data["ranked"])
])
st.dataframe(df, hide_index=True, use_container_width=True)

st.caption("Crimson = ground-truth culprit · Orange = dark vessel (AIS gap) · Blue = normal traffic")

st.divider()

# --- Candidate detail ---
st.subheader("Candidate detail — reason codes")
selected_mmsi = st.selectbox("Select a vessel to inspect",
                              [r["mmsi"] for r in data["ranked"]])
selected = next(r for r in data["ranked"] if r["mmsi"] == selected_mmsi)

c1, c2 = st.columns([1, 2])
c1.metric("Suspicion score", f"{selected['score']:.1f} / 100")
c1.metric("Vessel type", selected["vessel_type"])
c1.metric("Dark-vessel flag", "YES" if selected["dark_vessel_flag"] else "no")
if selected_mmsi == data["true_mmsi"]:
    c1.success("✓ matches ground-truth culprit")

with c2:
    st.markdown("**Why this vessel was ranked here:**")
    if selected["reasons"]:
        for reason in selected["reasons"]:
            st.markdown(f"- {reason}")
    else:
        st.markdown("- *(no scoring factor fired strongly)*")

st.divider()
st.caption("**Disclaimer:** Ranked candidates, not proof of guilt — for investigative prioritization only. "
           "AIS feed is synthetic for reproducibility; scoring logic is real.")

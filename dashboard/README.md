# Dashboard

Interactive Streamlit dashboard for AquaSentinel.

## Run locally

```bash
pip install streamlit streamlit-folium folium
streamlit run dashboard/app.py
```

Opens at http://localhost:8501.

## What it shows

- **Map:** detected slick (black star), 25-member ensemble backtracked origin cloud, 68% confidence region (pink circle), vessel tracks (crimson = true culprit, orange = dark vessel, steelblue = normal traffic)
- **Metrics row:** detection source, slick location, origin region radius, fleet size
- **Ranked suspects table:** position, MMSI, vessel type, suspicion score, dark-vessel flag, ground-truth match
- **Candidate detail:** select any vessel to see the specific reason codes that placed it there

## Data source

Runs `run_pipeline.py` in-process -- the same detection + drift ensemble + AIS attribution logic used by `run_demo.py`. No subprocess, no API layer.

## Disclaimer

Ranked candidates, not proof of guilt. AIS feed is synthetic for reproducibility; scoring logic is real.

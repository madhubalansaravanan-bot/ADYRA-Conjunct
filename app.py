import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sgp4.api import Satrec
from datetime import datetime, timezone

st.set_page_config(
    page_title="ADYRA Space",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ ADYRA SPACE")
st.subheader("Real-Time Satellite Position & Conjunction Analysis")

# --------------------------------------------------
# GET REAL TLE DATA FROM CELESTRAK
# --------------------------------------------------

@st.cache_data(ttl=3600)
def get_satellites():
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle"

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    lines = [x.strip() for x in response.text.splitlines() if x.strip()]

    satellites = []

    for i in range(0, len(lines) - 2, 3):
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        satellites.append({
            "name": name,
            "line1": line1,
            "line2": line2
        })

    return satellites


# --------------------------------------------------
# SATELLITE POSITION
# --------------------------------------------------

def satellite_position(line1, line2):

    satellite = Satrec.twoline2rv(line1, line2)

    now = datetime.now(timezone.utc)

    jd = now.timestamp() / 86400.0 + 2440587.5
    jd_int = int(jd)
    jd_frac = jd - jd_int

    error, position, velocity = satellite.sgp4(
        jd_int,
        jd_frac
    )

    if error != 0:
        return None

    position = np.array(position)
    velocity = np.array(velocity)

    # Position returned by SGP4 is km
    altitude = np.linalg.norm(position) - 6378.137

    speed = np.linalg.norm(velocity)

    # Approximate geocentric latitude / longitude
    longitude = np.degrees(
        np.arctan2(position[1], position[0])
    )

    latitude = np.degrees(
        np.arctan2(
            position[2],
            np.sqrt(position[0]**2 + position[1]**2)
        )
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "velocity": speed,
        "timestamp": now
    }


# --------------------------------------------------
# LOAD SATELLITES
# --------------------------------------------------

try:
    satellites = get_satellites()

except Exception as e:
    st.error(f"Unable to retrieve satellite data: {e}")
    st.stop()


names = [s["name"] for s in satellites]

selected = st.selectbox(
    "🔎 Select a satellite",
    names
)

sat = next(
    s for s in satellites
    if s["name"] == selected
)

data = satellite_position(
    sat["line1"],
    sat["line2"]
)


# --------------------------------------------------
# DISPLAY POSITION
# --------------------------------------------------

if data:

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latitude",
        f"{data['latitude']:.4f}°"
    )

    c2.metric(
        "Longitude",
        f"{data['longitude']:.4f}°"
    )

    c3.metric(
        "Altitude",
        f"{data['altitude']:.2f} km"
    )

    c4.metric(
        "Velocity",
        f"{data['velocity']:.2f} km/s"
    )

    st.caption(
        f"Estimated position calculated from current TLE • "
        f"{data['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )


# --------------------------------------------------
# EARTH MAP
# --------------------------------------------------

st.markdown("### 🌍 Current Satellite Position")

fig = go.Figure()

fig.add_trace(
    go.Scattergeo(
        lon=[data["longitude"]],
        lat=[data["latitude"]],
        mode="markers",
        marker=dict(
            size=14,
            color="cyan"
        ),
        text=[
            f"{selected}<br>"
            f"Altitude: {data['altitude']:.2f} km"
        ],
        hoverinfo="text"
    )
)

fig.update_geos(
    projection_type="orthographic",
    showland=True,
    showcountries=True,
    showocean=True
)

fig.update_layout(
    height=650,
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(
    fig,
    use_container_width=True
)

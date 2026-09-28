import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
from sgp4.api import Satrec
from datetime import datetime, timezone

st.set_page_config(
    page_title="ADYRA OrbitGuard",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ ADYRA OrbitGuard")
st.caption("Real-time satellite tracking & conjunction screening")

@st.cache_data(ttl=3600)
def get_satellites():
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle"

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    lines = [
        line.strip()
        for line in response.text.splitlines()
        if line.strip()
    ]

    satellites = []

    for i in range(0, len(lines) - 2, 3):
        satellites.append({
            "name": lines[i],
            "line1": lines[i + 1],
            "line2": lines[i + 2]
        })

    return satellites


def calculate_position(line1, line2):

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

    altitude = np.linalg.norm(position) - 6378.137
    speed = np.linalg.norm(velocity)

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
        "time": now
    }


# Get satellite data
try:
    satellites = get_satellites()
except Exception as e:
    st.error(f"Could not load satellite data: {e}")
    st.stop()


# Satellite selector
names = [sat["name"] for sat in satellites]

selected_name = st.selectbox(
    "🔎 Select Satellite",
    names
)

selected_satellite = next(
    sat for sat in satellites
    if sat["name"] == selected_name
)


# Calculate position
data = calculate_position(
    selected_satellite["line1"],
    selected_satellite["line2"]
)


if data:

    st.markdown("### 📍 Current Estimated Position")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latitude",
        f"{data['latitude']:.4f}°"
    )

    col2.metric(
        "Longitude",
        f"{data['longitude']:.4f}°"
    )

    col3.metric(
        "Altitude",
        f"{data['altitude']:.2f} km"
    )

    col4.metric(
        "Velocity",
        f"{data['velocity']:.2f} km/s"
    )

    st.caption(
        "Position calculated using the satellite's current TLE "
        f"at {data['time'].strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )


    # Map
    st.markdown("### 🌍 Satellite Location")

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lon=[data["longitude"]],
            lat=[data["latitude"]],
            mode="markers",
            marker=dict(
                size=15,
                color="cyan"
            ),
            text=[
                f"{selected_name}<br>"
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
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

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
st.caption("Real Satellite Tracking & Conjunction Analysis")

# --------------------------------------------------
# REAL TLE DATA
# GitHub mirror of CelesTrak data
# --------------------------------------------------

TLE_URL = (
    "https://raw.githubusercontent.com/"
    "caelo-works/tle-mirror/main/tle/stations.tle"
)


@st.cache_data(ttl=1800)
def load_satellites():

    response = requests.get(
        TLE_URL,
        timeout=30
    )

    response.raise_for_status()

    lines = [
        line.strip()
        for line in response.text.splitlines()
        if line.strip()
    ]

    satellites = []

    for i in range(0, len(lines) - 2, 3):

        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        if line1.startswith("1 ") and line2.startswith("2 "):

            satellites.append({
                "name": name,
                "line1": line1,
                "line2": line2
            })

    return satellites


# --------------------------------------------------
# CALCULATE SATELLITE POSITION
# --------------------------------------------------

def calculate_position(line1, line2):

    satellite = Satrec.twoline2rv(
        line1,
        line2
    )

    now = datetime.now(timezone.utc)

    jd = (
        now.timestamp() / 86400.0
        + 2440587.5
    )

    jd_int = int(jd)
    jd_fraction = jd - jd_int

    error, position, velocity = satellite.sgp4(
        jd_int,
        jd_fraction
    )

    if error != 0:
        return None

    position = np.array(position)
    velocity = np.array(velocity)

    # SGP4 position is km
    x, y, z = position

    # Approximate geocentric coordinates
    longitude = np.degrees(
        np.arctan2(y, x)
    )

    latitude = np.degrees(
        np.arctan2(
            z,
            np.sqrt(x*x + y*y)
        )
    )

    altitude = (
        np.linalg.norm(position)
        - 6378.137
    )

    speed = np.linalg.norm(
        velocity
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "velocity": speed,
        "time": now
    }


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:

    satellites = load_satellites()

except Exception as e:

    st.error(
        "Unable to load satellite data."
    )

    st.code(str(e))

    st.stop()


if not satellites:

    st.error(
        "No satellite data found."
    )

    st.stop()


st.success(
    f"🛰️ {len(satellites)} real satellite records loaded"
)


# --------------------------------------------------
# SATELLITE SELECTION
# --------------------------------------------------

satellite_names = [
    sat["name"]
    for sat in satellites
]

selected_name = st.selectbox(
    "🔎 Select Satellite",
    satellite_names
)


selected = next(
    sat
    for sat in satellites
    if sat["name"] == selected_name
)


# --------------------------------------------------
# POSITION
# --------------------------------------------------

data = calculate_position(
    selected["line1"],
    selected["line2"]
)


if data:

    st.markdown(
        "## 📍 Current Estimated Position"
    )

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
        "Position propagated from the latest available "
        "TLE using SGP4."
    )

    st.caption(
        "Time: "
        + data["time"].strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )


    # --------------------------------------------------
    # EARTH VISUALIZATION
    # --------------------------------------------------

    st.markdown(
        "## 🌍 Satellite Location"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lon=[data["longitude"]],
            lat=[data["latitude"]],
            mode="markers",
            marker=dict(
                size=15
            ),
            text=[
                selected_name
            ],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Latitude: %{lat:.4f}°<br>"
                "Longitude: %{lon:.4f}°"
                "<extra></extra>"
            )
        )
    )

    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        showcountries=True,
        showocean=True,
        coastlinecolor="gray"
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


# --------------------------------------------------
# DATA SOURCE
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Orbital data: CelesTrak GP data mirrored through "
    "GitHub. Position: SGP4 propagation."
)
# --------------------------------------------------
# 3D EARTH + SATELLITE
# --------------------------------------------------

st.markdown("## 🌍 3D Satellite Position")

# Earth
earth_radius = 6378.137

u = np.linspace(0, 2 * np.pi, 80)
v = np.linspace(0, np.pi, 40)

x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

fig = go.Figure()

# Earth surface
fig.add_trace(
    go.Surface(
        x=x_earth,
        y=y_earth,
        z=z_earth,
        surfacecolor=np.zeros_like(x_earth),
        colorscale=[
            [0, "#07152E"],
            [0.5, "#123C66"],
            [1, "#1D6FA5"]
        ],
        showscale=False,
        opacity=0.95,
        hoverinfo="skip"
    )
)

# Satellite position
sat_altitude = data["altitude"]

# Convert current lat/lon to Cartesian
lat = np.radians(data["latitude"])
lon = np.radians(data["longitude"])

r = earth_radius + sat_altitude

sx = r * np.cos(lat) * np.cos(lon)
sy = r * np.cos(lat) * np.sin(lon)
sz = r * np.sin(lat)

# Satellite
fig.add_trace(
    go.Scatter3d(
        x=[sx],
        y=[sy],
        z=[sz],
        mode="markers+text",
        marker=dict(
            size=9,
            color="#35D5F5"
        ),
        text=[selected_name],
        textposition="top center",
        name="Satellite",
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"Altitude: {data['altitude']:.2f} km<br>"
            f"Latitude: {data['latitude']:.4f}°<br>"
            f"Longitude: {data['longitude']:.4f}°"
            "<extra></extra>"
        )
    )
)

# Line from Earth to satellite
fig.add_trace(
    go.Scatter3d(
        x=[sx, sx * earth_radius / r],
        y=[sy, sy * earth_radius / r],
        z=[sz, sz * earth_radius / r],
        mode="lines",
        line=dict(
            color="#35D5F5",
            width=3
        ),
        name="Altitude"
    )
)

fig.update_layout(
    height=700,
    margin=dict(
        l=0,
        r=0,
        t=20,
        b=0
    ),
    paper_bgcolor="#040815",
    plot_bgcolor="#040815",
    scene=dict(
        xaxis=dict(
            visible=False
        ),
        yaxis=dict(
            visible=False
        ),
        zaxis=dict(
            visible=False
        ),
        aspectmode="data",
        camera=dict(
            eye=dict(
                x=1.6,
                y=1.6,
                z=1.2
            )
        )
    ),
    legend=dict(
        font=dict(
            color="white"
        )
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

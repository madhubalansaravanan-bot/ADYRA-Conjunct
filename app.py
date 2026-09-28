import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
from sgp4.api import Satrec
from datetime import datetime, timezone, timedelta
import math

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="ADYRA OrbitGuard",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ ADYRA OrbitGuard")
st.caption("Real-Time Satellite Tracking & Orbital Visualization")

# =========================================================
# CONSTANTS
# =========================================================

EARTH_RADIUS = 6378.137  # km

TLE_URL = (
    "https://raw.githubusercontent.com/"
    "caelo-works/tle-mirror/main/tle/stations.tle"
)

# =========================================================
# LOAD REAL SATELLITE DATA
# =========================================================

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


# =========================================================
# JULIAN DATE
# =========================================================

def julian_date(dt):

    return (
        dt.timestamp() / 86400.0
        + 2440587.5
    )


# =========================================================
# GREENWICH SIDEREAL TIME
# Converts ECI/TEME → Earth-fixed coordinates
# =========================================================

def gmst_angle(jd):

    T = (jd - 2451545.0) / 36525.0

    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T * T
        - (T ** 3) / 38710000.0
    )

    return math.radians(gmst % 360)


def eci_to_ecef(position, jd):

    theta = gmst_angle(jd)

    x, y, z = position

    x_ecef = (
        x * math.cos(theta)
        + y * math.sin(theta)
    )

    y_ecef = (
        -x * math.sin(theta)
        + y * math.cos(theta)
    )

    z_ecef = z

    return np.array([
        x_ecef,
        y_ecef,
        z_ecef
    ])


# =========================================================
# ECEF → LAT/LON/ALT
# =========================================================

def ecef_to_geodetic(position):

    x, y, z = position

    longitude = math.atan2(y, x)

    distance = math.sqrt(
        x * x + y * y
    )

    latitude = math.atan2(
        z,
        distance
    )

    altitude = (
        math.sqrt(
            x * x +
            y * y +
            z * z
        )
        - EARTH_RADIUS
    )

    return (
        math.degrees(latitude),
        math.degrees(longitude),
        altitude
    )


# =========================================================
# CURRENT SATELLITE POSITION
# =========================================================

def current_position(satellite):

    now = datetime.now(timezone.utc)

    jd = julian_date(now)

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

    ecef = eci_to_ecef(
        position,
        jd
    )

    latitude, longitude, altitude = (
        ecef_to_geodetic(ecef)
    )

    speed = np.linalg.norm(
        velocity
    )

    return {
        "eci": position,
        "ecef": ecef,
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "velocity": speed,
        "time": now
    }


# =========================================================
# PREDICT ORBIT
# =========================================================

def calculate_orbit(
    satellite,
    minutes=100,
    step=2
):

    now = datetime.now(timezone.utc)

    orbit_points = []
    ground_track = []

    for minute in np.arange(
        -20,
        minutes,
        step
    ):

        dt = now + timedelta(
            minutes=float(minute)
        )

        jd = julian_date(dt)

        jd_int = int(jd)
        jd_fraction = jd - jd_int

        error, position, velocity = satellite.sgp4(
            jd_int,
            jd_fraction
        )

        if error != 0:
            continue

        position = np.array(position)

        ecef = eci_to_ecef(
            position,
            jd
        )

        lat, lon, alt = ecef_to_geodetic(
            ecef
        )

        orbit_points.append({
            "position": ecef,
            "altitude": alt,
            "time": dt
        })

        ground_track.append({
            "latitude": lat,
            "longitude": lon
        })

    return orbit_points, ground_track


# =========================================================
# EARTH SPHERE
# =========================================================

def create_earth():

    u = np.linspace(
        0,
        2 * np.pi,
        100
    )

    v = np.linspace(
        0,
        np.pi,
        60
    )

    x = (
        EARTH_RADIUS
        * np.outer(
            np.cos(u),
            np.sin(v)
        )
    )

    y = (
        EARTH_RADIUS
        * np.outer(
            np.sin(u),
            np.sin(v)
        )
    )

    z = (
        EARTH_RADIUS
        * np.outer(
            np.ones(np.size(u)),
            np.cos(v)
        )
    )

    return x, y, z


# =========================================================
# LOAD DATA
# =========================================================

try:

    satellites = load_satellites()

except Exception as e:

    st.error(
        "Unable to load real satellite data."
    )

    st.code(str(e))

    st.stop()


if not satellites:

    st.error(
        "No satellite records found."
    )

    st.stop()


st.success(
    f"🛰️ {len(satellites)} real satellite records loaded"
)


# =========================================================
# SATELLITE SELECTOR
# =========================================================

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


# =========================================================
# CREATE SGP4 OBJECT
# =========================================================

satellite = Satrec.twoline2rv(
    selected["line1"],
    selected["line2"]
)


# =========================================================
# CURRENT POSITION
# =========================================================

data = current_position(
    satellite
)

if data is None:

    st.error(
        "SGP4 could not propagate this satellite."
    )

    st.stop()


# =========================================================
# INFORMATION CARDS
# =========================================================

st.markdown("## 📍 Current Satellite Position")

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
    f"{data['velocity']:.3f} km/s"
)

st.caption(
    "Position propagated using SGP4 from the latest available TLE."
)

st.caption(
    "UTC: "
    + data["time"].strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)


# =========================================================
# ORBIT CALCULATION
# =========================================================

orbit, ground = calculate_orbit(
    satellite,
    minutes=100,
    step=2
)

if not orbit:

    st.error(
        "Unable to calculate orbital trajectory."
    )

    st.stop()


# =========================================================
# 3D VISUALIZATION
# =========================================================

st.markdown("## 🌍 3D Orbital Position")


earth_x, earth_y, earth_z = create_earth()

fig = go.Figure()


# ---------------------------------------------------------
# EARTH
# ---------------------------------------------------------

fig.add_trace(
    go.Surface(
        x=earth_x,
        y=earth_y,
        z=earth_z,
        colorscale=[
            [0.0, "#061329"],
            [0.35, "#0A3154"],
            [0.65, "#12618A"],
            [1.0, "#1B8EB5"]
        ],
        showscale=False,
        opacity=0.95,
        hoverinfo="skip",
        name="Earth"
    )
)


# ---------------------------------------------------------
# ORBIT PATH
# ---------------------------------------------------------

orbit_x = [
    p["position"][0]
    for p in orbit
]

orbit_y = [
    p["position"][1]
    for p in orbit
]

orbit_z = [
    p["position"][2]
    for p in orbit
]

fig.add_trace(
    go.Scatter3d(
        x=orbit_x,
        y=orbit_y,
        z=orbit_z,
        mode="lines",
        line=dict(
            color="#35D5F5",
            width=5
        ),
        name="Predicted Orbit",
        hoverinfo="skip"
    )
)


# ---------------------------------------------------------
# CURRENT SATELLITE
# ---------------------------------------------------------

sat_x = data["ecef"][0]
sat_y = data["ecef"][1]
sat_z = data["ecef"][2]

fig.add_trace(
    go.Scatter3d(
        x=[sat_x],
        y=[sat_y],
        z=[sat_z],
        mode="markers+text",
        marker=dict(
            size=12,
            color="#FFB454",
            symbol="diamond"
        ),
        text=["🛰️ " + selected_name],
        textposition="top center",
        textfont=dict(
            size=14,
            color="white"
        ),
        name="Current Position",
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"Latitude: {data['latitude']:.4f}°<br>"
            f"Longitude: {data['longitude']:.4f}°<br>"
            f"Altitude: {data['altitude']:.2f} km"
            "<extra></extra>"
        )
    )
)


# ---------------------------------------------------------
# ALTITUDE LINE
# ---------------------------------------------------------

surface_scale = EARTH_RADIUS / (
    EARTH_RADIUS + data["altitude"]
)

surface_x = sat_x * surface_scale
surface_y = sat_y * surface_scale
surface_z = sat_z * surface_scale

fig.add_trace(
    go.Scatter3d(
        x=[
            surface_x,
            sat_x
        ],
        y=[
            surface_y,
            sat_y
        ],
        z=[
            surface_z,
            sat_z
        ],
        mode="lines",
        line=dict(
            color="#FFB454",
            width=4,
            dash="dash"
        ),
        name="Altitude"
    )
)


# ---------------------------------------------------------
# GROUND POINT
# ---------------------------------------------------------

fig.add_trace(
    go.Scatter3d(
        x=[surface_x],
        y=[surface_y],
        z=[surface_z],
        mode="markers",
        marker=dict(
            size=7,
            color="#FF5D73"
        ),
        name="Ground Position",
        hovertemplate=(
            f"Latitude: {data['latitude']:.4f}°<br>"
            f"Longitude: {data['longitude']:.4f}°"
            "<extra></extra>"
        )
    )
)


# =========================================================
# CAMERA / LAYOUT
# =========================================================

fig.update_layout(

    height=750,

    margin=dict(
        l=0,
        r=0,
        t=20,
        b=0
    ),

    paper_bgcolor="#040815",

    scene=dict(

        bgcolor="#040815",

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
                x=1.7,
                y=1.7,
                z=1.25
            )
        )
    ),

    legend=dict(
        font=dict(
            color="white",
            size=13
        ),
        bgcolor="rgba(4,8,21,0.75)"
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# GROUND TRACK
# =========================================================

st.markdown("## 🌐 Ground Track")

track_lat = [
    p["latitude"]
    for p in ground
]

track_lon = [
    p["longitude"]
    for p in ground
]

ground_fig = go.Figure()

ground_fig.add_trace(
    go.Scattergeo(
        lon=track_lon,
        lat=track_lat,
        mode="lines",
        line=dict(
            color="#35D5F5",
            width=3
        ),
        name="Ground Track"
    )
)

ground_fig.add_trace(
    go.Scattergeo(
        lon=[data["longitude"]],
        lat=[data["latitude"]],
        mode="markers",
        marker=dict(
            size=12,
            color="#FFB454"
        ),
        name="Current Position",
        hovertemplate=(
            f"Latitude: {data['latitude']:.4f}°<br>"
            f"Longitude: {data['longitude']:.4f}°"
            "<extra></extra>"
        )
    )
)

ground_fig.update_geos(
    projection_type="equirectangular",
    showland=True,
    showcountries=True,
    showocean=True,
    showcoastlines=True,
    bgcolor="#040815"
)

ground_fig.update_layout(
    height=500,
    margin=dict(
        l=0,
        r=0,
        t=10,
        b=0
    ),
    paper_bgcolor="#040815",
    font=dict(
        color="white"
    )
)

st.plotly_chart(
    ground_fig,
    use_container_width=True
)


# =========================================================
# DATA SOURCE
# =========================================================

st.markdown("---")

st.caption(
    "Orbital data: CelesTrak GP/TLE data mirror • "
    "Propagation: SGP4 • "
    "Coordinate conversion: ECI/TEME → ECEF"
)

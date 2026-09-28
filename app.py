import streamlit as st
import streamlit.components.v1 as components
import requests
import numpy as np
import math
from sgp4.api import Satrec
from datetime import datetime, timezone, timedelta

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="ADYRA OrbitGuard",
    page_icon="🛰️",
    layout="wide"
)

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 50% 0%,
            #10254a 0%,
            #050b18 45%,
            #02040a 100%
        );
    color: white;
}

h1 {
    font-weight: 700;
}

.metric-card {
    background: rgba(15, 30, 55, 0.75);
    border: 1px solid rgba(53,213,245,0.25);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.title("🛰️ ADYRA OrbitGuard")

st.caption(
    "Real-Time Satellite Tracking • Orbital Visualization • Conjunction Analysis"
)


# =========================================================
# TLE DATA
# =========================================================

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

        if (
            line1.startswith("1 ")
            and line2.startswith("2 ")
        ):

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
# GMST
# =========================================================

def gmst_angle(jd):

    T = (
        jd - 2451545.0
    ) / 36525.0

    gmst = (
        280.46061837
        + 360.98564736629
        * (jd - 2451545.0)
        + 0.000387933
        * T * T
        - (T ** 3) / 38710000.0
    )

    return math.radians(
        gmst % 360
    )


# =========================================================
# ECI → ECEF
# =========================================================

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

    return np.array([
        x_ecef,
        y_ecef,
        z
    ])


# =========================================================
# ECEF → GEOGRAPHIC
# =========================================================

def ecef_to_geodetic(position):

    x, y, z = position

    longitude = math.atan2(
        y,
        x
    )

    horizontal = math.sqrt(
        x*x + y*y
    )

    latitude = math.atan2(
        z,
        horizontal
    )

    radius = math.sqrt(
        x*x + y*y + z*z
    )

    altitude = (
        radius - 6378.137
    )

    return (
        math.degrees(latitude),
        math.degrees(longitude),
        altitude
    )


# =========================================================
# CURRENT POSITION
# =========================================================

def get_position(satellite):

    now = datetime.now(
        timezone.utc
    )

    jd = julian_date(now)

    jd_int = int(jd)
    fraction = jd - jd_int

    error, position, velocity = satellite.sgp4(
        jd_int,
        fraction
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
        "ecef": ecef,
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "velocity": speed,
        "time": now
    }


# =========================================================
# ORBIT / GROUND TRACK
# =========================================================

def calculate_track(
    satellite,
    before_minutes=30,
    after_minutes=180,
    step=1
):

    now = datetime.now(
        timezone.utc
    )

    points = []

    for minute in np.arange(
        -before_minutes,
        after_minutes,
        step
    ):

        dt = (
            now +
            timedelta(
                minutes=float(minute)
            )
        )

        jd = julian_date(dt)

        jd_int = int(jd)
        fraction = jd - jd_int

        error, position, velocity = satellite.sgp4(
            jd_int,
            fraction
        )

        if error != 0:
            continue

        position = np.array(position)

        ecef = eci_to_ecef(
            position,
            jd
        )

        latitude, longitude, altitude = (
            ecef_to_geodetic(ecef)
        )

        longitude = (
            (longitude + 180)
            % 360
        ) - 180

        points.append({
            "lat": latitude,
            "lon": longitude,
            "alt": altitude,
            "time": dt.isoformat()
        })

    return points


# =========================================================
# LOAD
# =========================================================

try:

    satellites = load_satellites()

except Exception as e:

    st.error(
        "Could not load satellite data."
    )

    st.code(
        str(e)
    )

    st.stop()


if len(satellites) == 0:

    st.error(
        "No satellite data available."
    )

    st.stop()


# =========================================================
# SELECT SATELLITE
# =========================================================

names = [
    x["name"]
    for x in satellites
]

selected_name = st.selectbox(
    "🛰️ Select Satellite",
    names
)

selected = next(
    x for x in satellites
    if x["name"] == selected_name
)


# =========================================================
# SGP4
# =========================================================

satellite = Satrec.twoline2rv(
    selected["line1"],
    selected["line2"]
)


# =========================================================
# CURRENT DATA
# =========================================================

data = get_position(
    satellite
)

if data is None:

    st.error(
        "Unable to calculate satellite position."
    )

    st.stop()


# =========================================================
# METRICS
# =========================================================

st.markdown("### 📍 Live Satellite Position")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Latitude",
        f"{data['latitude']:.4f}°"
    )

with c2:
    st.metric(
        "Longitude",
        f"{data['longitude']:.4f}°"
    )

with c3:
    st.metric(
        "Altitude",
        f"{data['altitude']:.2f} km"
    )

with c4:
    st.metric(
        "Velocity",
        f"{data['velocity']:.3f} km/s"
    )


# =========================================================
# ORBIT TRACK
# =========================================================

track = calculate_track(
    satellite,
    before_minutes=30,
    after_minutes=180,
    step=1
)


# =========================================================
# SEND DATA TO CESIUM
# =========================================================

import json

cesium_data = {
    "satellite": selected_name,

    "latitude": data["latitude"],

    "longitude": data["longitude"],

    "altitude": data["altitude"],

    "velocity": data["velocity"],

    "timestamp": data["time"].isoformat(),

    "track": track
}


# =========================================================
# CESIUM GLOBE
# =========================================================

st.markdown(
    "### 🌍 Real 3D Earth"
)

st.caption(
    "Drag to rotate • Scroll to zoom • "
    "Right-click/drag to tilt"
)


# Read HTML template
with open(
    "globe.html",
    "r",
    encoding="utf-8"
) as f:

    html = f.read()


# Inject JSON
html = html.replace(
    "__SATELLITE_DATA__",
    json.dumps(
        cesium_data
    )
)


components.html(
    html,
    height=760,
    scrolling=False
)


# =========================================================
# DETAILS
# =========================================================

st.markdown("---")

st.markdown(
    "### 🛰️ Satellite Details"
)

d1, d2 = st.columns(2)

with d1:

    st.write(
        "**Satellite:**",
        selected_name
    )

    st.write(
        "**Latitude:**",
        f"{data['latitude']:.5f}°"
    )

    st.write(
        "**Longitude:**",
        f"{data['longitude']:.5f}°"
    )


with d2:

    st.write(
        "**Altitude:**",
        f"{data['altitude']:.2f} km"
    )

    st.write(
        "**Velocity:**",
        f"{data['velocity']:.3f} km/s"
    )

    st.write(
        "**UTC:**",
        data["time"].strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


# =========================================================
# TLE
# =========================================================

with st.expander(
    "🔧 Raw TLE Data"
):

    st.code(
        selected["line1"]
    )

    st.code(
        selected["line2"]
    )


st.caption(
    "Propagation: SGP4 • Orbital elements: TLE"
)

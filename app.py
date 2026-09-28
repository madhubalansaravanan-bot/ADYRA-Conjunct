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


@st.cache_data(ttl=3600)
def get_satellites():

    sources = [
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
    ]

    last_error = None

    for url in sources:
        try:
            response = requests.get(
                url,
                timeout=60,
                headers={
                    "User-Agent": "ADYRA-OrbitGuard/1.0"
                }
            )

            response.raise_for_status()

            lines = [
                line.strip()
                for line in response.text.splitlines()
                if line.strip()
            ]

            satellites = []

            for i in range(0, len(lines) - 2, 3):

                if (
                    lines[i + 1].startswith("1 ")
                    and lines[i + 2].startswith("2 ")
                ):
                    satellites.append({
                        "name": lines[i],
                        "line1": lines[i + 1],
                        "line2": lines[i + 2]
                    })

            if satellites:
                return satellites

        except Exception as e:
            last_error = e

    raise Exception(
        f"All satellite data sources failed. Last error: {last_error}"
    )

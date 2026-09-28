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
st.caption("Real satellite tracking and conjunction screening")


# ---------------------------------------------------------
# LOAD REAL SATELLITE DATA
# ---------------------------------------------------------

@st.cache_data(ttl=3600)
def load_satellites():

    url = (
        "https://celestrak.org/NORAD/elements/"
        "gp.php?GROUP=STATIONS&FORMAT=JSON"
    )

    response = requests.get(
        url,
        timeout=60,
        headers={
            "User-Agent": "ADYRA-OrbitGuard/1.0"
        }
    )

    response.raise_for_status()

    data = response.json()

    satellites = []

    for sat in data:

        try:
            satellites.append({
                "name": sat["OBJECT_NAME"],
                "norad": sat.get("NORAD_CAT_ID"),
                "epoch": sat.get("EPOCH"),
                "inclination": sat.get("INCLINATION"),
                "raan": sat.get("RA_OF_ASC_NODE"),
                "eccentricity": sat.get("ECCENTRICITY"),
                "arg_perigee": sat.get("ARG_OF_PERICENTER"),
                "mean_anomaly": sat.get("MEAN_ANOMALY"),
                "mean_motion": sat.get("MEAN_MOTION"),
            })

        except KeyError:
            continue

    return satellites


# ---------------------------------------------------------
# CREATE SATELLITE OBJECT FROM OMM DATA
# ---------------------------------------------------------

def create_satellite(sat):

    epoch = datetime.fromisoformat(
        sat["epoch"].replace("Z", "+00:00")
    )

    year = epoch.year
    day_of_year = (
        epoch.timetuple().tm_yday
        + (
            epoch.hour * 3600
            + epoch.minute * 60
            + epoch.second
            + epoch.microsecond / 1e6
        ) / 86400
    )

    # SGP4 satellite object
    satellite = Satrec()

    satellite.sgp4init(
        0,
        "i",
        float(sat["mean_motion"]) ** (2 / 3),
        0.0,
        float(sat["eccentricity"]),
        np.radians(float(sat["arg_perigee"])),
        np.radians(float(sat["inclination"])),
        np.radians(float(sat["mean_anomaly"])),
        float(sat["mean_motion"]),
        0.0,
        np.radians(float(sat["raan"]))
    )

    return satellite


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

try:

    satellites = load_satellites()

    if not satellites:
        st.error("No satellite data was returned.")
        st.stop()

except Exception as e:

    st.error("Unable to connect to the satellite data source.")

    st.code(str(e))

    st.info(
        "The application is running, but the external satellite "
        "data service is currently unreachable."
    )

    st.stop()


# ---------------------------------------------------------
# SATELLITE SEARCH
# ---------------------------------------------------------

st.success(
    f"🛰️ Real satellite data loaded: {len(satellites)} objects"
)

names = [
    sat["name"]
    for sat in satellites
]


selected_name = st.selectbox(
    "🔎 Select Satellite",
    names
)


selected = next(
    sat for sat in satellites
    if sat["name"] == selected_name
)


# ---------------------------------------------------------
# SATELLITE INFORMATION
# ---------------------------------------------------------

st.markdown("## 🛰️ Satellite Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Satellite",
    selected["name"]
)

col2.metric(
    "NORAD ID",
    selected["norad"]
)

col3.metric(
    "Epoch",
    selected["epoch"]
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "Inclination",
    f"{float(selected['inclination']):.2f}°"
)

col5.metric(
    "Eccentricity",
    f"{float(selected['eccentricity']):.6f}"
)

col6.metric(
    "Mean Motion",
    f"{float(selected['mean_motion']):.4f} rev/day"
)


# ---------------------------------------------------------
# DATA SOURCE
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Data source: CelesTrak General Perturbations (GP) data. "
    "Orbital position calculations will use SGP4."
)

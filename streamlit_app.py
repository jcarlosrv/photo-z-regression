from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

MODEL_PATH = Path("models/xgb_photoz.pkl")
SAMPLE_PATH = Path("data/sample_small.csv")

R_MIN = 14.0
R_MAX = 22.0

RANGE_COLORS = {
    "In range": "#2ecc71",
    "Below range": "#3498db",
    "Above range": "#e74c3c",
}


def classify_range(r_mag):
    if r_mag < R_MIN:
        return "Below range"
    elif r_mag > R_MAX:
        return "Above range"
    else:
        return "In range"


st.set_page_config(page_title="Photo-z Predictor", layout="wide")

st.title("Photometric Redshift Predictor")
st.markdown(
    "Predicts galaxy redshift from SDSS photometric magnitudes (u, g, r, i, z) "
    "using an XGBoost model trained on the SDSS DR18 galaxy sample."
)


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


def add_colors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
    return df


model, features = load_model()

tab_single, tab_batch, tab_about = st.tabs(["Single prediction", "Batch (CSV)", "About"])

with tab_single:
    st.subheader("Enter photometric magnitudes")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        u = st.number_input("u", value=19.5, step=0.1)
    with col2:
        g = st.number_input("g", value=18.2, step=0.1)
    with col3:
        r = st.number_input("r", value=17.5, step=0.1)
    with col4:
        i = st.number_input("i", value=17.1, step=0.1)
    with col5:
        z = st.number_input("z", value=16.9, step=0.1)

    if st.button("Predict redshift"):
        row = pd.DataFrame([{"u": u, "g": g, "r": r, "i": i, "z": z}])
        row = add_colors(row)
        pred = model.predict(row[features])[0]
        st.metric("Predicted redshift z", f"{pred:.4f}")

        label = classify_range(r)
        color = RANGE_COLORS[label]
        st.markdown(
            f'<span style="background-color:{color};color:white;padding:4px 12px;'
            f'border-radius:8px;font-weight:bold;">{label}</span>'
            f"&nbsp; Based on r-band magnitude ({r:.1f}). "
            f"Valid range: {R_MIN} - {R_MAX}",
            unsafe_allow_html=True,
        )

with tab_batch:
    st.subheader("Upload a CSV with columns u, g, r, i, z")
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is None and SAMPLE_PATH.exists():
        if st.button("Use bundled sample"):
            df = pd.read_csv(SAMPLE_PATH)
        else:
            df = None
    elif uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = None

    if df is not None:
        df = add_colors(df)
        df["predicted_z"] = model.predict(df[features])
        df["range_label"] = df["r"].apply(classify_range)
        st.dataframe(df[["u", "g", "r", "i", "z", "predicted_z", "range_label"]].head(20))

        if "redshift" in df.columns:
            fig = px.scatter(
                df, x="redshift", y="predicted_z",
                color="range_label",
                color_discrete_map=RANGE_COLORS,
                opacity=0.5,
                labels={
                    "redshift": "True z",
                    "predicted_z": "Predicted z",
                    "range_label": "Magnitude range",
                },
                title="Predicted vs true redshift",
            )
            fig.add_shape(
                type="line", x0=0, y0=0, x1=df["redshift"].max(),
                y1=df["redshift"].max(), line=dict(dash="dash"),
            )
            st.plotly_chart(fig, use_container_width=True)
            mae = np.mean(np.abs(df["redshift"] - df["predicted_z"]))
            st.metric("MAE on uploaded batch", f"{mae:.4f}")

        st.download_button(
            "Download predictions CSV",
            df.to_csv(index=False),
            file_name="predictions.csv",
        )

with tab_about:
    st.markdown("""
**What is photometric redshift?**

Redshift tells us how fast a galaxy is moving away from us - and by Hubble's law, how far away it is.
The gold-standard way to measure it is **spectroscopy** (splitting light into a spectrum and
measuring line shifts), but that's expensive and slow.

**Photometric redshift** estimates the same quantity from broad-band filter magnitudes
(here: SDSS *u, g, r, i, z*) - much cheaper, applicable to orders of magnitude more galaxies.

**Model.** XGBoost regressor trained on SDSS DR18 galaxies with spectroscopic labels.
Features: the 5 magnitudes plus 4 color indices (u-g, g-r, r-i, i-z).

**Magnitude range labels.** Each prediction is tagged as *In range* (14-22 in r-band),
*Below range* (too bright, saturation regime), or *Above range* (too faint, extrapolation).
Predictions outside the training range are less reliable.

**Limitations.** Model is trained on the SDSS magnitude range and redshift < 1.
Predictions outside this range are extrapolation and unreliable.

Code: [github.com/jcarlosrv/photo-z-regression](https://github.com/jcarlosrv/photo-z-regression)
    """)
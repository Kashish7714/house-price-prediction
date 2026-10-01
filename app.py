# app.py — Streamlit frontend for the California House Price Prediction model.
# Run with:  streamlit run app.py

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="wide",
)

# ── Load model bundle ─────────────────────────────────────────────────────────
# @st.cache_resource ensures the file is read from disk only once per session,
# not on every user interaction.
@st.cache_resource
def load_model():
    """Load the trained model, scaler, and metadata from disk."""
    return joblib.load("model.joblib")

# Attempt to load; show a friendly error if train.py has not been run yet.
try:
    bundle = load_model()
except FileNotFoundError:
    st.error(
        "⚠️ **model.joblib not found.**  "
        "Please run `python train.py` first to train and save the model."
    )
    st.stop()

# Unpack the bundle
model         = bundle["model"]
scaler        = bundle["scaler"]
feature_names = bundle["feature_names"]
feat_min      = bundle["feature_min"]
feat_max      = bundle["feature_max"]
feat_mean     = bundle["feature_mean"]
r2_stored     = bundle["r2"]
mae_stored    = bundle["mae"]

# ── Human-readable labels and help text for each feature ─────────────────────
FEATURE_META = {
    "MedInc":     ("Median Income",          "Median income in block group (tens of thousands USD)"),
    "HouseAge":   ("House Age (years)",       "Median age of houses in the block group"),
    "AveRooms":   ("Avg Rooms per Household", "Average number of rooms per household"),
    "AveBedrms":  ("Avg Bedrooms per House",  "Average number of bedrooms per household"),
    "Population": ("Block Population",        "Total population of the block group"),
    "AveOccup":   ("Avg Occupancy",           "Average number of household members"),
    "Latitude":   ("Latitude",                "Block group latitude (California: ~32 – 42 N)"),
    "Longitude":  ("Longitude",               "Block group longitude (California: ~-124 – -114)"),
}

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🏡 California House Price Predictor")
st.markdown(
    "Adjust the sliders to describe a neighbourhood, then click **Predict Price** "
    "to estimate the median house value."
)

# ── Sidebar: input sliders ────────────────────────────────────────────────────
st.sidebar.header("🔧 Input Features")
st.sidebar.markdown("Use the sliders below to set neighbourhood characteristics.")

input_values = {}  # Will hold one float per feature

for feat in feature_names:
    label, help_text = FEATURE_META.get(feat, (feat, ""))

    f_min  = float(feat_min[feat])
    f_max  = float(feat_max[feat])
    f_mean = float(feat_mean[feat])

    # Round step size based on the feature's range so sliders feel natural
    feature_range = f_max - f_min
    step = round(feature_range / 200, 4) if feature_range > 0 else 0.01

    input_values[feat] = st.sidebar.slider(
        label=label,
        min_value=f_min,
        max_value=f_max,
        value=f_mean,          # Default to the dataset mean
        step=step,
        help=help_text,
        format="%.4f",
    )

# ── Main area: two columns ────────────────────────────────────────────────────
col_pred, col_map = st.columns([1, 1], gap="large")

# ── Left column: prediction ───────────────────────────────────────────────────
with col_pred:
    st.subheader("💰 Price Prediction")

    # Show a summary table of current slider values
    summary_df = pd.DataFrame(
        {
            "Feature": [FEATURE_META.get(f, (f,))[0] for f in feature_names],
            "Value":   [round(input_values[f], 4) for f in feature_names],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Predict button ────────────────────────────────────────────────────────
    if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
        # Build a single-row NumPy array in the same column order as training
        raw_input = np.array([[input_values[f] for f in feature_names]])

        # Scale the input using the same scaler fitted during training
        scaled_input = scaler.transform(raw_input)

        # Run inference
        prediction = model.predict(scaled_input)[0]

        # The target variable is in $100,000 units; convert to dollars
        price_dollars = prediction * 100_000

        st.success(f"### Estimated Median House Value")
        st.metric(
            label="Predicted Price",
            value=f"${price_dollars:,.0f}",
            help="Median house value for the described block group",
        )

        # Confidence context: show feature importance bar chart
        st.markdown("#### Feature Importances")
        importances = model.feature_importances_
        importance_df = pd.DataFrame(
            {
                "Feature":    [FEATURE_META.get(f, (f,))[0] for f in feature_names],
                "Importance": importances,
            }
        ).sort_values("Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature")["Importance"])

# ── Right column: map ─────────────────────────────────────────────────────────
with col_map:
    st.subheader("📍 Selected Location")

    lat = input_values["Latitude"]
    lon = input_values["Longitude"]

    # st.map expects a DataFrame with columns named 'lat' and 'lon'
    location_df = pd.DataFrame({"lat": [lat], "lon": [lon]})

    st.map(location_df, zoom=7)
    st.caption(f"Latitude: {lat:.4f} | Longitude: {lon:.4f}")

# ── Model Performance section ─────────────────────────────────────────────────
st.divider()
st.subheader("📊 Model Performance (Test Set)")

perf_col1, perf_col2, perf_col3 = st.columns(3)

with perf_col1:
    st.metric(
        label="R² Score",
        value=f"{r2_stored:.4f}",
        help="1.0 = perfect predictions. Values above 0.80 are generally good.",
    )

with perf_col2:
    st.metric(
        label="Mean Absolute Error",
        value=f"${mae_stored * 100_000:,.0f}",
        help="Average absolute difference between predicted and actual prices.",
    )

with perf_col3:
    st.metric(
        label="Algorithm",
        value="Random Forest",
        help="Ensemble of 100 decision trees with StandardScaler preprocessing.",
    )

st.markdown(
    "_Model trained on the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) "
    "from scikit-learn. Target variable is the median house value in $100,000 units._"
)

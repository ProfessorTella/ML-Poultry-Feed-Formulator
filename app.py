import streamlit as st
import pandas as pd
import time
import numpy as np
import joblib
import random
import json
import os
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.multioutput import MultiOutputRegressor
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ===============================
# CONFIGURATION
# ===============================
CONFIG_PATH = "data/feed_config.json"   # your JSON file path
BASE_MODEL_DIR = "data/models"          # each diet type has its subfolder

with open(CONFIG_PATH, "r") as f:
    FEED_CONFIG = json.load(f)

st.set_page_config(page_title="SmartFeed — Multi Diet Formulator", page_icon="🌾", layout="wide")
st.title("🌾 SmartFeed — Poultry Least Cost Feed Formulator")

# ===============================
# USER SELECTS FEED TYPE
# ===============================
feed_type = st.selectbox(
    "Select Feed Type:",
    options=list(FEED_CONFIG.keys()),
    format_func=lambda x: x.replace("_", " ").title()
)
config = FEED_CONFIG[feed_type]

ingredient_cols = config["ingredient_cols"]
nutrient_constraints = config["nutrient_constraints"]
bounds_map = config["bounds_map"]
nutrient_df = pd.DataFrame(config["nutrient_data"]).set_index("Ingredient")
target_cols = list(nutrient_constraints.keys())

# Ensure bounds for all ingredients (fill missing with defaults)
default_bounds = (0, 50)
bounds = [bounds_map.get(i, default_bounds) for i in ingredient_cols]

# ===============================
# MODEL LOADING / TRAINING
# ===============================
def build_features(df):
    expanded_frames = []
    for nut in nutrient_df.columns:
        nutrient_values = nutrient_df[nut].values / 100.0
        expanded = df[ingredient_cols].multiply(nutrient_values, axis=1)
        expanded.columns = [f"{ing}_{nut}" for ing in ingredient_cols]
        expanded_frames.append(expanded)
    return pd.concat([df[ingredient_cols]] + expanded_frames, axis=1)

model_dir = os.path.join(BASE_MODEL_DIR, feed_type)
os.makedirs(model_dir, exist_ok=True)
MODEL_PATH = os.path.join(model_dir, "nutrient_model.pkl")
FEATURE_PATH = os.path.join(model_dir, "feature_names.pkl")

# Load or train model
try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
except:
    # Assume user has an Excel file matching feed_type name
    data_path = f"{feed_type.replace('_', ' ')} diets.xlsx"
    if not os.path.exists(data_path):
        st.error(f"Missing dataset file: {data_path}")
        st.stop()

    feed_data = pd.read_excel(data_path)
    X = build_features(feed_data)
    y = feed_data[target_cols]
    feature_names = X.columns.tolist()

    kernel = RBF()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    model = MultiOutputRegressor(gpr)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, FEATURE_PATH)

# ===============================
# OPTIMIZATION FUNCTIONS
# ===============================
# ======================================
# ✅ Corrected prediction function
# ======================================
def predict_nutrients(x):
    """Predict nutrients from ingredient inclusion vector."""
    df = pd.DataFrame([x], columns=ingredient_cols)
    df = build_features(df)  # same feature expansion used in training
    df = df.reindex(columns=feature_names, fill_value=0)  # keep column order consistent
    pred = model.predict(df)[0]
    return np.maximum(pred, 0)  # clip any negative predictions to zero


# ======================================
# ✅ Robust optimization function
# ======================================
def optimize_feed(price_vector):
    """Perform least-cost formulation given ingredient prices."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    progress = 0

    def objective(x):
        return np.dot(price_vector, x)

    def make_constraints():
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 100}]
        for i, nut in enumerate(target_cols):
            min_val, max_val = nutrient_constraints[nut]
            eps = 1 if nut == "ME_kcal_per_kg" else 1e-3
            # keep i fixed in lambda using default arg
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i, min_val=min_val: predict_nutrients(x)[i] - (min_val - eps)})
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i, max_val=max_val: (max_val + eps) - predict_nutrients(x)[i]})
        return cons

    # --- Initial guess (normalized to 100%)
    x0 = np.full(len(ingredient_cols), 100 / len(ingredient_cols))
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    if np.sum(x0) != 100:
        x0 = x0 / np.sum(x0) * 100

    constraints = make_constraints()

     # --- Simulate progress animation
    p = random.randint(5, 10)
    for pct in range(0, 90, p):
        progress_bar.progress(pct)
        status_text.text(f"🔧 Optimizing feed... {pct}%")
        time.sleep(2.5)  # just a visual delay
    

    result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 5000, 'ftol': 1e-1})

    # --- Clear progress indicators
    progress_bar.progress(100)
    status_text.text("✅ Optimization complete!")

    if not result.success:
        st.warning(f"⚠️ Optimizer warning: {result.message}")

    nutrients = predict_nutrients(result.x)
    res = {
        **{ingredient_cols[i]: result.x[i] for i in range(len(ingredient_cols))},
        **{target_cols[j]: nutrients[j] for j in range(len(target_cols))},
        "Total_Cost": np.dot(price_vector, result.x)
    }
    return res


# ======================================
# ✅ Streamlit interface
# ======================================
st.markdown(f"### Enter Price per kg (₦) for **{feed_type.replace('_', ' ').title()}**")

with st.form("formulate"):
    cols = st.columns(3)
    prices = []
    for i, ing in enumerate(ingredient_cols):
        with cols[i % 3]:
            prices.append(st.number_input(f"{ing}", min_value=0.0, value=100.0, step=1.0))
    submitted = st.form_submit_button("🧮 Formulate Feed")

if submitted:
    # --- Ensure correct length and numeric conversion
    price_vector = np.array(prices, dtype=float)
    if len(price_vector) != len(ingredient_cols):
        st.error("❌ Price entries do not match ingredient count.")
        st.stop()


    with st.spinner("Optimizing feed formulation... ⏳"):
        res = optimize_feed(price_vector)

    st.success("✅ Optimization Complete!")

    # --- Total Cost
    st.subheader("💰 Total Feed Cost")
    st.metric("Cost per 100 kg", f"₦{res['Total_Cost']:,.2f}")

    # --- Ingredient Table
    st.subheader("📦 Ingredient Inclusion Levels (%)")
    df_ing = pd.DataFrame({
        "Ingredient": ingredient_cols,
        "Inclusion (%)": [res[i] for i in ingredient_cols]
    })
    st.dataframe(df_ing, hide_index=True, use_container_width=True)

    # --- Clean, compact Pie Chart
    import plotly.express as px
    fig = px.pie(
        df_ing,
        names="Ingredient",
        values="Inclusion (%)",
        title="Ingredient Inclusion Ratio",
        hole=0.3
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(showlegend=True, legend=dict(font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

    # --- Nutrient Composition
    st.subheader("🌿 Predicted Nutrient Composition")
    df_nut = pd.DataFrame({
        "Nutrient": target_cols,
        "Value": [res[n] for n in target_cols]
    })
    st.dataframe(df_nut, hide_index=True, use_container_width=True)

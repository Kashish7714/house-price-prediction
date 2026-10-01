# train.py — Load the California Housing dataset, train a regression model,
# evaluate it, and save the trained model + scaler to disk.

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)

X = housing.data          # Feature matrix (8 numeric columns)
y = housing.target        # Target: median house value in $100,000 units

# ── 2. Train / test split ─────────────────────────────────────────────────────
# 80 % training data, 20 % held-out test data; fixed seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# ── 3. Feature scaling ────────────────────────────────────────────────────────
# StandardScaler standardises each feature to zero mean and unit variance.
# We fit ONLY on training data to avoid data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Train model ────────────────────────────────────────────────────────────
# RandomForestRegressor averages many decision trees, which reduces overfitting
# and handles non-linear relationships well.
print("\nTraining RandomForestRegressor (n_estimators=100)...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n-- Model Performance on Test Set --")
print(f"  R2 Score             : {r2:.4f}")
print(f"  Mean Absolute Error  : ${mae * 100_000:,.0f}  (in dollars)")

# ── 6. Persist model + scaler + metadata ─────────────────────────────────────
# We bundle the scaler together with the model so the app only needs to load
# one file and the transformation step is never accidentally skipped.
feature_names = list(X.columns)

payload = {
    "model":         model,
    "scaler":        scaler,
    "feature_names": feature_names,
    "r2":            round(r2, 4),
    "mae":           round(mae, 4),
    # Store per-feature min / max so the Streamlit sliders have sensible ranges
    "feature_min":   X.min().to_dict(),
    "feature_max":   X.max().to_dict(),
    "feature_mean":  X.mean().to_dict(),
}

joblib.dump(payload, "model.joblib")
print("\nModel saved to model.joblib")

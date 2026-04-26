from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

DATA_PATH = Path("data/SDSS_DR18_pt1.csv")
MODEL_PATH = Path("models/xgb_photoz.pkl")
RANDOM_STATE = 42


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["class"] == "GALAXY"].copy()

    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]

    df = df.dropna(subset=["u", "g", "r", "i", "z", "redshift"])
    df = df[(df["redshift"] > 0) & (df["redshift"] < 1.0)]
    return df


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:20s} | MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    df = load_and_prepare(DATA_PATH)
    print(f"Loaded {len(df):,} galaxy rows after filtering.")

    features = ["u", "g", "r", "i", "z", "u_g", "g_r", "r_i", "i_z"]
    X = df[features]
    y = df["redshift"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\nModel comparison on hold-out set:\n")

    lr = LinearRegression().fit(X_train, y_train)
    evaluate("LinearRegression", y_test, lr.predict(X_test))

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE
    ).fit(X_train, y_train)
    evaluate("RandomForest", y_test, rf.predict(X_test))

    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ).fit(X_train, y_train)
    evaluate("XGBoost", y_test, xgb.predict(X_test))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": xgb, "features": features}, MODEL_PATH)
    print(f"\nSaved XGBoost model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
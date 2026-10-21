import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
from src.data_loader import load_dataset, PROJECT_ROOT
from src.preprocessing import clean_data

TARGET = "Temperature_C"
NUM = ["Month", "Day", "Year", "Day_Of_Year", "Humidity_pct", "Pressure_hPa", "Wind_Speed_kmph", "Cloud_Cover_pct", "Rainfall_mm", "Visibility_km", "Solar_Radiation_Wm2", "Air_Quality_Index", "Dew_Point_C"]
CAT = ["Season"]

def train(data_path=None):
    df = clean_data(load_dataset(data_path) if data_path else load_dataset())
    usable = df.dropna(subset=[TARGET]).copy()
    if len(usable) < 10: raise ValueError("At least 10 valid temperature rows are needed for evaluation.")
    features = [c for c in NUM + CAT if c in usable.columns]
    X, y = usable[features], usable[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    numeric = [c for c in features if c in NUM]; categorical = [c for c in features if c in CAT]
    transformers = []
    if numeric: transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical: transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    prep = ColumnTransformer(transformers)
    estimators = {"Linear Regression": LinearRegression(), "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)}
    results, fitted, predictions = {}, {}, {}
    for name, estimator in estimators.items():
        pipe = Pipeline([("preprocessor", clone(prep)), ("model", estimator)])
        pipe.fit(Xtr, ytr); pred = pipe.predict(Xte)
        results[name] = {"MAE": float(mean_absolute_error(yte, pred)), "RMSE": float(np.sqrt(mean_squared_error(yte, pred))), "R2": float(r2_score(yte, pred))}
        fitted[name] = pipe
        predictions[name] = pred
    (PROJECT_ROOT / "models").mkdir(exist_ok=True); (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    dump({"pipeline": fitted["Random Forest"], "features": features, "model_name": "Random Forest", "target": TARGET}, PROJECT_ROOT / "models" / "temperature_model.joblib")
    (PROJECT_ROOT / "reports" / "model_metrics.json").write_text(json.dumps({"metrics": results, "rows": len(usable), "train_rows": len(Xtr), "test_rows": len(Xte), "method": "random 80/20 split, seed 42"}, indent=2))
    reports = PROJECT_ROOT / "reports"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(yte, predictions["Random Forest"], alpha=.65, color="#e76f51")
    low, high = float(min(yte.min(), predictions["Random Forest"].min())), float(max(yte.max(), predictions["Random Forest"].max()))
    axes[0].plot([low, high], [low, high], "--", color="#243b53"); axes[0].set(xlabel="Observed °C", ylabel="Estimated °C", title="Random Forest: observed vs estimated")
    residuals = yte.to_numpy() - predictions["Random Forest"]
    axes[1].scatter(predictions["Random Forest"], residuals, alpha=.65, color="#457b9d"); axes[1].axhline(0, linestyle="--", color="#243b53"); axes[1].set(xlabel="Estimated °C", ylabel="Residual °C", title="Random Forest residuals")
    fig.tight_layout(); fig.savefig(reports / "model_diagnostics.png", dpi=160); plt.close(fig)
    summary = {"rows": int(len(df)), "columns": list(df.columns), "dtypes": {k: str(v) for k,v in df.dtypes.items()}, "missing_values": {k: int(v) for k,v in df.isna().sum().items()}, "duplicate_rows_after_cleaning": 0, "numeric_summary": df.select_dtypes(include="number").describe().to_dict(), "climate_condition_counts": df["Climate_Condition"].value_counts(dropna=False).to_dict() if "Climate_Condition" in df else {}, "limitations": ["No geographic coordinates or station identifier are assumed.", "Continuity and geographic consistency of dated rows are unverified.", "Climate_Condition labels are not verified official heatwave labels."]}
    (reports / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].hist(temps if (temps := df[TARGET].dropna()) is not None else [], bins=25, color="#e76f51"); axes[0,0].set(title="Temperature distribution", xlabel="°C", ylabel="Rows")
    if "Month" in df: df.groupby("Month")[TARGET].mean().plot(ax=axes[0,1], marker="o", color="#457b9d"); axes[0,1].set(title="Mean temperature by month", xlabel="Month", ylabel="°C")
    if "Season" in df: df.boxplot(column=TARGET, by="Season", ax=axes[1,0]); axes[1,0].set(title="Temperature by season", xlabel="Season", ylabel="°C")
    corr = df.select_dtypes(include="number").corr(); axes[1,1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); axes[1,1].set(title="Numeric feature correlation", xticks=range(len(corr)), xticklabels=corr.columns, yticks=range(len(corr)), yticklabels=corr.columns); axes[1,1].tick_params(axis="x", labelrotation=90)
    fig.tight_layout(); fig.savefig(reports / "eda_overview.png", dpi=160); plt.close(fig)
    return results

if __name__ == "__main__":
    print(json.dumps(train(), indent=2))

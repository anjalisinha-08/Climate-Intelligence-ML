# Climate Intelligence for Heatwave Monitoring, Prediction, and Early Warning

A local Streamlit application for historical weather analysis, contemporaneous temperature estimation, configurable heat-risk demonstrations, and deterministic advisory templates.

## Dataset and scope

Place `climate_probabilistic_latent_variable_dataset (1).csv` in `data/`, or set `CLIMATE_DATA_PATH` to its location. Expected columns are Year, Month, Day, Season, Temperature_C, Humidity_pct, Pressure_hPa, Wind_Speed_kmph, Cloud_Cover_pct, Rainfall_mm, Visibility_km, Solar_Radiation_Wm2, Air_Quality_Index, Dew_Point_C, and Climate_Condition. The loader validates the schema. Cleaning removes duplicate rows, coerces malformed numerics to missing, bounds-checks common physical ranges, normalizes category casing, validates calendar dates, and derives day-of-year and missing calendar seasons.

No file was present in this workspace when this project was created, so no data inspection, fitted model, or actual performance metrics are included. The app will not substitute fake data. Verify the actual CSV's distributions, labels, and dates when adding it. The model uses `Climate_Condition` as a contemporaneous feature and excludes `Temperature_C` from inputs; if that category was itself derived from temperature, remove it to avoid target leakage.

## Architecture

- `src/data_loader.py`: configurable CSV loading and schema checks.
- `src/preprocessing.py`: shared data cleaning and calendar features.
- `src/train_temperature_model.py`: seeded 80/20 benchmark of Linear Regression and Random Forest, MAE/RMSE/R², serialized pipeline and metrics.
- `src/prediction.py`: inference through the saved pipeline.
- `src/risk_engine.py`, `src/advisories.py`: transparent configurable categories and deterministic precautions.
- `app.py`: six-page Streamlit dashboard for overview, predictions, analytics, monitoring, advisories, and evaluation.
- `tests/`: preprocessing, model-missing behavior, and risk category tests.

The app's random split is an evaluation of estimating a contemporaneous target from other weather observations. The brief gives no evidence these rows are continuous daily observations or one location, so the system does not claim forecast skill. `Hot` is not treated as an official heatwave label; no supervised heatwave classifier is trained. The configurable temperature threshold and Normal/Watch/Warning/Severe risk categories are educational and are not official IMD criteria.

## Run on macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data models reports
cp "/path/to/climate_probabilistic_latent_variable_dataset (1).csv" data/
python -m src.train_temperature_model
streamlit run app.py
```

Alternatively, set the source CSV path before training and launching:

```bash
export CLIMATE_DATA_PATH="/path/to/climate_probabilistic_latent_variable_dataset (1).csv"
python -m src.train_temperature_model
streamlit run app.py
```

Run checks with `pytest -q`. Model training is separate from dashboard execution. The trained pipeline is saved to `models/temperature_model.joblib`; metrics are in `reports/model_metrics.json`.

## Limitations and future work

This dataset description does not establish coordinates, station identity, continuous chronology, official heatwave labels, or verified observations. The dashboard therefore offers no geographic hotspot map, live feed, IMD integration, or multi-day forecast. Application thresholds need to be configured against verified, applicable official guidance before operational use. Advisory timestamps are not inferred from date-only dataset rows.

For real forecasting, obtain quality-controlled location-specific observations with timestamps, construct lagged temperatures and rolling statistics using only past observations, define a forecast horizon (such as next-day maximum), split chronologically, and score separately for each horizon. A future source adapter should validate provenance, units, timestamps, and location before supplying observations. Official criteria and warnings should come from verified official guidance. No LLM or paid API is required here.

# Climate Intelligence 🌡️

### Turning weather observations into clear, explainable heat-risk insights.

Climate Intelligence is a local machine-learning dashboard for exploring weather data, estimating temperature from current weather conditions, and demonstrating configurable heat-risk advisories. It combines a reproducible scikit-learn workflow with an interactive Streamlit interface.

> **Scope:** This is an educational project. Temperature estimates use contemporaneous inputs; they are not future forecasts. Configurable risk categories are not official IMD alerts.

<!-- SCREENSHOT: Save the dashboard Overview page as `docs/screenshots/01-overview.png` and place it here. -->
<!-- ![Climate Intelligence overview](docs/screenshots/01-overview.png) -->

## At a glance

- **1,000** dataset records · **15** source columns
- **Two** temperature regressors compared: Linear Regression and Random Forest
- **Six** dashboard areas: Overview, ML Predictions, Climate Analytics, Heatwave Monitoring, Early Warning Centre, and Model Evaluation
- **No fabricated live data, locations, forecasts, or official heatwave labels**

## Dashboard tour

### 1. Overview — understand the dataset

Review record counts, observed temperature summaries, the temperature distribution, monthly patterns, and the latest calendar dates represented in the file. These are dataset statistics, not live weather.

<!-- SCREENSHOT: Save the Overview page with its metrics and charts as `docs/screenshots/01-overview.png`. -->
<!-- ![Overview dashboard](docs/screenshots/01-overview.png) -->

### 2. ML Predictions — estimate temperature from entered conditions

Enter weather and calendar variables to estimate temperature with the saved Random Forest pipeline. The page labels this as an estimate from contemporaneous conditions, not a forecast. It then applies the selected demonstration threshold to show a risk category and precautions.

<!-- SCREENSHOT: Save the completed ML Predictions page showing inputs and a result as `docs/screenshots/02-ml-predictions.png`. -->
<!-- ![ML temperature prediction](docs/screenshots/02-ml-predictions.png) -->

### 3. Climate Analytics — explore patterns

Compare temperatures by season, inspect correlations between numeric variables, and review high-temperature records under the configured threshold.

<!-- SCREENSHOT: Save the Climate Analytics page with the seasonal chart and correlation view as `docs/screenshots/03-climate-analytics.png`. -->
<!-- ![Climate analytics](docs/screenshots/03-climate-analytics.png) -->

### 4. Heatwave Monitoring — review configurable risk counts

See record counts across Normal, Watch, Warning, and Severe risk ranges. The dashboard does not show a geographic map because the dataset does not provide verified location coordinates.

<!-- SCREENSHOT: Save the Heatwave Monitoring page with category counts and the location-data notice as `docs/screenshots/04-heatwave-monitoring.png`. -->
<!-- ![Heat-risk monitoring](docs/screenshots/04-heatwave-monitoring.png) -->

### 5. Early Warning Centre — communicate practical precautions

Generate deterministic advisory text for citizens, farmers, health departments, and local authorities. The page explains the input and limitations, and does not claim to issue an official warning.

<!-- SCREENSHOT: Save the Early Warning Centre page with an audience selected and advisory visible as `docs/screenshots/05-early-warning.png`. -->
<!-- ![Early warning advisory](docs/screenshots/05-early-warning.png) -->

### 6. Model Evaluation — compare measured performance

View model metrics and the evaluation approach. The training process also saves actual-versus-estimated and residual plots in `reports/model_diagnostics.png`.

<!-- SCREENSHOT: Save the Model Evaluation page with the comparison metrics visible as `docs/screenshots/06-model-evaluation.png`. -->
<!-- ![Model evaluation](docs/screenshots/06-model-evaluation.png) -->

## Model results

The models were evaluated on a seeded random 80/20 split of the 1,000 rows. The data does not establish a continuous, location-specific time series, so this split measures temperature estimation from other weather observations; it does **not** measure future forecast skill.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Linear Regression baseline | 0.590 °C | 0.733 °C | 0.982 |
| Random Forest regressor | 0.723 °C | 0.930 °C | 0.972 |

Linear Regression performed better on this holdout. The application’s saved prediction pipeline is Random Forest, the primary model requested for the project. Scores describe this dataset and split; they should not be interpreted as evidence of operational forecasting performance.

## How it works

1. **Load and validate** the CSV schema.
2. **Clean** duplicate rows, malformed numeric values, out-of-range measurements, inconsistent category casing, and invalid calendar dates.
3. **Engineer calendar features**, including day of year and a season where the supplied season is missing.
4. **Fit and compare** Linear Regression and Random Forest in preprocessing pipelines with median/mode imputation, scaling, and one-hot encoding.
5. **Evaluate** with MAE, RMSE, and R²; save the Random Forest pipeline, metrics, EDA summary, and diagnostic charts.
6. **Explore** historical data and run estimates in the Streamlit dashboard.

`Climate_Condition` is not used as a model input because it may encode temperature-related information. It is also not treated as an official heatwave label. No supervised heatwave classifier is trained.

## Dataset

The project expects `data/climate_probabilistic_latent_variable_dataset (1).csv` with these columns:

`Year`, `Month`, `Day`, `Season`, `Temperature_C`, `Humidity_pct`, `Pressure_hPa`, `Wind_Speed_kmph`, `Cloud_Cover_pct`, `Rainfall_mm`, `Visibility_km`, `Solar_Radiation_Wm2`, `Air_Quality_Index`, `Dew_Point_C`, `Climate_Condition`.

The app does not assume that the dataset contains coordinates, station identity, continuous daily observations, verified official measurements, or official heatwave labels. It therefore does not invent hotspots, live weather, official warnings, or multi-day forecasts.

## Run locally on macOS

```bash
git clone <your-repository-url>
cd <repository-folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the CSV in `data/`, then train and launch:

```bash
python -m src.train_temperature_model
streamlit run app.py
```

If the CSV is stored elsewhere, set its path first:

```bash
export CLIMATE_DATA_PATH="/path/to/climate_probabilistic_latent_variable_dataset (1).csv"
python -m src.train_temperature_model
streamlit run app.py
```

Run the tests with:

```bash
pytest -q
```

## Repository map

```text
app.py                         Streamlit dashboard
src/data_loader.py             CSV loading and schema checks
src/preprocessing.py           Cleaning and calendar feature engineering
src/train_temperature_model.py Model training, comparison, and reports
src/prediction.py              Saved-pipeline inference
src/risk_engine.py             Configurable demonstration risk categories
src/advisories.py              Deterministic precautions and messages
data/                          Source dataset
models/                        Trained model pipeline
reports/                       Metrics, dataset summary, and diagnostic plots
tests/                         Basic preprocessing, prediction, and risk tests
```

## Screenshot files to add

Put the screenshots in `docs/screenshots/` using these names. Each one belongs in the matching dashboard section above; the Markdown image lines are ready but commented out until the image files exist.

| File name | Capture this page |
|---|---|
| `01-overview.png` | Overview metrics and charts |
| `02-ml-predictions.png` | Prediction form and a generated estimate |
| `03-climate-analytics.png` | Seasonal comparison and correlations |
| `04-heatwave-monitoring.png` | Risk counts and location-data notice |
| `05-early-warning.png` | Advisory with an audience selected |
| `06-model-evaluation.png` | Model comparison metrics |

After adding each image, remove the surrounding `<!--` and `-->` from its matching image line above. GitHub will then render it in place.

## Limitations and next steps

The risk thresholds are configurable educational settings, not IMD alert criteria. Use verified, applicable official guidance and reliable local observations before considering operational use. A true forecasting extension would need quality-controlled location-specific timestamps, past-only lag and rolling features, a defined forecast horizon, chronological train/test splits, and horizon-specific evaluation.

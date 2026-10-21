from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "climate_probabilistic_latent_variable_dataset (1).csv"
EXPECTED = ["Year", "Month", "Day", "Season", "Temperature_C", "Humidity_pct", "Pressure_hPa", "Wind_Speed_kmph", "Cloud_Cover_pct", "Rainfall_mm", "Visibility_km", "Solar_Radiation_Wm2", "Air_Quality_Index", "Dew_Point_C", "Climate_Condition"]

def load_dataset(path=DEFAULT_DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Put the provided CSV in the project's data/ folder or set CLIMATE_DATA_PATH.")
    df = pd.read_csv(path)
    missing = sorted(set(EXPECTED) - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    return df

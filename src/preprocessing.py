import numpy as np
import pandas as pd

NUMERIC_RANGES = {"Month": (1, 12), "Day": (1, 31), "Humidity_pct": (0, 100), "Cloud_Cover_pct": (0, 100), "Wind_Speed_kmph": (0, 300), "Rainfall_mm": (0, 2000), "Visibility_km": (0, 100), "Solar_Radiation_Wm2": (0, 2000), "Air_Quality_Index": (0, 1000), "Pressure_hPa": (800, 1200)}

def clean_data(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.drop_duplicates().reset_index(drop=True)
    for c in out.columns:
        if c not in ("Season", "Climate_Condition"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col in out:
            out.loc[~out[col].between(lo, hi), col] = np.nan
    for col in ("Season", "Climate_Condition"):
        if col in out:
            out[col] = out[col].astype("string").str.strip().str.title().replace({"": pd.NA, "Nan": pd.NA, "<NA>": pd.NA})
    if {"Year", "Month", "Day"}.issubset(out.columns):
        dates = pd.to_datetime(dict(year=out.Year, month=out.Month, day=out.Day), errors="coerce")
        out["Day_Of_Year"] = dates.dt.dayofyear
        out.loc[dates.isna(), ["Year", "Month", "Day", "Day_Of_Year"]] = np.nan
        season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Pre-Monsoon", 4: "Pre-Monsoon", 5: "Pre-Monsoon", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Post-Monsoon", 10: "Post-Monsoon", 11: "Post-Monsoon"}
        derived = out.Month.map(season_map)
        if "Season" not in out: out["Season"] = derived
        else: out["Season"] = out.Season.fillna(derived)
    return out

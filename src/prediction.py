from pathlib import Path
from joblib import load
import pandas as pd
from src.data_loader import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "temperature_model.joblib"

def predict_temperature(values, model_path=MODEL_PATH):
    if not Path(model_path).exists(): raise FileNotFoundError("Trained model is missing. Run: python -m src.train_temperature_model")
    bundle = load(model_path)
    X = pd.DataFrame([{k: values.get(k) for k in bundle["features"]}])
    return float(bundle["pipeline"].predict(X)[0])

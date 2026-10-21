import pytest
from src.prediction import predict_temperature

def test_prediction_reports_missing_model():
    with pytest.raises(FileNotFoundError):
        predict_temperature({}, model_path="/tmp/absent-climate-model.joblib")

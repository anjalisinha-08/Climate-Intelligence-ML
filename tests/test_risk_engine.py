from src.risk_engine import assess_risk

def test_risk_categories():
    assert assess_risk(30)["category"] == "Normal"
    assert assess_risk(36)["category"] == "Watch"
    assert assess_risk(39)["category"] == "Warning"
    assert assess_risk(40)["category"] == "Severe risk"
    assert assess_risk(None)["category"] == "Insufficient data"

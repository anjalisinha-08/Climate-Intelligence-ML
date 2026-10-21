PRECAUTIONS = {"Normal": ["Continue routine heat safety practices."], "Watch": ["Drink water regularly; reduce strenuous outdoor activity during the hottest hours."], "Warning": ["Limit outdoor exertion, use shade, and check on people at higher risk."], "Severe risk": ["Avoid strenuous outdoor activity, stay cool, hydrate, and seek medical care for heat illness symptoms."], "Insufficient data": ["No advisory can be generated without a usable temperature observation."]}

def make_advisory(risk, audience="Citizens", temperature=None, timestamp=None):
    category = risk["category"]
    return {"category": category, "audience": audience, "condition": f"Observed or estimated temperature: {temperature:.1f}°C" if temperature is not None else "Temperature unavailable", "precautions": PRECAUTIONS.get(category, PRECAUTIONS["Insufficient data"]), "timestamp": timestamp or "Not available", "forecast_horizon": "None; this is not a forecast", "limitations": risk["explanation"]}

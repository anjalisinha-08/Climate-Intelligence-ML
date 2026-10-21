def assess_risk(temperature_c, threshold_c=40, watch_margin_c=5, warning_margin_c=2):
    """Configurable educational categories, not an official alert."""
    if temperature_c is None:
        return {"category": "Insufficient data", "explanation": "A temperature value is required."}
    t = float(temperature_c)
    if t >= threshold_c: category, explanation = "Severe risk", f"Temperature is at or above configured threshold ({threshold_c:.1f}°C)."
    elif t >= threshold_c - warning_margin_c: category, explanation = "Warning", "Temperature is close to the configured threshold."
    elif t >= threshold_c - watch_margin_c: category, explanation = "Watch", "Temperature is elevated relative to the configured threshold."
    else: category, explanation = "Normal", "Temperature is below the configured watch range."
    return {"category": category, "explanation": explanation + " Educational categories only; not official IMD criteria."}

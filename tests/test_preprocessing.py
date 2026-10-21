import pandas as pd
from src.preprocessing import clean_data

def test_clean_data_deduplicates_and_derives_calendar_features():
    frame = pd.DataFrame({"Year":[2024,2024],"Month":[1,13],"Day":[2,2],"Season":[None,"summer"],"Temperature_C":[22,25],"Climate_Condition":[" mild ","Hot"]})
    cleaned = clean_data(frame)
    assert len(cleaned) == 2
    assert cleaned.loc[0,"Day_Of_Year"] == 2
    assert pd.isna(cleaned.loc[1,"Month"])
    assert cleaned.loc[0,"Climate_Condition"] == "Mild"

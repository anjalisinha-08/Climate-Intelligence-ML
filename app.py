import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data_loader import load_dataset, DEFAULT_DATA_PATH
from src.preprocessing import clean_data
from src.prediction import predict_temperature
from src.risk_engine import assess_risk
from src.advisories import make_advisory

st.set_page_config(page_title="Climate Intelligence", page_icon="🌡️", layout="wide")
st.markdown("""
<style>
.stApp { background:#f4f7fb; color:#142235; }
.block-container { padding-top:1.4rem; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stRadio label, .stCheckbox label { color:#24364b !important; font-weight:600; }
[data-testid="stAlert"] { background:#fff3cd !important; border:1px solid #d9a441 !important; }
[data-testid="stAlert"] p, [data-testid="stAlert"] div,
[data-testid="stAlert"] span { color:#4b3510 !important; }
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background:#fff !important; color:#142235 !important; border-color:#aab6c5 !important;
}
[data-testid="stNumberInput"] button { color:#24364b !important; }
[data-testid="stFormSubmitButton"] button { background:#17324d !important; color:#fff !important; border:0; }
[data-testid="stFormSubmitButton"] button:hover { background:#244b6b !important; color:#fff !important; }
</style>
""", unsafe_allow_html=True)
st.title("🌡️ Climate Intelligence")
st.caption("Historical analysis and educational estimates · No live weather or official warnings")
with st.sidebar:
    st.header("Climate Monitor")
    page = st.radio("Navigate", ["Overview", "ML Predictions", "Climate Analytics", "Heatwave Monitoring", "Early Warning Centre", "Model Evaluation"])
    data_path = os.environ.get("CLIMATE_DATA_PATH", str(DEFAULT_DATA_PATH))
    threshold = st.number_input("Demonstration high-temperature threshold (°C)", 30.0, 55.0, 40.0, 0.5)
try:
    df = clean_data(load_dataset(data_path))
except Exception as e:
    st.error(str(e)); st.info("Place the provided CSV in data/ or set CLIMATE_DATA_PATH. No sample data is substituted."); st.stop()
temps = df.Temperature_C.dropna()
high = df[df.Temperature_C >= threshold]
if page == "Overview":
    st.subheader("Dataset overview")
    a,b,c,d=st.columns(4); a.metric("Records", f"{len(df):,}"); b.metric("Mean observed", f"{temps.mean():.1f} °C" if len(temps) else "—"); c.metric("Maximum", f"{temps.max():.1f} °C" if len(temps) else "—"); d.metric("Minimum", f"{temps.min():.1f} °C" if len(temps) else "—")
    st.caption(f"{len(high)} records ({len(high)/max(len(temps),1):.1%} of records with temperature) meet the configured threshold. Dataset statistics are historical, not live.")
    left,right=st.columns(2); left.plotly_chart(px.histogram(df, x="Temperature_C", nbins=30, title="Temperature distribution", color_discrete_sequence=["#e76f51"]), use_container_width=True)
    right.plotly_chart(px.box(df, x="Month", y="Temperature_C", title="Observed temperatures by month", color_discrete_sequence=["#e76f51"]), use_container_width=True)
    st.subheader("Latest calendar dates present in the file"); st.dataframe(df.sort_values(["Year","Month","Day"], na_position="first").tail(10), use_container_width=True)
elif page == "ML Predictions":
    st.subheader("Temperature estimate from entered conditions")
    st.warning("This estimates temperature from contemporaneous inputs. It is not a future forecast.")
    with st.form("prediction"):
        cols=st.columns(3); vals={}
        fields=[("Humidity_pct","Humidity (%)",50,0,100), ("Dew_Point_C","Dew point (°C)",20,-20,40), ("Wind_Speed_kmph","Wind speed (km/h)",10,0,200), ("Pressure_hPa","Pressure (hPa)",1013,850,1100), ("Solar_Radiation_Wm2","Solar radiation (W/m²)",400,0,1400), ("Cloud_Cover_pct","Cloud cover (%)",30,0,100), ("Rainfall_mm","Rainfall (mm)",0,0,300), ("Month","Month",6,1,12), ("Day","Day",15,1,31), ("Year","Year",2024,1900,2100), ("Day_Of_Year","Day of year",166,1,366), ("Air_Quality_Index","Air quality index",50,0,500)]
        for i,(key,label,default,lo,hi) in enumerate(fields): vals[key]=cols[i%3].number_input(label, min_value=lo, max_value=hi, value=default)
        vals["Season"]=cols[0].selectbox("Season",["Winter","Pre-Monsoon","Monsoon","Post-Monsoon"]); vals["Climate_Condition"]=cols[1].selectbox("Climate condition", sorted(df.Climate_Condition.dropna().unique().tolist()) or ["Unknown"])
        go=st.form_submit_button("Estimate temperature")
    if go:
        try:
            t=predict_temperature(vals); risk=assess_risk(t,threshold); st.metric("Estimated temperature",f"{t:.1f} °C"); st.write(f"**Configured risk:** {risk['category']}"); st.write(risk["explanation"]); st.write("Precautions:", "; ".join(make_advisory(risk,temperature=t)["precautions"]))
        except Exception as e: st.error(str(e))
elif page == "Climate Analytics":
    st.subheader("Climate analytics")
    st.plotly_chart(px.box(df,x="Season",y="Temperature_C",title="Seasonal temperature comparison",color="Season"),use_container_width=True)
    numeric=df.select_dtypes(include="number"); st.plotly_chart(px.imshow(numeric.corr(),text_auto=".2f",color_continuous_scale="RdBu_r",title="Weather variable correlations"),use_container_width=True)
    st.write("Extreme observations under configured threshold"); st.dataframe(high.sort_values("Temperature_C",ascending=False).head(20),use_container_width=True)
elif page == "Heatwave Monitoring":
    st.subheader("Configured heat-risk monitoring")
    st.info("No coordinates or verified regional identifiers are assumed. Geographic hotspots cannot be mapped from this dataset.")
    groups=pd.cut(df.Temperature_C,[-float("inf"),threshold-5,threshold-2,threshold,float("inf")],labels=["Normal","Watch","Warning","Severe risk"])
    st.plotly_chart(px.bar(groups.value_counts().reindex(["Normal","Watch","Warning","Severe risk"]).rename_axis("Category").reset_index(name="Records"),x="Category",y="Records",color="Category",title="Records by configured temperature category"),use_container_width=True)
    st.dataframe(high.sort_values("Temperature_C",ascending=False),use_container_width=True)
elif page == "Early Warning Centre":
    st.subheader("Rule-based advisory templates")
    audience=st.selectbox("Audience",["Citizens","Farmers","Health departments","Local authorities"])
    if len(temps):
        observation=float(temps.max()); risk=assess_risk(observation,threshold); advisory=make_advisory(risk,audience,observation)
        st.write(f"**{advisory['category']} · {audience}**"); st.write(advisory["condition"]); st.write("Recommendations:"); [st.write("• "+x) for x in advisory["precautions"]]; st.caption(f"Timestamp: {advisory['timestamp']} · Forecast horizon: {advisory['forecast_horizon']}"); st.caption(advisory["limitations"])
    else: st.warning("Insufficient temperature observations to create an advisory.")
    st.info("Deterministic educational template. Operational warnings require official criteria, regional context, and verified observations.")
elif page == "Model Evaluation":
    st.subheader("Temperature model evaluation")
    p=Path(__file__).parent/"reports"/"model_metrics.json"
    if p.exists(): st.json(pd.read_json(p).to_dict())
    else: st.info("Metrics unavailable. Add the real dataset and run `python -m src.train_temperature_model`.")
    st.write("Models: Linear Regression baseline and Random Forest primary regressor. Evaluation uses a seeded random 80/20 split because continuous, location-specific chronology is not established. Inputs include contemporaneous weather values; results do not demonstrate future forecast skill.")

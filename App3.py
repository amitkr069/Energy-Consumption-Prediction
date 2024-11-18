import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import math
import numpy as np
import plotly.express as px
import  seaborn as sns
# import preprocess, helper, helper2
# import predict_odi, score_predict
import plotly.figure_factory as ff
from pathlib import Path
import datetime

st.markdown("<h1 style='text-align: center;'>Energy Consumption Prediction</h1>", unsafe_allow_html=True)

# model = joblib.load('xgboost_model_fold_3.joblib')
model = load_model('lstm_model_fold_5.keras', custom_objects={'mse': MeanSquaredError()})

col1, col2 = st.columns(2)
with col1:
    global_active_power = st.slider("Global Active Power (kW)", min_value=0.10, max_value=7.0, step=0.001)
with col2:
    global_reactive_power = st.number_input("Global Reactive Power (kW)", min_value=0.0)

col1, col2 = st.columns(2)
with col1:
    voltage = st.number_input("Voltage (V)", min_value=225.0, max_value=253.0)
with col2:
    global_intensity = st.number_input("Global Intensity (A)", min_value=0.5, max_value=29.0)

col1, col2, col3 = st.columns(3)
with col1:
    hour = st.selectbox("Hour (24-hour format)", list(range(0, 24)))
with col2:
    day_of_week = st.selectbox("Day of the Week", list(range(1, 32)))
with col3:
    day_of_month = st.selectbox("Day of Month", list(range(1, 32)))

col1, col2 = st.columns(2)
with col1:
    day_of_year = st.slider("Day of Year", min_value=1, max_value=365)
with col2:
    week_of_year = st.slider("Week of Year", min_value=1, max_value=52)

col1, col2, col3 = st.columns(3)
with col1:
    month = st.selectbox("Month", list(range(1, 13)))
with col2:
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])
with col3:
    year = st.number_input("Year", min_value=1900, max_value=2100, step=1)

temperature = st.slider("Temperature (°C)", min_value=-14.0, max_value=35.0, step=0.01)

dew_point = st.slider("Dew Point (°C)", min_value=9.0, max_value=15.0, step=0.01)

relative_humidity = st.slider("Relative Humidity (%)", min_value=0.0, max_value=100.0, step=0.01)

col1, col2 = st.columns(2)
with col1:
    precipitation = st.number_input("Precipitation (mm)", min_value=0.0)
with col2:
    wind_direction = st.number_input("Wind Direction", min_value=0.0)
col1, col2, col3 = st.columns(3)
with col1:
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0)
with col2:
    pressure = st.number_input("Pressure (hPa)", min_value=0.0)
with col3:
    sum_sub_meterings = st.number_input("Sum of Sub Meterings", min_value=0.0)

global_active_power_rolling_24 = st.slider("Global Active Power Rolling 24 (kW)", min_value=0.0, max_value=10.0, step=0.001)
global_active_power_rolling_48 = st.slider("Global Active Power Rolling 48 (kW)", min_value=0.0, max_value=10.0, step=0.001)
global_active_power_rolling_72 = st.slider("Global Active Power Rolling 72 (kW)", min_value=0.0, max_value=10.0, step=0.001)
global_active_power_rolling_96 = st.slider("Global Active Power Rolling 96 (kW)", min_value=0.0, max_value=10.0, step=0.001)
global_active_power_rolling_120 = st.slider("Global Active Power Rolling 120 (kW)", min_value=0.0, max_value=10.0, step=0.001)


if st.button('Predict Energy Consumption'):

    # dictdir = {}
    new_data = pd.DataFrame({
        'Global_active_power': [global_active_power],
        'Global_reactive_power': [global_reactive_power],
        'Voltage': [voltage],
        'Global_intensity': [global_intensity],
        'Hour': [hour],
        'Day_ofweek': [day_of_week],
        'Day_ofmonth': [day_of_month],
        'Day_ofyear': [day_of_year],
        'Week_ofyear': [week_of_year],
        'Month': [month],
        'Quarter': [quarter],
        'Year': [year],
        'temp': [temperature],
        'dwpt': [dew_point],
        'rhum': [relative_humidity],
        'prcp': [precipitation],
        'wdir': [wind_direction],
        'wspd': [wind_speed],
        'pres': [pressure],
        'Sum_Sub_Meterings': [sum_sub_meterings],
        'Global_active_power_rolling_24': [global_active_power_rolling_24],
        'Global_active_power_rolling_48': [global_active_power_rolling_48],
        'Global_active_power_rolling_72': [global_active_power_rolling_72],
        'Global_active_power_rolling_96': [global_active_power_rolling_96],
        'Global_active_power_rolling_120': [global_active_power_rolling_120]

    })


    #xgb
    # expected_columns = model.get_booster().feature_names
    # new_data = new_data.reindex(columns=expected_columns, fill_value=0)

    input_array = new_data.values
    result = model.predict(input_array)

    # result = model.predict(new_data)
    st.header('Energy Consumption: ' + str(math.pow(10,result[0][0])))
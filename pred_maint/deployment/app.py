import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

repo_id = "mrhea/pred-maint"

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id=repo_id, filename="best_pred_maint_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Predictive Maintainence App")
st.write("The Predictive Maintenance Application can analyze historical and real-time engine sensor data to identify potential failures.")
st.write("Kindly enter the vehicle telemetry data to assess the engine health parameters such as RPM, temperature, pressure, and other sensor readings")

# Collect user input
Engine_RPM=st.number_input("Engine_RPM: The number of revolutions per minute (RPM) of the engine, indicating engine speed.", min_value=0, max_value=10000, value=791.24)
Lub_Oil_Pressure=st.number_input("Lub_Oil_Pressure: The pressure of the lubricating oil in the engine, essential for reducing friction and wear. It is defined in bar or kilopascals (kPa).", min_value=0, max_value=100, value=3.1)
Fuel_Pressure=st.number_input("Fuel_Pressure: The pressure at which fuel is supplied to the engine, critical for proper combustion. It is defined in bar or kilopascals (kPa).", min_value=0, max_value=100, value=6.2)
Coolant_Pressure=st.number_input("Coolant_Pressure: The pressure of the engine coolant, affecting engine temperature regulation. It is defined in bar or kilopascals (kPa).", min_value=0, max_value=100, value=2.167)
Lub_Oil_Temperature=st.number_input("Lub_Oil_Temperature: The temperature of the lubricating oil, which impacts viscosity and engine performance. It is defined in degrees Celsius (°C).", min_value=-20, max_value=1000, value=76.81)
Coolant_Temperature=st.number_input("Coolant_Temperature: The temperature of the engine coolant, crucial for preventing overheating. It is defined in degrees Celsius (°C) .", min_value=-20, max_value=1000, value=78.346)

Specific_Lubrication_Index=Lub_Oil_Pressure/Engine_RPM
Volumetric_Coolant_Flow_Proxy=Coolant_Pressure/Engine_RPM
Thermal_Crossover_Delta=Lub_Oil_Temperature-Coolant_Temperature
Safety_Metric=Coolant_Pressure/Coolant_Temperature
Oil-to-Coolant_Pressure_Differential = Lub_Oil_Pressure-Coolant_Pressure
# Create a DataFrame with user input


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Engine rpm': Engine_RPM,
    'Lub oil pressure': Lub_Oil_Pressure,
    'Fuel pressure': Fuel_Pressure,
    'Coolant pressure': Coolant_Pressure,
    'lub oil temp': Lub_Oil_Temperature,
    'Coolant temp': Coolant_Temperature,
    #Calculated Metrics
    'Specific Lubrication Index': Specific_Lubrication_Index,
    'Volumetric Coolant Flow Proxy': Volumetric_Coolant_Flow_Proxy,
    'Thermal Crossover Delta': Thermal_Crossover_Delta,
    'Safety Metric': Safety_Metric
    'Oil-to-Coolant Pressure Differential': Oil-to-Coolant_Pressure_Differential
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "IS likely to require maintenance" if prediction == 1 else "IS NOT likely to require maintenance"
    st.write(f"Based on the information provided, the vehicle engine {result}.")

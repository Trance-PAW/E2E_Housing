import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# Cargar los modelos
pipeline = joblib.load('pipeline.sav')
model = joblib.load('modelLR.sav')

# Título de la aplicación
st.title("Predicción del Valor de Casas en California")

# Entrada de datos del usuario
longitude = st.number_input("Longitud", value=-118.0)
latitude = st.number_input("Latitud", value=34.0)
housing_median_age = st.number_input("Edad Mediana de la Vivienda", value=25)
total_rooms = st.number_input("Total de Habitaciones", value=3000)
total_bedrooms = st.number_input("Total de Dormitorios", value=500)
population = st.number_input("Población", value=1000)
households = st.number_input("Número de Hogares", value=300)
median_income = st.number_input("Ingreso Mediano", value=3.5)
ocean_proximity = st.selectbox("Cercanía al Océano", ("<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"))

# Preparar los datos para la predicción
input_data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})

# Aplicar las transformaciones necesarias (pipeline)
input_data_prepared = pipeline.transform(input_data)

# Hacer la predicción
if st.button("Predecir"):
    prediction = model.predict(input_data_prepared)
    st.success(f"El valor estimado de la casa es: ${prediction[0]:,.2f}")
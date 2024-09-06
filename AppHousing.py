import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

# Cargar el modelo y el pipeline
try:
    full_pipeline = joblib.load('pipeline.sav')
    st.write("Pipeline cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el pipeline: {e}")

try:
    model = joblib.load('modelRL.sav')
    st.write("Modelo cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# Cargar los datos del conjunto de datos de viviendas
@st.cache
def load_data():
    csv_path = "data/housing.csv"
    return pd.read_csv(csv_path)

housing = load_data()

# Título de la aplicación
st.title("Predicción del Valor de Casas en California")

# Mostrar el conjunto de datos
if st.checkbox("Mostrar datos crudos"):
    st.subheader("Datos crudos")
    st.write(housing.head())

# Sección de predicciones
st.subheader("Hacer una predicción")
longitude = st.number_input("Longitud", value=-118.0)
latitude = st.number_input("Latitud", value=34.0)
housing_median_age = st.number_input("Edad Mediana de la Vivienda", value=25)
total_rooms = st.number_input("Total de Habitaciones", value=3000)
total_bedrooms = st.number_input("Total de Dormitorios", value=500)
population = st.number_input("Población", value=1000)
households = st.number_input("Número de Hogares", value=300)
median_income = st.number_input("Ingreso Mediano", value=3.5)
ocean_proximity = st.selectbox("Cercanía al Océano", ("<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"))

# Predicción del modelo
if st.button("Predecir"):
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

    # Mostrar columnas de input_data
    st.write("Columnas de input_data:", input_data.columns)

    # Verificar si el pipeline y el modelo se cargaron correctamente antes de hacer la predicción
    if 'full_pipeline' in locals() and 'model' in locals():
        try:
            # Aplicar las transformaciones necesarias
            input_data_prepared = full_pipeline.transform(input_data)
            st.write("Datos transformados correctamente")

            # Hacer la predicción
            prediction = model.predict(input_data_prepared)
            st.success(f"El valor estimado de la casa es: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
    else:
        st.error("El pipeline o el modelo no están disponibles para hacer predicciones.")

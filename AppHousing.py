# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:11:07 2024

@author: jesus
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request

# Descargar los archivos del modelo desde GitHub
@st.cache
def cargar_modelo():
    url_model = "https://raw.githubusercontent.com/Trance-PAW/E2E_Housing/main/modelLR.sav"
    url_pipeline = "https://raw.githubusercontent.com/Trance-PAW/E2E_Housing/main/pipeline.sav"
    
    # Descargar archivos
    urllib.request.urlretrieve(url_model, 'modelLR.sav')
    urllib.request.urlretrieve(url_pipeline, 'pipeline.sav')
    
    # Cargar modelo y pipeline
    model = joblib.load('modelLR.sav')
    pipeline = joblib.load('pipeline.sav')
    return model, pipeline

def main():
    st.title("Predicción del Valor de Casas en California")

    # Recibir las entradas del usuario
    longitude = st.number_input("Longitud", value=-118.0, min_value=-124.35, max_value=-114.31)
    latitude = st.number_input("Latitud", value=34.0, min_value=32.54, max_value=41.95)
    housing_median_age = st.number_input("Edad Mediana de la Vivienda", value=25)
    total_rooms = st.number_input("Total de Habitaciones", value=3000)
    total_bedrooms = st.number_input("Total de Dormitorios", value=500)
    population = st.number_input("Población", value=1000)
    households = st.number_input("Número de Hogares", value=300)
    median_income = st.number_input("Ingreso Mediano", value=3.5)
    ocean_proximity = st.selectbox("Cercanía al Océano", ("<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"))

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
        
        # Aplicar el pipeline y hacer la predicción
        model, pipeline = cargar_modelo()
        input_data_prepared = pipeline.transform(input_data)
        prediction = model.predict(input_data_prepared)

        # Mostrar el resultado
        st.success(f"El valor estimado de la casa es: ${prediction[0]:,.2f}")

if __name__ == '__main__':
    main()

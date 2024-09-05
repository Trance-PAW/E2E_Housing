import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import load

# Cargar el mejor modelo entrenado
model = load("best_model.joblib")

# Cargar los datos del conjunto de datos de viviendas
@st.cache
def load_data():
    csv_path = "datasets/housing/housing.csv"
    return pd.read_csv(csv_path)

housing = load_data()

# Título de la aplicación
st.title("Predicción del Valor de Casas en California")

# Mostrar el conjunto de datos
if st.checkbox("Mostrar datos crudos"):
    st.subheader("Datos crudos")
    st.write(housing.head())

# Visualización de histogramas
st.subheader("Visualización de los datos")
if st.checkbox("Mostrar histogramas"):
    st.write("Histograma de las variables numéricas")
    fig, ax = plt.subplots(figsize=(20, 15))
    housing.hist(bins=50, figsize=(20, 15), ax=ax)
    st.pyplot(fig)

# Mostrar la correlación de las variables
if st.checkbox("Mostrar matriz de correlación"):
    st.write("Matriz de correlación de las variables numéricas")
    housing_numeric = housing.select_dtypes(include=[np.number])  # Seleccionamos solo las columnas numéricas
    corr_matrix = housing_numeric.corr()
    st.write(corr_matrix["median_house_value"].sort_values(ascending=False))

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

    # Aplicar las transformaciones necesarias (si las tienes en el pipeline del modelo)
    input_data_prepared = full_pipeline.transform(input_data)

    # Hacer la predicción
    prediction = model.predict(input_data_prepared)
    st.success(f"El valor estimado de la casa es: ${prediction[0]:,.2f}")

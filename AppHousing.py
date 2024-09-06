import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin




# Definir la clase CombinedAttributesAdder
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
# Cargar los modelos
pipeline = joblib.load('pipeline.sav')
model = joblib.load('modelLR.sav')

# Título de la aplicación
st.title("Predicción del Valor de Casas en California")

# Entrada de datos del usuario
longitude = st.number_input("Longitud", value=-118.0, min_value=-124.35, max_value=-114.31)
latitude = st.number_input("Latitud", value=34.0, min_value=32.54, max_value=41.95)
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
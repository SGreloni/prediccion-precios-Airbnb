Network_URL ="http: // 192.168.0.13: 8501"
Local_URL = "http: // localhost: 8501"

import streamlit as st
import pandas as pd
import geopy
import xgboost
import pickle
from PIL import Image

import numpy as np

np.bool = np.bool_ #to prevent an error from occurring
np.int = np.int_
np.float = np.float_
np.object = np.object_
np.string = np.string_


modelo = pickle.load(open("price_model.pickle.dat", "rb"))

st.title("Predictor de precios en Airbnb")

img = Image.open("imagen_depto.jpg")
st.image(img, width = 600)

st.sidebar.header("Acerca de")
st.sidebar.markdown("El modelo se trata de un regresor XGBoost, entrenado por [Santiago Greloni](https://www.linkedin.com/in/santiago-greloni-4892a9196/)")
st.sidebar.markdown("En [este repositorio](https://github.com/SGreloni/prediccion-precios-Airbnb) se encuentra el \
                    código con el que fue entrenado y un análisis exploratorio de los datos (en un Jupyter Notebook).\
                    Ahí mismo también se puede encontrar el código con el que se realizó esta misma página")


st.markdown("#### Esta aplicación web devuelve el precio estimativo por noche de la estadía dadas ciertas características\
            de la propiedad.")


st.markdown("### Comodidades")

pileta = st.checkbox("Pileta")
aire = st.checkbox("Aire acondicionado")
TV = st.checkbox("Televisión")


st.markdown("### Tipo de Propiedad")

tipos_posibles = ["Departamento entero", "Condominio entero", "Casa entera",\
                  "Loft entero", "Habitación privada en departamento",\
                  "Habitación privada en casa", "Habitación compartida"]

tipo_propiedad = st.selectbox("Seleccione el tipo de propiedad.", tipos_posibles)

tp_propiedad = {}
for i in range(len(tipos_posibles)):
    if tipo_propiedad == tipos_posibles[i]:
        tp_propiedad[tipos_posibles[i]] = 1
    else:
        tp_propiedad[tipos_posibles[i]] = 0



st.markdown("### Ambientes")


dormitorios = st.slider("Indique la cantidad de dormitorios", 1,  20)

banos = st.slider("Indique la cantidad de baños", 1, 20)

es_privado = st.checkbox("Los baños son privados")

st.markdown("### Capacidad")

huespedes = st.slider("Indique la cantidad de personas que pueden hospedarse", 1,  20)

camas = st.slider("Indique la cantidad de camas disponibles", 1, 20)


st.markdown("### Ubicación")

if st.checkbox("Prefiero escribir la dirección o esquina más cercana"):
    direccion = st.text_input("Escriba la dirección de la propiedad o la esquina más cercana. (Si aparece un mensaje de \
    error escriba la dirección de otra forma o destilde el checkbox para escribir la latitud y longitud.)")
    locator = geopy.geocoders.Nominatim(user_agent="AirbnbPrecios", timeout=10)
    location = locator.geocode(f"{direccion}, Ciudad Autónoma de Buenos Aires, Argentina")
    latitud = location.latitude
    longitud = location.longitude
else:
    latitud = st.text_input("Escriba la latitud de la propiedad.")
    longitud = st.text_input("Escriba la longitud de la propiedad.")


if st.checkbox("¡Listo!"):

    # Datos que se mostrarán al usuario
    propiedad = \
        {"latitud": [float(latitud)], "longitud": [float(longitud)], "huespedes": [huespedes],
               "dormitorios": [dormitorios], "camas": [camas], "baños": [banos],
               "baños_privados": [int(es_privado)], "pileta": [int(pileta)], "TV": [int(TV)], "Aire acondicionado": [int(aire)],
               "Departamento entero": [tp_propiedad["Departamento entero"]],
               "Condominio entero": [tp_propiedad["Condominio entero"]],
               "Casa entera": [tp_propiedad["Casa entera"]],
               "Loft entero": [tp_propiedad["Loft entero"]],
               "Habitación privada en departamento": [tp_propiedad["Habitación privada en departamento"]],
               "Habitación privada en casa": [tp_propiedad["Habitación privada en casa"]],
               "Habitación compartida": [tp_propiedad["Habitación compartida"]]}

    propiedad_df = pd.DataFrame(propiedad)
    df_transpuesta = propiedad_df.transpose(copy=True)

    #Input del modelo (nótese que los nombres de las columnas son distintos
    col_modelo = ['latitude', 'longitude', 'accommodates', 'bedrooms', 'beds', 'baños', 'es_privado', 'Pileta', 'TV',
                  'Aire acondicionado', 'property_type_Entire apartment', 'property_type_Entire condominium',
                  'property_type_Entire house', 'property_type_Entire loft', 'property_type_Private room in apartment',
                  'property_type_Private room in house', 'property_type_Shared room in apartment']

    propiedad_modelo_df = propiedad_df.copy()
    propiedad_modelo_df.columns = col_modelo


    st.markdown("#### Puede chequear que los datos estén correctos en la siguiente tabla:")
    st.write(df_transpuesta)


    if st.button("GENERAR PREDICCIÓN"):
        prediccion = modelo.predict(propiedad_modelo_df)
        st.markdown("### El precio estimado es de \${} por noche.".format(int(prediccion)))


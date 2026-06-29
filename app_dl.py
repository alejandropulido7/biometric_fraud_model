import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# 1. Configuración de la Página
st.set_page_config(
    page_title="Detector Biométrico - Deep Learning",
    page_icon="🧠",
    layout="centered"
)

# 2. Carga de Artefactos en Caché
@st.cache_resource
def cargar_recursos_dl():
    modelo_path = 'modelo_dl_fraudes.keras'
    scaler_path = 'scaler_dl.joblib'
    columnas_path = 'columnas_dl.joblib'
    
    # Validaciones de existencia
    for path in [modelo_path, scaler_path, columnas_path]:
        if not os.path.exists(path):
            st.error(f"Error crítico: No se encontró el archivo '{path}'.")
            st.stop()
            
    modelo = load_model(modelo_path)
    scaler = joblib.load(scaler_path)
    columnas = joblib.load(columnas_path)
    
    return modelo, scaler, columnas

modelo_dl, scaler, columnas_ordenadas = cargar_recursos_dl()

# 3. Interfaz de Usuario
st.title("🧠 Red Neuronal de Detección de Fraude")
st.write("Análisis de riesgos utilizando Perceptrón Multicapa (TensorFlow/Keras).")

st.markdown("---")

with st.expander("ℹ️ ¿Cómo evalúa el riesgo nuestro modelo de Inteligencia Artificial?"):
    st.markdown("""
    Este sistema de **Deep Learning** analiza el comportamiento biométrico del cliente para prevenir ataques con huellas sintéticas o suplantación de identidad. El modelo no solo mira los números, sino la relación entre estas 4 variables clave:

    * ⏳ **Ventana de tiempo (Horas):** El periodo en el que ocurren las transacciones. Un alto volumen de actividad concentrado en muy pocas horas es un fuerte indicador de un ataque rápido.
    * 👆 **Cantidad total de intentos:** El número de veces que el usuario colocó su huella. Los ataques de "fuerza bruta" suelen registrar una cantidad de intentos anormalmente alta.
    * ✅ **Intentos exitosos:** Autenticaciones que sí fueron validadas. Esto permite a la IA proteger a clientes genuinos; si hay muchos fallos pero también éxitos proporcionales, el sistema deduce que el cliente puede tener huellas desgastadas o afecciones como dermatitis, reduciendo el riesgo.
    * 🏢 **Sucursales distintas visitadas:** Un cliente normal suele operar en una sola sucursal. Múltiples intentos fallidos saltando entre diferentes sucursales físicas disparan una alerta crítica de ataque distribuido.
    """)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    rango_horas = st.number_input("Ventana de tiempo (Horas)", min_value=1, max_value=72, value=2)
    intentos = st.number_input("Cantidad total de intentos", min_value=1, max_value=100, value=3)
    
with col2:
    exitosas = st.number_input("Intentos exitosos", min_value=0, max_value=100, value=1)
    sucursales = st.number_input("Sucursales distintas visitadas", min_value=1, max_value=20, value=1)

# 4. Ejecución del Análisis
if st.button("Evaluar con Inteligencia Artificial", type="primary", use_container_width=True):
    
    # Validación lógica de negocio
    if exitosas > intentos:
        st.error("Error: Los éxitos no pueden ser mayores que los intentos totales.")
    else:
        # Ingeniería de variables en vivo
        tasa_errores = (intentos - exitosas) / intentos
        
        datos_entrada = pd.DataFrame([{
            'rango_horas': rango_horas,
            'cantidad_intentos_transaccion': intentos,
            'cantidad_transacciones_exitosas': exitosas,
            'cantidad_sucursales_distintas': sucursales,
            'tasa_errores': tasa_errores
        }])
        
        # 🌟 BLINDAJE 1: Ordenar las columnas exactamente igual que en el entrenamiento
        datos_entrada = datos_entrada[columnas_ordenadas]
        
        # 🌟 BLINDAJE 2: Escalar los datos (Traducir a idioma de Red Neuronal)
        datos_escalados = scaler.transform(datos_entrada)
        
        # 5. Predicción (TensorFlow devuelve una matriz 2D, ej: [[0.85]])
        probabilidad_bruta = modelo_dl.predict(datos_escalados, verbose=0)[0][0]
        prob_fraude = probabilidad_bruta * 100
        
        # 6. Umbrales y Visualización
        st.divider()
        st.subheader("Dictamen de la Red Neuronal")
        
        if prob_fraude >= 90.0:
            st.error(f"🚨 **RIESGO ALTO**")
            st.write("Patrón detectado como ataque distribuido o suplantación. **Acción: Bloqueo.**")
            st.progress(prob_fraude / 100)
            
        elif prob_fraude >= 50.0:
            st.warning(f"⚠️ **RIESGO MEDIO**")
            st.write("Comportamiento anómalo. **Acción: Solicitar Segundo Factor (MFA).**")
            st.progress(prob_fraude / 100)
            
        else:
            st.success(f"✅ **RIESGO NULO**")
            st.write("Operación biométrica congruente con el historial normal.")
            st.progress(prob_fraude / 100)
            
        st.caption(f"Certeza de fraude (Activación Sigmoide): {prob_fraude:.2f}%")
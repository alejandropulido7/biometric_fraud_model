import streamlit as st
import pandas as pd
import joblib
import os

# 1. Configuración de la Página
st.set_page_config(
    page_title="Detector de Fraude Biométrico",
    page_icon="🛡️",
    layout="centered"
)

# 2. Carga del Modelo en Caché (Manejo de errores incluido)
#@st.cache_resource
def cargar_modelo2():
    modelo_path2 = 'modelo_lightgbm_fraudes.joblib'
    if not os.path.exists(modelo_path2):
        st.error(f"Error crítico: No se encontró el artefacto '{modelo_path2}'.")
        return None
    return joblib.load(modelo_path2)

modelo = cargar_modelo2()

# 3. Encabezado de la Interfaz
st.title("🛡️ Sistema de Alerta Temprana: Fraude Biométrico - LightGBM")
st.write("Ingresa los parámetros de comportamiento del cliente en la ventana de tiempo analizada para calcular su score de riesgo.")

st.markdown("---")

# 4. Formulario de Entrada de Datos (Distribución en columnas)
col1, col2 = st.columns(2)

with col1:
    rango_horas = st.number_input("Ventana de tiempo (Horas)", min_value=1, max_value=72, value=2)
    intentos = st.number_input("Cantidad total de intentos", min_value=1, max_value=100, value=3)
    
with col2:
    exitosas = st.number_input("Intentos exitosos", min_value=0, max_value=100, value=1)
    sucursales = st.number_input("Sucursales distintas visitadas", min_value=1, max_value=20, value=1)

# 5. Ejecución del Análisis
if st.button("Analizar Comportamiento", type="primary", use_container_width=True):
    
    if modelo is None:
        st.stop()
        
    # Validación de integridad de negocio
    if exitosas > intentos:
        st.error("Error lógico: Los intentos exitosos no pueden superar el total de intentos registrados.")
    else:
        # Ingeniería de Características en Tiempo Real (Debe ser idéntica al entrenamiento)
        tasa_errores = (intentos - exitosas) / intentos
        
        datos_entrada = pd.DataFrame([{
            'rango_horas': rango_horas,
            'cantidad_intentos_transaccion': intentos,
            'cantidad_sucursales_distintas': sucursales,
            'cantidad_transacciones_exitosas': exitosas,
            'tasa_errores': tasa_errores
        }])
        
        # 6. Predicción Basada en Probabilidades (predict_proba)
        # predict_proba devuelve [Probabilidad_Clase_0, Probabilidad_Clase_1]
        probabilidades = modelo.predict_proba(datos_entrada)[0]
        prob_fraude = probabilidades[1] * 100  # Extraemos la prob. de fraude (Clase 1) en formato %
        
        # 7. Lógica de Umbrales (Thresholding)
        UMBRAL_MEDIO = 30.0
        UMBRAL_ALTO = 70.0
        
        st.divider()
        st.subheader("Resultado del Análisis de Riesgo:")
        
        # Clasificación visual tipo semáforo
        if prob_fraude >= UMBRAL_ALTO:
            st.error("🚨 **RIESGO ALTO: FRAUDE INMINENTE** 🚨")
            st.write("El comportamiento coincide con un patrón de ataque o suplantación. Acción recomendada: **Bloqueo preventivo**.")
            st.progress(prob_fraude / 100)
            
        elif prob_fraude >= UMBRAL_MEDIO:
            st.warning("⚠️ **RIESGO MEDIO: COMPORTAMIENTO SOSPECHOSO** ⚠️")
            st.write("Existen anomalías en la frecuencia o en la tasa de errores. Acción recomendada: **Solicitar segundo factor (OTP)**.")
            st.progress(prob_fraude / 100)
            
        else:
            st.success("✅ **RIESGO NULO: COMPORTAMIENTO GENUINO**")
            st.write("Los intentos biométricos se encuentran dentro de los parámetros de confianza.")
            st.progress(prob_fraude / 100)
            
        # Transparencia del modelo (Métrica para el analista)
        st.caption(f"Score de Probabilidad (LightGBM): {prob_fraude:.2f}% de certeza de fraude.")
        st.write(f"Probabilidad de fraude calculada: {prob_fraude:.2f}%")
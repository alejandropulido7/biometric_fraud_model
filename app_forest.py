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

# 2. Carga del Modelo (con manejo de errores)
#@st.cache_resource
def cargar_modelo():
    modelo_path = 'modelo_iso_forest.joblib'
    if not os.path.exists(modelo_path):
        st.error(f"Error: No se encontró el archivo '{modelo_path}'. Asegúrate de que esté en la misma carpeta.")
        return None
    return joblib.load(modelo_path)

modelo = cargar_modelo()

# 3. Interfaz de Usuario
st.title("🛡️ Sistema de Detección de Anomalías - Isolation forest")
st.write("Análisis de riesgos basado en comportamiento biométrico.")

st.markdown("---")

# Inputs del usuario
col1, col2 = st.columns(2)

with col1:
    rango_horas = st.number_input("Ventana de tiempo (Horas)", min_value=1, max_value=48, value=2)
    intentos = st.number_input("Total intentos", min_value=1, max_value=100, value=3)

with col2:
    exitosas = st.number_input("Intentos exitosos", min_value=0, max_value=100, value=1)
    sucursales = st.number_input("Sucursales distintas", min_value=1, max_value=20, value=1)

# Botón de ejecución
if st.button("Evaluar Riesgo", type="primary", use_container_width=True):
    if modelo is None:
        st.stop()
        
    # Validación lógica
    if exitosas > intentos:
        st.error("Error: Los éxitos no pueden ser mayores que los intentos totales.")
    else:
        # Ingeniería de variables
        tasa_errores = (intentos - exitosas) / intentos
        
        datos_entrada = pd.DataFrame([{
            'rango_horas': rango_horas,
            'cantidad_intentos_transaccion': intentos,
            'cantidad_transacciones_exitosas': exitosas,
            'cantidad_sucursales_distintas': sucursales,
            'tasa_errores': tasa_errores
        }])
        
        # 🌟 Reordenar las columnas automáticamente para evitar el ValueError
        datos_entrada = datos_entrada[modelo.feature_names_in_]
        
        # 4. Predicción con Isolation Forest
        # decision_function: negativo = anomalía (fraude), positivo = normal
        score = modelo.decision_function(datos_entrada)[0]
        
        # 5. Visualización de los 3 Niveles de Riesgo
        st.divider()
        st.subheader("Resultado del Análisis")
        
        # Umbrales ajustables para Isolation Forest
        if score <= -0.01:
            st.error(f"🚨 **RIESGO ALTO** (Score: {score:.3f})")
            st.write("Patrón altamente anómalo detectado. Acción recomendada: **BLOQUEO INMEDIATO**.")
            st.progress(0.9)
            
        elif score <= 0.01:
            st.warning(f"⚠️ **RIESGO MEDIO** (Score: {score:.3f})")
            st.write("El comportamiento presenta desviaciones sospechosas. Acción recomendada: **SOLICITAR SEGUNDO FACTOR**.")
            st.progress(0.5)
            
        else:
            st.success(f"✅ **RIESGO NULO** (Score: {score:.3f})")
            st.write("Comportamiento dentro de los parámetros normales de operación.")
            st.progress(0.1)
            
        st.caption(f"Score técnico (Isolation Forest): {score:.4f}")

        st.write(f"Tasa errores calculada: {tasa_errores:.2%}")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# 1. Configuración de la Página (Cambiamos a "wide" para aprovechar el espacio)
st.set_page_config(
    page_title="Detector Biométrico IA",
    page_icon="🛡️",
    layout="wide"
)

# 2. Inicializar Variables de Sesión (Para que los botones de casos funcionen)
if 'horas' not in st.session_state:
    st.session_state.horas = 2
if 'intentos' not in st.session_state:
    st.session_state.intentos = 3
if 'exitos' not in st.session_state:
    st.session_state.exitos = 1
if 'sucursales' not in st.session_state:
    st.session_state.sucursales = 1

# Funciones para actualizar el formulario según el caso seleccionado
def cargar_caso(h, i, e, s):
    st.session_state.horas = h
    st.session_state.intentos = i
    st.session_state.exitos = e
    st.session_state.sucursales = s

# 3. Carga de Artefactos de Deep Learning
@st.cache_resource
def cargar_recursos_dl():
    modelo_path = 'modelo_dl_fraudes.keras'
    scaler_path = 'scaler_dl.joblib'
    columnas_path = 'columnas_dl.joblib'
    
    if not os.path.exists(modelo_path):
        return None, None, None
            
    modelo = load_model(modelo_path)
    scaler = joblib.load(scaler_path)
    columnas = joblib.load(columnas_path)
    return modelo, scaler, columnas

modelo_dl, scaler, columnas_ordenadas = cargar_recursos_dl()

# 4. BARRA LATERAL (SIDEBAR) - Navegación
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2910/2910809.png", width=100)
st.sidebar.title("Menú Principal")

# Creamos 3 items de navegación
menu = st.sidebar.radio(
    "Selecciona un módulo:",
    ["🔮 Simulador y Predicción", "🧠 Características del Modelo"]
)

st.sidebar.divider()
st.sidebar.caption("Proyecto - Machine Learning")
st.sidebar.caption("Autor: Harley Alejandro Pulido Cardona")


# ==========================================
# PANTALLA 2: MODELO Y DATASET
# ==========================================
if menu == "🧠 Características del Modelo":
    st.title("📊 Ficha Técnica de la IA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Arquitectura del Modelo")
        st.write("**Algoritmo:** Red Neuronal Profunda / Perceptrón Multicapa (MLP)")
        st.write("**Framework:** TensorFlow / Keras")
        st.write("**Capas Ocultas:** 2 capas (6 y 16 neuronas) con activación ReLU")
        st.write("**Regularización:** Dropout (0.3 y 0.2) para evitar sobreajuste")
        st.write("**Salida:** Función Sigmoide (0 a 100%)")
        
        st.divider()
        st.subheader("Rendimiento (Métricas)")
        st.metric(label="Accuracy (Precisión Global)", value="99.46 %")
        st.metric(label="ROC-AUC", value="99.87 %")
        st.metric(label="Loss (Pérdida)", value="0.0172")

    with col2:
        st.subheader("El Dataset Sintético")
        st.write("El modelo fue entrenado con un volumen de datos robusto generado mediante reglas estadísticas de negocio bancario real.")
        st.write("- **Volumen:** 100,000 registros transaccionales.")
        st.write("- **Variable Objetivo:** `es_fraude` (Clasificación Binaria).")
        st.write("- **Variables Predictoras:**")
        st.markdown("""
        1. `rango_horas`: Ventana de tiempo.
        2. `cantidad_intentos_transaccion`: Total de interacciones.
        3. `cantidad_sucursales_distintas`: Dispersión geográfica.
        4. `tasa_errores`: Relación calculada entre intentos fallidos y totales.
        """)
        st.success("Nota: No se utilizó PII (Información Personal Identificable), garantizando el cumplimiento de normativas de protección de datos.")


# ==========================================
# PANTALLA 1: PREDICCIÓN Y CASOS
# ==========================================
elif menu == "🔮 Simulador y Predicción":
    st.title("🔮 Simulador de Riesgo en Tiempo Real")
    
    if modelo_dl is None:
        st.error("⚠️ No se encontraron los archivos del modelo. Asegúrate de tener los .keras y .joblib en la carpeta.")
        st.stop()

    # Dividimos la pantalla: 70% formulario / 30% Casos de uso
    col_form, col_casos = st.columns([2.5, 1.5])
    
    with col_form:
        
        st.write("El sistema evalúa en tiempo real el comportamiento de autenticación biométrica de clientes en sucursales bancarias para predecir la probabilidad de un ataque de suplantación "
        "(ej. uso de huellas sintéticas). Analizando ventanas de tiempo, intentos fallidos y sucursales visitadas, " \
        "el modelo protege a las entidades financieras reduciendo el riesgo de materialización de fraudes, " \
        "sin penalizar injustamente a clientes legítimos con dificultades de autenticación (como dermatitis).")

        st.subheader("Parámetros del Cliente")

        with st.container(border=True):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                rango_horas = st.number_input("Ventana de tiempo (Horas)", min_value=1, max_value=72, key='horas')
                intentos = st.number_input("Cantidad total de intentos", min_value=1, max_value=100, key='intentos')
            with f_col2:
                exitosas = st.number_input("Intentos exitosos", min_value=0, max_value=100, key='exitos')
                sucursales = st.number_input("Sucursales visitadas", min_value=1, max_value=20, key='sucursales')
            
            analizar_btn = st.button("🚀 Evaluar con Inteligencia Artificial", type="primary", use_container_width=True)


# def cargar_caso(h, i, e, s):
#     st.session_state.horas = h
#     st.session_state.intentos = i
#     st.session_state.exitos = e
#     st.session_state.sucursales = s

    with col_casos:
        st.subheader("🧪 Cargar Casos de Prueba")
        st.caption("Haz clic para autocompletar el formulario:")
        
        # 🌟 SOLUCIÓN: Usamos 'on_click' y 'args' para actualizar el estado antes de redibujar la pantalla
        st.button("🟢 1. Cliente Normal", use_container_width=False, 
                  on_click=cargar_caso, args=(2, 5, 0, 1))
            
        st.button("🔴 2. Riesgo Alto", use_container_width=False, 
                  on_click=cargar_caso, args=(3, 13, 0, 3))
            
        st.button("🟡 3. Riesgo Medio (Dos sucursales)", use_container_width=False, 
                  on_click=cargar_caso, args=(2, 10, 2, 2))
            
        st.button("🟠 4. Riesgo Medio Alto (Varios intentos)", use_container_width=False, 
                  on_click=cargar_caso, args=(7, 14, 5, 3))
            
        st.button("✅ 5. Riesgo Nulo (Posible Dermatitis)", use_container_width=False, 
                  on_click=cargar_caso, args=(1, 10, 0, 1))

    # Ejecución del Análisis
    if analizar_btn:
        if exitosas > intentos:
            st.error("Error: Los éxitos no pueden ser mayores que los intentos totales.")
        else:
            tasa_errores = (intentos - exitosas) / intentos
            
            datos_entrada = pd.DataFrame([{
                'rango_horas': rango_horas,
                'cantidad_intentos_transaccion': intentos,
                'cantidad_transacciones_exitosas': exitosas,
                'cantidad_sucursales_distintas': sucursales,
                'tasa_errores': tasa_errores
            }])
            
            datos_entrada = datos_entrada[columnas_ordenadas]
            datos_escalados = scaler.transform(datos_entrada)
            
            probabilidad_bruta = modelo_dl.predict(datos_escalados, verbose=0)[0][0]
            prob_fraude = probabilidad_bruta * 100
            
            st.divider()
            st.subheader("Dictamen de la Red Neuronal")
            
            if prob_fraude >= 90.0:
                st.error(f"🚨 **RIESGO ALTO**")
                st.write("Patrón detectado como ataque distribuido o suplantación. **Acción: Bloqueo preventivo.**")
                st.progress(prob_fraude / 100)
            elif prob_fraude >= 50.0:
                st.warning(f"⚠️ **RIESGO MEDIO**")
                st.write("Comportamiento anómalo. **Acción: Solicitar Segundo Factor (MFA - OTP).**")
                st.progress(prob_fraude / 100)
            else:
                st.success(f"✅ **RIESGO NULO**")
                st.write("Operación biométrica congruente con el historial normal.")
                st.progress(prob_fraude / 100)
                
            st.caption(f"Certeza de fraude calculada por la IA: {prob_fraude:.2f}%")
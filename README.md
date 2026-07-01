# 🛡️ Sistema de Alerta Temprana de Fraude Biométrico

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-yellow.svg)

Una solución basada en **Deep Learning** diseñada para entidades financieras. Este sistema evalúa en tiempo real el comportamiento de autenticación biométrica de clientes en sucursales bancarias, prediciendo la probabilidad de ataques de suplantación (ej. uso de huellas sintéticas o ataques distribuidos) sin penalizar a clientes genuinos con dificultades biométricas legítimas (como dermatitis).

[🚀 Probar la Aplicación en Vivo](https://biometricfraudmodel-guu6uifsgdsjpqeaaum64n.streamlit.app/)

---

## ✨ Características Principales

* **Red Neuronal Profunda (MLP):** Utiliza un modelo de Perceptrón Multicapa construido con TensorFlow/Keras, superando en el benchmark a algoritmos tradicionales como Isolation Forest, LightGBM y XGBoost.
* **Feature Engineering en Tiempo Real:** Calcula dinámicamente la tasa de errores del usuario y estandariza las métricas instantáneamente antes de la inferencia.
* **Clasificación de Riesgo Inteligente:** Clasifica las operaciones en tres niveles semaforizados:
  * 🟢 **Riesgo Nulo (< 50%):** Comportamiento genuino.
  * 🟡 **Riesgo Medio (50% - 90%):** Alerta preventiva, sugiere solicitud de Segundo Factor (OTP).
  * 🔴 **Riesgo Alto (> 90%):** Alerta crítica, sugiere bloqueo preventivo por ataque inminente.
* **Interfaz Interactiva:** Despliegue ágil y amigable construido con Streamlit.

---

## 🏗️ Arquitectura del Proyecto

El sistema está compuesto por los siguientes archivos clave:

```text
📂 biometric-fraud-detection
│
├── 📓 entrenamiento_modelos.ipynb   # Notebook con generación de datos sintéticos, EDA y Benchmark
├── 🐍 app_dl.py                     # Código fuente de la aplicación en Streamlit
├── 🧠 modelo_dl_fraudes.keras       # Modelo de Deep Learning exportado
├── ⚖️ scaler_dl.joblib              # Escalador de datos (StandardScaler)
├── 🗺️ columnas_dl.joblib            # Mapeo del orden exacto de variables de entrenamiento
├── 📦 requirements.txt              # Dependencias del proyecto
└── 📜 README.md                     # Este archivo

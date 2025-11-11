import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# ==========================
# 1. Cargar Modelos y Artefactos
# ==========================
@st.cache_resource
def load_model():
    try:
        # Cargar modelos y artefactos
        model = joblib.load('models/xgboost_model.pkl')
        categorical_mapping = joblib.load('models/categorical_mapping.pkl')  # Cargar el diccionario de codificación

        # Aseguramos que el scaler es el objeto correcto (MinMaxScaler o similar)
        scaler = joblib.load('models/scaler.pkl')

        REFERENCE_DATA_PATH = 'data/reference_data.csv'
        if not os.path.exists(REFERENCE_DATA_PATH):
            st.error(f"Error: No se encontró la data de referencia en '{REFERENCE_DATA_PATH}'. Necesaria para evaluación de simulaciones.")
            return None, None, None, None, None

        df_reference = pd.read_csv(REFERENCE_DATA_PATH)

        if 'Attrition' not in df_reference.columns:
            st.error("Error: La data de referencia debe contener la columna 'Attrition' para la evaluación.")
            return None, None, None, None, None

        # Solución al error 'invalid literal for int(): 'Yes''
        df_reference['Attrition'] = df_reference['Attrition'].replace({'Yes': 1, 'No': 0})

        true_labels_reference = df_reference['Attrition'].astype(int).copy()
        df_reference_features = df_reference.drop(columns=['Attrition'], errors='ignore').copy()
        
        st.success("✅ Modelo y artefactos cargados correctamente.")
        return model, categorical_mapping, scaler, df_reference_features, true_labels_reference

    except FileNotFoundError:
        st.error("Error: Archivos del modelo (xgboost_model.pkl, categorical_mapping.pkl, scaler.pkl) no encontrados. Asegúrate de tener la carpeta 'models' con los 3 archivos.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error al cargar artefactos o data de referencia: {e}")
        return None, None, None, None, None


# ================================
# 2. Funciones de Preprocesamiento
# ================================
def preprocess_data(df, model_columns, categorical_mapping, scaler):
    df_processed = df.copy()

    # Verificar si faltan columnas y agregarlas
    for col in model_columns:
        if col not in df_processed.columns:
            df_processed[col] = np.nan  # Agregar columnas faltantes con NaN

    # Rellenar nulos y codificar
    df_processed = df_processed.drop_duplicates()
    numeric_cols_for_fillna = df_processed.select_dtypes(include=np.number).columns.tolist()
    
    # Solo rellenamos las columnas numéricas que están en model_columns
    cols_to_fill = list(set(numeric_cols_for_fillna) & set(model_columns))
    df_processed[cols_to_fill] = df_processed[cols_to_fill].fillna(df_processed[cols_to_fill].mean())

    nominal_categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

    for col in nominal_categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).str.strip().str.upper()

            if col in categorical_mapping:
                # Mapeo de valores conocidos
                df_processed[col] = df_processed[col].map(categorical_mapping[col])

            # Rellenar valores desconocidos (si los hay) con el valor de 'DESCONOCIDO' o -1
            df_processed[col] = df_processed[col].fillna(categorical_mapping.get(col, {}).get('DESCONOCIDO', -1))
        
    # --- Ordenar y seleccionar solo las columnas de FEATURES ANTES DE ESCALAR ---
    df_to_scale = df_processed[model_columns].copy()
    
    # Escalado
    try:
        # El scaler espera solo las columnas de las features
        df_processed[model_columns] = scaler.transform(df_to_scale)
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}. ¿El scaler fue entrenado correctamente con la forma de la data?")
        return None

    # Finalmente, devolver el DataFrame solo con las columnas que el modelo espera y en el orden correcto
    return df_processed[model_columns]


# ============================
# Función para la Predicción
# ============================
def predict(model, df_processed):
    try:
        # Hacer predicciones con el modelo
        probabilidad_renuncia = model.predict_proba(df_processed)[:, 1]
        predictions = (probabilidad_renuncia > 0.5).astype(int)  # Aplicar umbral para clasificación binaria
        return probabilidad_renuncia, predictions
    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")
        return None, None


# ============================
#  Interfaz Streamlit
# ============================
def main():
    st.set_page_config(page_title="Predicción y Simulación de Renuncia", layout="wide")
    st.title("📊 Modelo de Predicción y Simulación de Renuncia de Empleados")
    st.markdown("Elige entre cargar un archivo o ingresar datos manualmente para hacer predicciones y análisis.")

    model, categorical_mapping, scaler, df_reference_features, true_labels_reference = load_model()
    if model is None:
        return

    # Columnas que debe tener el dataset cargado
    model_feature_columns = [
        'Edad', 'ViajeDeNegocios', 'TarifaDiaria', 'Departamento', 'DistanciaDeCasa', 'Educacion', 'CampoEducacion', 
        'SatisfaccionAmbiente', 'Genero', 'TarifaHora', 'InvolucramientoTrabajo', 'NivelTrabajo', 'RolTrabajo', 'SatisfaccionTrabajo', 
        'EstadoCivil', 'IngresoMensual', 'TasaMensual', 'NumEmpresasTrabajadas', 'TiempoExtra', 'IncrementoSalarial', 
        'RendimientoTrabajo', 'SatisfaccionRelaciones', 'NivelOpcionesAcciones', 'TotalAñosTrabajados', 'TiempoEntrenamiento', 
        'EquilibrioTrabajoVida', 'AñosEnEmpresa', 'AñosEnRolActual', 'AñosUltimaPromocion', 'AñosConGerenteActual', 
        'IntencionPermanencia', 'CargaLaboralPercibida', 'SatisfaccionSalarial', 'ConfianzaEmpresa', 'NumeroTardanzas', 
        'NumeroFaltas'
    ]

    # 1. Opción para cargar archivo CSV o Excel
    st.header("Hoja 1: Cargar archivo para predicción")
    uploaded_file = st.file_uploader("Sube un archivo CSV o Excel (.csv, .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.info(f"✅ Archivo cargado correctamente. Total de filas: {len(df)}")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return

        # Procesar el archivo cargado
        df_features_uploaded = df.drop(columns=['Attrition'], errors='ignore').copy()
        processed_df = preprocess_data(df_features_uploaded, model_feature_columns, categorical_mapping, scaler)
        
        if processed_df is None:
            st.error("No se puede continuar con la predicción debido a un error de preprocesamiento.")
            return
        
        # Ejecutar predicción con los datos cargados
        if st.button("🚀 Ejecutar Predicción (Cargar archivo)"):
            st.info("Ejecutando el modelo sobre los datos cargados...")
            probabilidad_renuncia, predictions = predict(model, processed_df)

            # Unir resultados a la data original
            df['Prediction_Renuncia'] = predictions
            df['Probabilidad_Renuncia'] = probabilidad_renuncia
            st.success("✅ Predicción de datos cargados Completada!")
            st.dataframe(df[['Prediction_Renuncia', 'Probabilidad_Renuncia']])

    # 2. Hoja para ingresar datos manuales
    st.header("Hoja 2: Ingresar datos manualmente para predicción")
    with st.form("manual_form"):
        manual_data = {}
        for col in model_feature_columns:
            if col == 'Edad':
                manual_data[col] = st.number_input(f"{col}", min_value=18, max_value=100, value=30)
            elif col == 'IngresoMensual':
                manual_data[col] = st.number_input(f"{col}", min_value=0, max_value=100000, value=30000)
            else:
                manual_data[col] = st.text_input(f"{col}")

        submit_button = st.form_submit_button("🚀 Predecir (Ingresar manualmente)")

        if submit_button:
            # Convertir los datos manuales en un DataFrame para preprocesamiento
            df_manual = pd.DataFrame([manual_data])
            processed_manual_data = preprocess_data(df_manual, model_feature_columns, categorical_mapping, scaler)

            if processed_manual_data is not None:
                probabilidad_renuncia, predictions = predict(model, processed_manual_data)
                st.success("✅ Predicción Manual Completa!")
                st.write(f"Probabilidad de renuncia: {probabilidad_renuncia[0]:.2f}")
                st.write(f"Predicción: {'Renunciará' if predictions[0] == 1 else 'No Renunciará'}")
            else:
                st.error("Error en el preprocesamiento de los datos manuales.")


if __name__ == "__main__":
    main()
















import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.preprocessing import MinMaxScaler
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
        # df_reference_features ahora incluye todas las columnas EXCEPTO Attrition y se usará para el merge.
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

    # Validar columnas
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
        
    # Ordenar y seleccionar solo las columnas de FEATURES ANTES DE ESCALAR
    df_to_scale = df_processed[model_columns].copy()
    
    # Escalado
    try:
        df_processed[model_columns] = scaler.transform(df_to_scale)
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}. ¿El scaler fue entrenado correctamente con la forma de la data?")
        return None

    # Finalmente, devolver el DataFrame solo con las columnas que el modelo espera y en el orden correcto
    return df_processed[model_columns]


# ============================
#  Funciones de Evaluación
# ============================
def evaluate_metrics(true_labels, predictions):
    acc = (true_labels == predictions).mean()
    return acc


# ============================
#  3. Interfaz Streamlit
# ============================
def main():
    st.set_page_config(page_title="Predicción de Renuncia", layout="wide")
    st.title("📊 Modelo de Predicción de Renuncia de Empleados")
    st.markdown("Carga tu archivo de datos para obtener predicciones.")

    model, categorical_mapping, scaler, df_reference_features, true_labels_reference = load_model()
    if model is None:
        return

    # Estas columnas DEBEN coincidir con las usadas en el entrenamiento y en el orden exacto.
    model_feature_columns = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
        'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager',
        'IntencionPermanencia', 'CargaLaboralPercibida', 'SatisfaccionSalarial', 
        'ConfianzaEmpresa', 'NumeroTardanzas', 'NumeroFaltas' 
    ]
    
    uploaded_file = st.file_uploader("Sube un archivo CSV o Excel (.csv, .xlsx) para PREDICCIÓN", type=["csv", "xlsx"])
    
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

        df_original = df.copy() 
        df_features_uploaded = df_original.drop(columns=['Attrition'], errors='ignore').copy()
        
        # Usamos df_features_uploaded para el preprocesamiento
        processed_df = preprocess_data(df_features_uploaded, model_feature_columns, categorical_mapping, scaler)
        
        if processed_df is None:
            st.error("No se puede continuar con la predicción debido a un error de preprocesamiento.")
            return 

        st.header("1. Predicción con Datos Cargados")
        
        if st.button("🚀 Ejecutar Predicción"):
            st.info("Ejecutando el modelo sobre los datos cargados...")

            # Realizar predicción
            probabilidad_renuncia = model.predict_proba(processed_df)[:, 1]
            predictions = (probabilidad_renuncia > 0.5).astype(int)
            
            # Unir resultados a la data original
            df_original['Prediction_Renuncia'] = predictions
            df_original['Probabilidad_Renuncia'] = probabilidad_renuncia

            # Recomendaciones y visualización
            display_risk_alert(df_original)

            # Botón de descarga
            st.download_button(
                label="⬇️ Descargar Resultados de Predicción (Excel)",
                data=export_results_to_excel(df_original),
                file_name="predicciones_resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ============================
#  Alertas y Visualización
# ============================
def display_risk_alert(df):
    st.markdown("---")
    st.header("🚨 Alerta de Empleados con Riesgo de Deserción")

    # Categorizar el riesgo
    df['Riesgo'] = pd.cut(df['Probabilidad_Renuncia'], bins=[0, 0.3, 0.7, 1], labels=["Bajo", "Medio", "Alto"])
    
    # Semaforización de riesgo
    colors = {
        "Bajo": "green",
        "Medio": "yellow",
        "Alto": "red"
    }
    
    st.dataframe(df[['EmployeeNumber', 'Name', 'Riesgo', 'Probabilidad_Renuncia']].style.applymap(lambda v: f'background-color: {colors[v]}', subset=['Riesgo']))

    # Gráfico Circular de Distribución de Riesgo
    fig, ax = plt.subplots(figsize=(5, 5))
    risk_dist = df['Riesgo'].value_counts()
    ax.pie(risk_dist, labels=risk_dist.index, autopct='%1.1f%%', startangle=90, colors=["#28a745", "#f9d04e", "#dc3545"])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Gráfico de Barra para Distribución de Riesgo por Departamento
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby(['Department', 'Riesgo']).size().unstack().plot(kind='bar', stacked=True, ax=ax, color=['#28a745', '#f9d04e', '#dc3545'])
    ax.set_ylabel('Número de Empleados')
    ax.set_xlabel('Departamento')
    ax.set_title('Distribución de Riesgo por Departamento')
    st.pyplot(fig)

# ============================
#  Ejecutar la app
# ============================
if __name__ == "__main__":
    main()










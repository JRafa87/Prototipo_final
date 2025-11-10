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


# ============================
# 2. Preprocesamiento de Datos
# ============================
def preprocess_data(df, model_columns, categorical_mapping, scaler):
    df_processed = df.copy()

    # Rellenar nulos y codificar
    df_processed = df_processed.drop_duplicates()
    numeric_cols_for_fillna = df_processed.select_dtypes(include=np.number).columns.tolist()
    cols_to_fill = list(set(numeric_cols_for_fillna) & set(model_columns))
    df_processed[cols_to_fill] = df_processed[cols_to_fill].fillna(df_processed[cols_to_fill].mean())

    nominal_categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    for col in nominal_categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).str.strip().str.upper()
            if col in categorical_mapping:
                df_processed[col] = df_processed[col].map(categorical_mapping[col])
            df_processed[col] = df_processed[col].fillna(categorical_mapping.get(col, {}).get('DESCONOCIDO', -1))
        
    # Eliminar columnas que no forman parte del modelo, como 'FechaIngreso' y 'FechaSalida'
    df_processed = df_processed.drop(columns=['FechaIngreso', 'FechaSalida'], errors='ignore')

    # Ordenar y seleccionar solo las columnas de FEATURES ANTES DE ESCALAR
    df_to_scale = df_processed[model_columns].copy()
    try:
        df_processed[model_columns] = scaler.transform(df_to_scale)
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}. ¿El scaler fue entrenado correctamente con la forma de la data?")
        return None

    return df_processed[model_columns]


# ============================
# 3. Funciones para Mostrar Alertas y Resultados
# ============================
def display_risk_alert(df, model, categorical_mapping, scaler, model_feature_columns):
    # Preprocesar los datos
    processed_df = preprocess_data(df, model_feature_columns, categorical_mapping, scaler)

    if processed_df is None:
        st.error("No se puede continuar con la predicción debido a un error de preprocesamiento.")
        return 

    # Hacer la predicción de la probabilidad de renuncia
    probabilidad_renuncia = model.predict_proba(processed_df)[:, 1]
    df['Probabilidad_Renuncia'] = probabilidad_renuncia

    # Clasificar riesgo
    df['Riesgo'] = pd.cut(df['Probabilidad_Renuncia'], bins=[0, 0.33, 0.66, 1], labels=["Bajo", "Medio", "Alto"])

    # Colores para semaforización
    colors = {
        "Bajo": "green",
        "Medio": "yellow",
        "Alto": "red"
    }

    # Mostrar la tabla con el riesgo semaforizado
    st.subheader("Alertas de Riesgo de Deserción")
    st.dataframe(df[['EmployeeNumber', 'Name', 'Riesgo', 'Probabilidad_Renuncia']].style.applymap(lambda v: f'background-color: {colors[v]}', subset=['Riesgo']))

    # Gráfico de barras de probabilidad de renuncia por riesgo
    st.subheader("Distribución de Riesgo de Deserción")
    risk_counts = df['Riesgo'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(risk_counts.index, risk_counts.values, color=['green', 'yellow', 'red'])
    ax.set_xlabel("Riesgo")
    ax.set_ylabel("Número de Empleados")
    st.pyplot(fig)

    # Gráfico circular de la distribución de riesgo
    st.subheader("Distribución Circular del Riesgo")
    fig, ax = plt.subplots()
    ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=['green', 'yellow', 'red'])
    st.pyplot(fig)

    # Recomendaciones
    st.subheader("Recomendaciones Estratégicas")
    high_risk_df = df[df['Riesgo'] == 'Alto']

    if not high_risk_df.empty:
        st.markdown(f"**Acción sugerida para empleados con alto riesgo de deserción ({len(high_risk_df)} empleados):**")
        st.info("Realizar entrevistas de retención, ofrecer incentivos o revisar las condiciones laborales.")

    # Mostrar la lista con el riesgo y la probabilidad
    st.subheader("Empleados con Alto Riesgo")
    st.dataframe(high_risk_df[['EmployeeNumber', 'Name', 'Riesgo', 'Probabilidad_Renuncia']])

# ============================
# 4. Interfaz Streamlit
# ============================
def main():
    st.set_page_config(page_title="Predicción de Deserción de Empleados", layout="wide")
    st.title("📊 Predicción de Deserción de Empleados")

    # Cargar el modelo y artefactos
    model, categorical_mapping, scaler, df_reference_features, true_labels_reference = load_model()
    if model is None:
        return 

    # Definir las columnas esperadas por el modelo
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
    
    # Subir archivo CSV
    uploaded_file = st.file_uploader("Sube tu archivo CSV con datos de empleados", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Datos cargados con éxito, procesando información...")
        display_risk_alert(df, model, categorical_mapping, scaler, model_feature_columns)

if __name__ == "__main__":
    main()














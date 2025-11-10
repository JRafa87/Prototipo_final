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
        scaler = joblib.load('models/scaler.pkl')

        REFERENCE_DATA_PATH = 'data/reference_data.csv'
        if not os.path.exists(REFERENCE_DATA_PATH):
            st.error(f"Error: No se encontró la data de referencia en '{REFERENCE_DATA_PATH}'. Necesaria para evaluación de simulaciones.")
            return None, None, None, None, None

        df_reference = pd.read_csv(REFERENCE_DATA_PATH)
        if 'Attrition' not in df_reference.columns:
            st.error("Error: La data de referencia debe contener la columna 'Attrition' para la evaluación.")
            return None, None, None, None, None

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


# ============================
# 2. Función de Preprocesamiento
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

    df_to_scale = df_processed[model_columns].copy()

    try:
        df_processed[model_columns] = scaler.transform(df_to_scale)
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}. ¿El scaler fue entrenado correctamente con la forma de la data?")
        return None

    return df_processed[model_columns]


# ============================
# 3. Función para Semaforización y Predicción
# ============================
def display_risk_alert(df, model, categorical_mapping, scaler, model_feature_columns):
    processed_df = preprocess_data(df, model_feature_columns, categorical_mapping, scaler)

    if processed_df is None:
        st.error("No se puede continuar con la predicción debido a un error de preprocesamiento.")
        return

    # Realizar la predicción de probabilidad de renuncia
    probabilidad_renuncia = model.predict_proba(processed_df)[:, 1]
    df['Probabilidad_Renuncia'] = probabilidad_renuncia

    # Semaforización: Asignar color basado en la probabilidad
    def semaforo(probabilidad):
        if probabilidad <= 0.33:
            return 'Bajo', 'green'
        elif probabilidad <= 0.66:
            return 'Medio', 'yellow'
        else:
            return 'Alto', 'red'

    # Crear la columna de 'Riesgo' y 'Color' basado en la probabilidad
    df['Riesgo'], df['Color'] = zip(*df['Probabilidad_Renuncia'].apply(semaforo))

    # Mostrar la tabla con la semaforización
    st.subheader("Alertas de Riesgo de Deserción")
    st.dataframe(df[['EmployeeNumber', 'Probabilidad_Renuncia', 'Riesgo']].style.applymap(
        lambda v: f'background-color: {v}', subset=['Color']
    ))

    # Mostrar un gráfico de barras para la distribución de los riesgos
    st.subheader("Distribución de Riesgo de Deserción")
    risk_counts = df['Riesgo'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(risk_counts.index, risk_counts.values, color=['green', 'yellow', 'red'])
    ax.set_xlabel("Riesgo")
    ax.set_ylabel("Número de Empleados")
    st.pyplot(fig)

    # Gráfico circular
    st.subheader("Distribución Circular del Riesgo")
    fig, ax = plt.subplots()
    ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=['green', 'yellow', 'red'])
    st.pyplot(fig)

    # Recomendaciones para los empleados con alto riesgo
    st.subheader("Recomendaciones Estratégicas")
    high_risk_df = df[df['Riesgo'] == 'Alto']

    if not high_risk_df.empty:
        st.markdown(f"**Acción sugerida para empleados con alto riesgo de deserción ({len(high_risk_df)} empleados):**")
        st.info("Realizar entrevistas de retención, ofrecer incentivos o revisar las condiciones laborales.")

    # Mostrar empleados con alto riesgo
    st.subheader("Empleados con Alto Riesgo")
    st.dataframe(high_risk_df[['EmployeeNumber', 'Probabilidad_Renuncia', 'Riesgo']])


# ============================
# 4. Interfaz Streamlit
# ============================
def main():
    st.set_page_config(page_title="Predicción de Deserción", layout="wide")
    st.title("📊 Modelo de Predicción de Deserción de Empleados")
    st.markdown("Sube tu archivo de datos para obtener las predicciones de deserción.")

    # Cargar el modelo y artefactos
    model, categorical_mapping, scaler, df_reference_features, true_labels_reference = load_model()
    if model is None:
        return

    # Columnas del modelo (debe coincidir con las columnas del entrenamiento)
    model_feature_columns = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
        'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager', 'IntencionPermanencia', 'CargaLaboralPercibida', 
        'SatisfaccionSalarial', 'ConfianzaEmpresa', 'NumeroTardanzas', 'NumeroFaltas'
    ]

    # Subir archivo
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

        # Mostrar alertas de riesgo y predicciones
        display_risk_alert(df, model, categorical_mapping, scaler, model_feature_columns)


# ============================
# Inicio de la Aplicación
# ============================
if __name__ == "__main__":
    main()















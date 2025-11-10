import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt

# ============================
# 1. Cargar Modelos y Artefactos
# ============================
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
            return None, None, None, None

        df_reference = pd.read_csv(REFERENCE_DATA_PATH)

        if 'Attrition' not in df_reference.columns:
            st.error("Error: La data de referencia debe contener la columna 'Attrition' para la evaluación.")
            return None, None, None, None

        df_reference['Attrition'] = df_reference['Attrition'].replace({'Yes': 1, 'No': 0})
        true_labels_reference = df_reference['Attrition'].astype(int).copy()
        df_reference_features = df_reference.drop(columns=['Attrition'], errors='ignore').copy()
        
        st.success("✅ Modelo y artefactos cargados correctamente.")
        return model, categorical_mapping, scaler, df_reference_features, true_labels_reference

    except FileNotFoundError:
        st.error("Error: Archivos del modelo (xgboost_model.pkl, categorical_mapping.pkl, scaler.pkl) no encontrados. Asegúrate de tener la carpeta 'models' con los 3 archivos.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error al cargar artefactos o data de referencia: {e}")
        return None, None, None, None


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
        
    # Ordenar y seleccionar solo las columnas de FEATURES ANTES DE ESCALAR
    df_to_scale = df_processed[model_columns].copy()
    try:
        df_processed[model_columns] = scaler.transform(df_to_scale)
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}. ¿El scaler fue entrenado correctamente con la forma de la data?")
        return None

    return df_processed[model_columns]


# ============================
# 3. Mostrar Alerta de Riesgo
# ============================
def display_risk_alert(df, model, categorical_mapping, scaler, model_feature_columns):
    df_features = df.drop(columns=['Attrition'], errors='ignore').copy()  # Asegúrate de que 'Attrition' no esté en el DataFrame
    df_processed = preprocess_data(df_features, model_feature_columns, categorical_mapping, scaler)

    if df_processed is None:
        st.error("No se puede continuar con la predicción debido a un error de preprocesamiento.")
        return

    # Realizar predicciones de probabilidad de renuncia con el modelo cargado
    probabilidad_renuncia = model.predict_proba(df_processed)[:, 1]  # Obtener la probabilidad de la clase "1" (renuncia)

    # Crear la columna de Probabilidad de Renuncia en el DataFrame original
    df['Probabilidad_Renuncia'] = probabilidad_renuncia

    # Crear la columna de Riesgo basado en la probabilidad de renuncia
    df['Riesgo'] = pd.cut(df['Probabilidad_Renuncia'], bins=[0, 0.3, 0.7, 1], labels=["Bajo", "Medio", "Alto"])

    # Semaforización de riesgo
    colors = {
        "Bajo": "green",
        "Medio": "yellow",
        "Alto": "red"
    }

    # Mostrar la tabla con semaforización
    st.markdown("### Empleados con Riesgo de Deserción")
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

    # Recomendaciones Estratégicas
    st.markdown("### Recomendaciones Estratégicas para Reducir la Deserción")
    high_risk_df = df[df['Riesgo'] == 'Alto']
    if not high_risk_df.empty:
        st.subheader(f"Intervención Prioritaria para {len(high_risk_df)} Empleados de Alto Riesgo")
        st.write("Sugerencia: Iniciar entrevistas de retención confidenciales con los empleados de alto riesgo para entender sus preocupaciones y ofrecer soluciones personalizadas.")

    # Recomendaciones para áreas con alto riesgo de deserción
    high_risk_by_dept = df[df['Riesgo'] == 'Alto'].groupby('Department').size()
    if not high_risk_by_dept.empty:
        st.subheader("Áreas con Mayor Riesgo de Deserción")
        st.write(f"Departamentos con más empleados de alto riesgo de deserción: {high_risk_by_dept.index.tolist()}")
        st.write("Sugerencia: Evaluar la carga de trabajo, liderazgo y satisfacción en estas áreas para mitigar la deserción.")

# ============================
# 4. Función Principal de Streamlit
# ============================
def main():
    st.set_page_config(page_title="Predicción y Simulación de Renuncia", layout="wide")
    st.title("📊 Modelo de Predicción y Simulación de Renuncia de Empleados")
    st.markdown("Carga tu archivo de datos para obtener predicciones. Las simulaciones usan una **data de referencia** cargada en el servidor para evaluación.")

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
        'IntencionPermanencia', 'CargaLaboralPercibida', 'SatisfaccionSalarial', 'ConfianzaEmpresa',
        'NumeroTardanzas', 'NumeroFaltas', 'FechaIngreso', 'FechaSalida'
    ]

    # Subir archivo CSV
    uploaded_file = st.file_uploader("Sube tu archivo CSV con datos de empleados", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Datos cargados con éxito, procesando información...")
        display_risk_alert(df, model, categorical_mapping, scaler, model_feature_columns)

if __name__ == "__main__":
    main()













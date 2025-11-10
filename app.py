import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ============================
# 1. Cargar el Modelo y Escalador
# ============================
def load_model():
    model = joblib.load('xgboost_model.pkl')  # Asegúrate de tener tu modelo XGBoost guardado como un archivo .pkl
    return model

def load_scaler():
    scaler = joblib.load('scaler.pkl')  # Asegúrate de tener tu escalador guardado como un archivo .pkl
    return scaler

# ============================
# 2. Preprocesamiento de Datos
# ============================
def preprocess_data(df, model_feature_columns, categorical_mapping, scaler):
    # Convertir variables categóricas
    for col, mapping in categorical_mapping.items():
        df[col] = df[col].map(mapping).fillna(-1)
    
    # Eliminar columnas no necesarias
    df = df[model_feature_columns]
    
    # Escalar datos
    try:
        df_scaled = scaler.transform(df)
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}")
        return None
    
    return pd.DataFrame(df_scaled, columns=model_feature_columns)

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

    # Verificar si las columnas necesarias existen
    if 'Riesgo' not in df.columns or 'Color' not in df.columns:
        st.error("Las columnas necesarias para la semaforización ('Riesgo' y 'Color') no fueron creadas correctamente.")
        return

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
# 4. Función principal
# ============================
def main():
    # Cargar el modelo y el escalador
    model = load_model()
    scaler = load_scaler()

    # Definir las columnas de características del modelo y el mapeo de variables categóricas
    model_feature_columns = ['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
                             'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction',
                             'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
                             'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike',
                             'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
                             'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                             'YearsWithCurrManager', 'IntencionPermanencia', 'CargaLaboralPercibida', 'SatisfaccionSalarial', 
                             'ConfianzaEmpresa', 'NumeroTardanzas', 'NumeroFaltas', 'FechaIngreso', 'FechaSalida']
    
    categorical_mapping = {
        'BusinessTravel': {'Travel_Rarely': 0, 'Travel_Frequently': 1, 'Non-Travel': 2},
        'Department': {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2},
        'EducationField': {'Life Sciences': 0, 'Other': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4, 'Human Resources': 5},
        'Gender': {'Male': 0, 'Female': 1},
        'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2},
        'OverTime': {'No': 0, 'Yes': 1}
    }

    # Subir archivo CSV de empleados
    uploaded_file = st.file_uploader("Cargar archivo CSV con los datos de empleados", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Datos de Empleados")
        st.dataframe(df.head())

        # Mostrar alertas de riesgo
        display_risk_alert(df, model, categorical_mapping, scaler, model_feature_columns)

# ============================
# Ejecutar la aplicación
# ============================
if __name__ == "__main__":
    main()
















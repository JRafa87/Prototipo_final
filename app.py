import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
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


# ============================
# Funciones de Preprocesamiento
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
        
    # Escalado
    try:
        df_to_scale = df_processed[model_columns].copy()
        df_processed[model_columns] = scaler.transform(df_to_scale)
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}. ¿El scaler fue entrenado correctamente con la forma de la data?")
        return None

    return df_processed[model_columns]

# ============================
# Evaluación de Desbalance
# ============================
def check_class_imbalance(df, target_column='Attrition'):
    class_counts = df[target_column].value_counts()
    total_instances = len(df)
    class_percentages = class_counts / total_instances * 100
    return class_percentages.min() < 30

# ============================
# Evaluación de Predicción
# ============================
def dynamic_threshold(probabilidad_renuncia, imbalance_detected):
    threshold = 0.3 if imbalance_detected else 0.5
    predictions = (probabilidad_renuncia > threshold).astype(int)
    return predictions

# ============================
# Función de Visualización del Riesgo
# ============================
def plot_risk_circular(risk_score):
    # Mapa de colores de riesgo
    if risk_score > 0.7:
        color = 'red'
    elif risk_score > 0.4:
        color = 'yellow'
    else:
        color = 'green'

    # Crear un gráfico circular con el riesgo
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie([risk_score, 1 - risk_score], labels=["Riesgo", "Seguro"], colors=[color, 'lightgray'], startangle=90, counterclock=False, autopct='%1.1f%%')
    ax.set_title("Riesgo de Deserción")
    st.pyplot(fig)

# ============================
# Función de Recomendaciones Contextuales
# ============================
def display_recommendations(df_results):
    st.markdown("---")
    st.header("💡 Recomendaciones Estratégicas")

    high_risk_threshold = 0.70
    df_high_risk = df_results[df_results['Probabilidad_Renuncia'] > high_risk_threshold]

    if df_high_risk.empty:
        st.success("El riesgo de deserción es bajo. Mantenga las políticas actuales.")
        return

    st.subheader("1. Intervención Individual Prioritaria")
    st.markdown(f"**Enfocarse en los {len(df_high_risk)} empleados con la más alta probabilidad de renuncia** (Probabilidad > {high_risk_threshold * 100:.0f}%).")
    
    # Agregar un ícono de ojo para desplegar más información
    with st.expander("Ver más detalles de intervención"):
        st.info("Acción sugerida: Realizar entrevistas de retención confidenciales para entender sus preocupaciones y ofrecer soluciones personalizadas.")

# ============================
# Función de Consulta Manual
# ============================
def manual_query_and_simulation(df_reference_features, model, categorical_mapping, scaler, model_feature_columns):
    st.header("Consulta Manual de Deserción")

    employee_id = st.text_input("ID de Empleado (si deseas consultar un solo empleado):")

    if employee_id:
        employee_data = df_reference_features[df_reference_features['EmployeeNumber'] == int(employee_id)]
        if not employee_data.empty:
            processed_data = preprocess_data(employee_data, model_feature_columns, categorical_mapping, scaler)
            if processed_data is not None:
                probabilidad_renuncia = model.predict_proba(processed_data)[:, 1]
                st.write(f"**Probabilidad de Renuncia para el empleado {employee_id}:** {probabilidad_renuncia[0]:.2f}")
                plot_risk_circular(probabilidad_renuncia[0])
                display_recommendations(employee_data)
        else:
            st.warning("No se encontró el empleado con ese ID.")

    # Para consulta grupal
    group = st.selectbox("Seleccionar grupo de empleados para consulta:", df_reference_features['Department'].unique())
    group_data = df_reference_features[df_reference_features['Department'] == group]
    
    if st.button(f"Consultar deserción para el grupo {group}"):
        processed_group_data = preprocess_data(group_data, model_feature_columns, categorical_mapping, scaler)
        if processed_group_data is not None:
            probabilidad_renuncia_group = model.predict_proba(processed_group_data)[:, 1]
            group_data['Probabilidad_Renuncia'] = probabilidad_renuncia_group
            group_data['Riesgo'] = group_data['Probabilidad_Renuncia'].apply(lambda x: 'Alto' if x > 0.7 else 'Bajo' if x < 0.4 else 'Moderado')
            st.write(group_data[['EmployeeNumber', 'Probabilidad_Renuncia', 'Riesgo']])
            st.success("Consulta de grupo realizada exitosamente.")
            
# ============================
# Interfaz Streamlit Principal
# ============================
def main():
    st.set_page_config(page_title="Predicción de Deserción de Empleados", layout="wide")
    st.title("📊 Modelo de Predicción de Deserción de Empleados")
    st.markdown("Carga un archivo CSV o Excel y visualiza la probabilidad de deserción de los empleados.")

    model, categorical_mapping, scaler, df_reference_features, true_labels_reference = load_model()
    if model is None:
        return

    model_feature_columns = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
        'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager'
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
        
        # Preprocesar la data cargada
        processed_df = preprocess_data(df_features_uploaded, model_feature_columns, categorical_mapping, scaler)
        
        if processed_df is None:
            st.error("No se puede continuar con la predicción debido a un error de preprocesamiento.")
            return 

        st.header("1. Predicción con Datos Cargados")
        if st.button("🚀 Ejecutar Predicción"):
            st.info("Ejecutando el modelo sobre los datos cargados...")
            
            imbalance_detected = check_class_imbalance(df, target_column='Attrition')
            probabilidad_renuncia = model.predict_proba(processed_df)[:, 1]
            predictions = dynamic_threshold(probabilidad_renuncia, imbalance_detected)

            df_original['Prediction_Renuncia'] = predictions
            df_original['Probabilidad_Renuncia'] = probabilidad_renuncia

            # Visualizar el riesgo de deserción
            for i, row in df_original.iterrows():
                st.subheader(f"Empleado {row['EmployeeNumber']}")
                plot_risk_circular(row['Probabilidad_Renuncia'])

            # Mostrar recomendaciones
            display_recommendations(df_original)

    # Consultas manuales
    st.divider()
    manual_query_and_simulation(df_reference_features, model, categorical_mapping, scaler, model_feature_columns)

# ====================================
# Inicio de la Aplicación
# ====================================
if __name__ == "__main__":
    main()

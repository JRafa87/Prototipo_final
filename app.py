import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# =======================
# Funciones de carga
# =======================
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

        # Solución al error 'invalid literal for int(): 'Yes''
        df_reference['Attrition'] = df_reference['Attrition'].replace({'Yes': 1, 'No': 0})

        true_labels_reference = df_reference['Attrition'].astype(int).copy()
        df_reference_features = df_reference.drop(columns=['Attrition'], errors='ignore').copy()
        
        st.success("✅ Modelo y artefactos cargados correctamente.")
        return model, categorical_mapping, scaler, df_reference_features, true_labels_reference

    except FileNotFoundError:
        st.error("Error: Archivos del modelo no encontrados.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error al cargar artefactos o data de referencia: {e}")
        return None, None, None, None

# =======================
# Preprocesamiento de datos
# =======================
def preprocess_data(df, model_columns, categorical_mapping, scaler):
    df_processed = df.copy()

    # Rellenar nulos y codificar
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
        st.error(f"Error al escalar los datos: {e}.")
        return None

    return df_processed[model_columns]

# =======================
# Mostrar riesgo de deserción
# =======================
def plot_risk_circular(probabilidad_renuncia):
    fig, ax = plt.subplots(figsize=(5,5))
    labels = ['Deserción Baja', 'Deserción Alta']
    sizes = [100 - probabilidad_renuncia * 100, probabilidad_renuncia * 100]
    colors = ['green', 'red']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# =======================
# Simulación de escenarios
# =======================
def simulate_scenario(df_features, model, categorical_mapping, scaler, model_feature_columns):
    # Preprocesar la data ingresada
    processed_data = preprocess_data(df_features, model_feature_columns, categorical_mapping, scaler)

    if processed_data is not None:
        # Ejecutar la predicción
        probabilidad_renuncia = model.predict_proba(processed_data)[:, 1]
        prediction = 'Sí' if probabilidad_renuncia > 0.5 else 'No'
        return probabilidad_renuncia, prediction
    return None, None

# =======================
# Función principal de la interfaz
# =======================
def manual_query_and_simulation(df_reference_features, model, categorical_mapping, scaler, model_feature_columns):
    st.header("Consulta Manual de Deserción y Simulación de Escenarios")

    st.markdown("### Ingresa las variables del empleado o grupo en español:")

    # Crear un formulario de entrada para el empleado
    with st.form(key="manual_input_form"):
        # Ingresar las variables del empleado
        edad = st.number_input("Edad", min_value=18, max_value=100, value=30)
        viaje_negocios = st.selectbox("Viajes de negocio", options=["No", "Viaja", "Frecuente"])
        departamento = st.selectbox("Departamento", options=["Ventas", "TI", "Recursos Humanos", "Marketing", "Operaciones"])
        ingresos_mensuales = st.number_input("Ingreso mensual", min_value=1000, max_value=20000, value=5000)
        nivel_satisfaccion_trabajo = st.slider("Satisfacción en el trabajo", min_value=1, max_value=4, value=3)
        años_en_empresa = st.number_input("Años en la empresa", min_value=0, max_value=50, value=5)
        horas_trabajadas = st.number_input("Horas trabajadas a la semana", min_value=30, max_value=80, value=40)
        salario_hora = st.number_input("Salario por hora", min_value=10, max_value=100, value=25)

        submit_button = st.form_submit_button(label="Simular Escenario")

    if submit_button:
        # Crear un DataFrame para las variables ingresadas
        input_data = pd.DataFrame({
            "Age": [edad],
            "BusinessTravel": [viaje_negocios],
            "Department": [departamento],
            "MonthlyIncome": [ingresos_mensuales],
            "JobSatisfaction": [nivel_satisfaccion_trabajo],
            "YearsAtCompany": [años_en_empresa],
            "HourlyRate": [salario_hora],
        })

        # Predecir el riesgo de deserción con el modelo
        probabilidad_renuncia, prediction = simulate_scenario(input_data, model, categorical_mapping, scaler, model_feature_columns)

        if probabilidad_renuncia is not None:
            st.markdown(f"**Probabilidad de deserción:** {probabilidad_renuncia * 100:.2f}%")
            st.markdown(f"**Predicción de deserción:** {prediction}")

            # Mostrar gráfico circular del riesgo
            plot_risk_circular(probabilidad_renuncia)

            # Recomendaciones basadas en la predicción
            if probabilidad_renuncia > 0.5:
                st.warning("¡Este empleado tiene un alto riesgo de deserción! Recomendamos intervenir pronto.")
            else:
                st.success("Este empleado tiene un bajo riesgo de deserción.")

# =======================
# Función principal de la app
# =======================
def main():
    st.set_page_config(page_title="Simulador de Deserción de Empleados", layout="wide")
    st.title("Simulador de Deserción de Empleados")

    model, categorical_mapping, scaler, df_reference_features, true_labels_reference = load_model()
    if model is None:
        return 

    model_feature_columns = [
        'Age', 'BusinessTravel', 'Department', 'MonthlyIncome', 'JobSatisfaction', 
        'YearsAtCompany', 'HourlyRate'
    ]

    # Crear una barra de navegación (opciones de pestañas)
    st.sidebar.header("Menú")
    app_mode = st.sidebar.radio("Selecciona una opción", ["Cargar archivo CSV", "Simulación manual"])

    if app_mode == "Cargar archivo CSV":
        st.header("Carga de datos y predicción automática")

        # Subir archivo CSV para predicción automática
        uploaded_file = st.file_uploader("Sube un archivo CSV o Excel (.csv, .xlsx) para PREDICCIÓN", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # Procesar y hacer predicción para cada fila
                st.write(df.head())
                st.markdown("### Predicciones automáticas de deserción")
                df_processed = preprocess_data(df, model_feature_columns, categorical_mapping, scaler)
                if df_processed is not None:
                    prediction_proba = model.predict_proba(df_processed)[:, 1]
                    df['Probabilidad de Deserción'] = prediction_proba
                    df['Predicción'] = np.where(prediction_proba > 0.5, "Sí", "No")
                    st.write(df[['Probabilidad de Deserción', 'Predicción']])

                    # Mostrar gráficos o análisis adicionales si es necesario
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")

    elif app_mode == "Simulación manual":
        # Llamar a la función de consulta manual y simulación
        manual_query_and_simulation(df_reference_features, model, categorical_mapping, scaler, model_feature_columns)

# =======================
# Iniciar la app
# =======================
if __name__ == "__main__":
    main()


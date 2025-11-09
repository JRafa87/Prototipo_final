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
# Visualizar riesgo de deserción
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
# Simulación y predicción
# =======================
def make_predictions(df, model, categorical_mapping, scaler, model_feature_columns):
    df_processed = preprocess_data(df, model_feature_columns, categorical_mapping, scaler)

    if df_processed is not None:
        # Predicción
        probabilidad_renuncia = model.predict_proba(df_processed)[:, 1]
        predictions = (probabilidad_renuncia > 0.5).astype(int)
        
        # Añadir las predicciones al DataFrame original
        df['Prediction_Renuncia'] = predictions
        df['Probabilidad_Renuncia'] = probabilidad_renuncia

        return df
    return None

# =======================
# Función para ingresar manualmente las variables
# =======================
def simulate_scenario(model, categorical_mapping, scaler, model_feature_columns):
    st.write("### Simulación Manual de Escenario")

    # Crear campos para ingresar las variables manualmente
    edad = st.number_input("Edad", min_value=18, max_value=100, value=30)
    ingreso_mensual = st.number_input("Ingreso Mensual", min_value=1000, max_value=20000, value=3000)
    satisfaccion_laboral = st.slider("Satisfacción Laboral", min_value=1, max_value=4, value=3)
    años_en_empresa = st.number_input("Años en la Empresa", min_value=0, max_value=50, value=5)

    # Crear un dataframe con estas variables
    input_data = {
        'Age': [edad],
        'MonthlyIncome': [ingreso_mensual],
        'JobSatisfaction': [satisfaccion_laboral],
        'YearsAtCompany': [años_en_empresa],
    }

    input_df = pd.DataFrame(input_data)

    # Preprocesar el dataframe y hacer la predicción
    result_df = make_predictions(input_df, model, categorical_mapping, scaler, model_feature_columns)

    if result_df is not None:
        st.write("### Resultados de la Predicción Manual")
        st.write(result_df[['Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany', 'Prediction_Renuncia', 'Probabilidad_Renuncia']])
        
        # Mostrar gráfico de riesgo
        st.write("Gráfico de riesgo de deserción")
        plot_risk_circular(result_df['Probabilidad_Renuncia'][0])

        # Recomendaciones
        st.write("### Recomendaciones:")
        if result_df['Probabilidad_Renuncia'][0] > 0.7:
            st.markdown("**Recomendación:** Priorizar intervención, realizar entrevistas de retención, ofrecer soluciones personalizadas.")
        else:
            st.markdown("**Recomendación:** Continuar con las políticas actuales, monitorear periódicamente su satisfacción.")

# =======================
# Función principal
# =======================
def main():
    st.set_page_config(page_title="Simulador de Deserción de Empleados", layout="wide")
    st.title("Simulador de Deserción de Empleados")

    model, categorical_mapping, scaler, df_reference_features, true_labels_reference = load_model()
    if model is None:
        return

    # Definir las columnas del modelo
    model_feature_columns = [
        'Age', 'BusinessTravel', 'Department', 'MonthlyIncome', 'JobSatisfaction', 
        'YearsAtCompany', 'HourlyRate', 'DistanceFromHome', 'Education', 'Gender',
        'JobRole', 'MaritalStatus', 'OverTime'
    ]

    # Crear un menú de selección de pestañas
    page = st.radio("Selecciona una opción", ["Predicción con archivo CSV", "Simulación Manual de Escenario"])

    if page == "Predicción con archivo CSV":
        # Cargar el archivo CSV
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(df.head())  # Muestra los primeros 5 registros del archivo subido

            if st.button("Realizar Predicciones"):
                # Realizar la predicción
                result_df = make_predictions(df, model, categorical_mapping, scaler, model_feature_columns)

                if result_df is not None:
                    # Mostrar resultados
                    st.write("Resultados de las predicciones")
                    st.write(result_df[['EmployeeNumber', 'Prediction_Renuncia', 'Probabilidad_Renuncia']])

                    # Mostrar gráfico de riesgo
                    st.write("Gráfico de riesgo de deserción")
                    for index, row in result_df.iterrows():
                        plot_risk_circular(row['Probabilidad_Renuncia'])

                    # Recomendaciones
                    st.write("### Recomendaciones:")
                    for index, row in result_df.iterrows():
                        if row['Probabilidad_Renuncia'] > 0.7:
                            st.write(f"Empleado {row['EmployeeNumber']} con alta probabilidad de deserción ({row['Probabilidad_Renuncia']*100:.2f}%)")
                            st.markdown("**Recomendación:** Priorizar intervención, realizar entrevistas de retención, ofrecer soluciones personalizadas.")
                        else:
                            st.write(f"Empleado {row['EmployeeNumber']} con baja probabilidad de deserción ({row['Probabilidad_Renuncia']*100:.2f}%)")
                            st.markdown("**Recomendación:** Continuar con las políticas actuales, monitorear periódicamente su satisfacción.")

    elif page == "Simulación Manual de Escenario":
        simulate_scenario(model, categorical_mapping, scaler, model_feature_columns)

# =======================
# Ejecutar la app
# =======================
if __name__ == "__main__":
    main()





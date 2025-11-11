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
        model = joblib.load('models/xgboost_model.pkl')
        categorical_mapping = joblib.load('models/categorical_mapping.pkl')
        scaler = joblib.load('models/scaler.pkl')

        REFERENCE_DATA_PATH = 'data/reference_data.csv'
        if not os.path.exists(REFERENCE_DATA_PATH):
            st.error(f"Error: No se encontró la data de referencia en '{REFERENCE_DATA_PATH}'.")
            return None, None, None, None, None

        df_reference = pd.read_csv(REFERENCE_DATA_PATH)
        if 'Attrition' not in df_reference.columns:
            st.error("Error: La data de referencia debe contener la columna 'Attrition'.")
            return None, None, None, None, None

        df_reference['Attrition'] = df_reference['Attrition'].replace({'Yes': 1, 'No': 0})
        true_labels_reference = df_reference['Attrition'].astype(int).copy()
        df_reference_features = df_reference.drop(columns=['Attrition'], errors='ignore').copy()

        st.success("✅ Modelo y artefactos cargados correctamente.")
        return model, categorical_mapping, scaler, df_reference_features, true_labels_reference

    except FileNotFoundError:
        st.error("Error: No se encontraron los archivos del modelo (xgboost_model.pkl, categorical_mapping.pkl, scaler.pkl).")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error al cargar artefactos o data de referencia: {e}")
        return None, None, None, None, None


# ================================
# 2. Funciones de Preprocesamiento
# ================================
def preprocess_data(df, model_columns, categorical_mapping, scaler):
    df_processed = df.copy()
    df_processed = df_processed.drop_duplicates()
    numeric_cols_for_fillna = df_processed.select_dtypes(include=np.number).columns.tolist()
    cols_to_fill = list(set(numeric_cols_for_fillna) & set(model_columns))
    df_processed[cols_to_fill] = df_processed[cols_to_fill].fillna(df_processed[cols_to_fill].mean())

    nominal_categorical_cols = [
        'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
        'MaritalStatus', 'OverTime'
    ]

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
        st.error(f"Error al escalar los datos: {e}")
        return None

    return df_processed[model_columns]


# ============================
# Función de Recomendación
# ============================
def generar_recomendacion_personalizada(row):
    recomendaciones = []

    if row.get('IntencionPermanencia', 3) <= 2:
        recomendaciones.append("Reforzar conversaciones de desarrollo profesional.")
    if row.get('CargaLaboralPercibida', 3) >= 4:
        recomendaciones.append("Revisar la carga laboral y redistribuir tareas.")
    if row.get('SatisfaccionSalarial', 3) <= 2:
        recomendaciones.append("Evaluar ajustes salariales o beneficios.")
    if row.get('ConfianzaEmpresa', 3) <= 2:
        recomendaciones.append("Fomentar la transparencia y la confianza.")
    if row.get('NumeroTardanzas', 0) > 3 or row.get('NumeroFaltas', 0) > 1:
        recomendaciones.append("Revisar causas de ausentismo y ofrecer apoyo.")

    if not recomendaciones:
        recomendaciones.append("Sin alertas relevantes. Mantener seguimiento preventivo.")

    return " | ".join(recomendaciones)


# ============================
# Función Exportación
# ============================
def export_results_to_excel(df, filename="predicciones_resultados.xlsx"):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as output:
        df.to_excel(output, sheet_name='Predicciones', index=False)
    return filename


# ============================
# 3. Interfaz Streamlit
# ============================
def main():
    st.set_page_config(page_title="Predicción de Renuncia", layout="wide")
    st.title("📊 Modelo de Predicción de Renuncia de Empleados")
    st.markdown("Carga tu archivo de datos para obtener predicciones y análisis de riesgo.")

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
        'YearsSinceLastPromotion', 'YearsWithCurrManager',
        'IntencionPermanencia', 'CargaLaboralPercibida', 'SatisfaccionSalarial', 
        'ConfianzaEmpresa', 'NumeroTardanzas', 'NumeroFaltas' 
    ]

    uploaded_file = st.file_uploader("📂 Sube un archivo CSV o Excel (.csv, .xlsx)", type=["csv", "xlsx"])
    
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
        processed_df = preprocess_data(df_features_uploaded, model_feature_columns, categorical_mapping, scaler)
        
        if processed_df is None:
            st.error("❌ Error de preprocesamiento. Verifica tus datos.")
            return

        st.header("1️⃣ Predicción de Renuncia")
        
        if st.button("🚀 Ejecutar Predicción"):
            st.info("Ejecutando modelo...")

            probabilidad_renuncia = model.predict_proba(processed_df)[:, 1]
            predictions = (probabilidad_renuncia > 0.5).astype(int)
            df_original['Prediction_Renuncia'] = predictions
            df_original['Probabilidad_Renuncia'] = probabilidad_renuncia
            df_original['Recomendacion'] = df_original.apply(generar_recomendacion_personalizada, axis=1)

            st.success("✅ Predicción completada correctamente")

            # ========================
            # 🔹 Alerta de alto riesgo
            # ========================
            high_risk_threshold = 0.61
            df_high_risk = df_original[df_original['Probabilidad_Renuncia'] > high_risk_threshold]
            count_high_risk = len(df_high_risk)

            if count_high_risk > 0:
                st.error(f"🚨 Se detectaron **{count_high_risk} empleados** con ALTA probabilidad de renuncia (>61%).")
            else:
                st.success("🎉 No se detectaron empleados con alto riesgo de renuncia.")

            # ========================
            # 🔹 Tabla Top 10 empleados
            # ========================
            st.subheader("👥 Top 10 empleados con mayor probabilidad de renuncia")

            df_top10 = df_original.sort_values(by='Probabilidad_Renuncia', ascending=False).head(10).copy()

            def color_prob(val):
                if val > 0.61:
                    return 'background-color: #ff4d4d; color: white;'  # rojo
                elif 0.50 <= val <= 0.60:
                    return 'background-color: #ffcc00; color: black;'  # amarillo
                else:
                    return 'background-color: #70db70; color: black;'  # verde

            st.dataframe(
                df_top10[['EmployeeNumber', 'Department', 'JobRole', 'MonthlyIncome', 
                          'Probabilidad_Renuncia', 'Recomendacion']]
                .style.applymap(color_prob, subset=['Probabilidad_Renuncia'])
                .format({'Probabilidad_Renuncia': "{:.2%}"})
            )

            # ========================
            # 🔹 Promedio por Departamento
            # ========================
            st.subheader("🏢 Promedio de probabilidad de renuncia por Departamento")
            dept_summary = (
                df_original.groupby('Department')['Probabilidad_Renuncia']
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )

            st.bar_chart(dept_summary.set_index('Department'))

            # ========================
            # 🔹 Botón de descarga
            # ========================
            st.download_button(
                label="⬇️ Descargar Resultados (Excel)",
                data=export_results_to_excel(df_original),
                file_name="predicciones_resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ============================
#  Inicio de la Aplicación
# ============================
if __name__ == "__main__":
    main()















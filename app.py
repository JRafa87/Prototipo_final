import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# ==========================
# 1. Cargar modelos y data
# ==========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/xgboost_model.pkl')
        categorical_mapping = joblib.load('models/categorical_mapping.pkl')
        scaler = joblib.load('models/scaler.pkl')

        REFERENCE_DATA_PATH = 'data/reference_data.csv'
        if not os.path.exists(REFERENCE_DATA_PATH):
            st.error(f"No se encontró la data de referencia: '{REFERENCE_DATA_PATH}'.")
            return None, None, None, None, None

        df_reference = pd.read_csv(REFERENCE_DATA_PATH)
        df_reference['Attrition'] = df_reference['Attrition'].replace({'Yes': 1, 'No': 0})
        true_labels_reference = df_reference['Attrition']
        df_reference_features = df_reference.drop(columns=['Attrition'], errors='ignore')

        st.success("✅ Modelo y artefactos cargados correctamente.")
        return model, categorical_mapping, scaler, df_reference_features, true_labels_reference

    except Exception as e:
        st.error(f"Error al cargar artefactos o data: {e}")
        return None, None, None, None, None


# ==========================
# 2. Preprocesamiento
# ==========================
def preprocess_data(df, model_columns, categorical_mapping, scaler):
    df_processed = df.copy().drop_duplicates()
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())

    categorical_cols = [
        'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
        'MaritalStatus', 'OverTime'
    ]

    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).str.strip().str.upper()
            if col in categorical_mapping:
                df_processed[col] = df_processed[col].map(categorical_mapping[col])
            df_processed[col] = df_processed[col].fillna(categorical_mapping.get(col, {}).get('DESCONOCIDO', -1))

    df_processed[model_columns] = scaler.transform(df_processed[model_columns])
    return df_processed[model_columns]


# ==========================
# 3. Recomendaciones
# ==========================
def generar_recomendacion_personalizada(row):
    recomendaciones = []
    if row.get('IntencionPermanencia', 3) <= 2:
        recomendaciones.append("Reforzar conversaciones de desarrollo profesional.")
    if row.get('CargaLaboralPercibida', 3) >= 4:
        recomendaciones.append("Revisar carga laboral y redistribuir tareas.")
    if row.get('SatisfaccionSalarial', 3) <= 2:
        recomendaciones.append("Evaluar ajustes salariales o beneficios.")
    if row.get('ConfianzaEmpresa', 3) <= 2:
        recomendaciones.append("Fomentar transparencia y confianza.")
    if row.get('NumeroTardanzas', 0) > 3 or row.get('NumeroFaltas', 0) > 1:
        recomendaciones.append("Revisar causas de ausentismo y ofrecer apoyo.")

    if not recomendaciones:
        recomendaciones.append("Sin alertas relevantes. Seguimiento preventivo.")
    return " • ".join(recomendaciones)


# ==========================
# 4. App principal
# ==========================
def main():
    st.set_page_config(page_title="Predicción de Renuncia", layout="wide")
    st.title("📊 Predicción de Renuncia de Empleados")

    model, categorical_mapping, scaler, df_reference_features, true_labels_reference = load_model()
    if model is None:
        return

    model_columns = [
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

    uploaded_file = st.file_uploader("📂 Sube un archivo CSV o Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.info(f"✅ Archivo cargado correctamente. Total de filas: {len(df)}")

        processed_df = preprocess_data(df.drop(columns=['Attrition'], errors='ignore'), model_columns, categorical_mapping, scaler)
        if processed_df is None:
            return

        if st.button("🚀 Ejecutar Predicción"):
            prob = model.predict_proba(processed_df)[:, 1]
            df['Probabilidad_Renuncia'] = prob
            df['Prediction_Renuncia'] = (prob > 0.5).astype(int)
            df['Recomendacion'] = df.apply(generar_recomendacion_personalizada, axis=1)

            # Semaforización según umbrales
            def color_riesgo(p):
                if p > 0.61:
                    return "🔴 Alto"
                elif 0.50 <= p <= 0.60:
                    return "🟡 Medio"
                else:
                    return "🟢 Bajo"

            df['Nivel_Riesgo'] = df['Probabilidad_Renuncia'].apply(color_riesgo)
            df['Probabilidad_%'] = (df['Probabilidad_Renuncia'] * 100).round(2).astype(str) + '%'

            # Tabla resumen
            df_tabla = df[['EmployeeNumber', 'JobRole', 'Department', 'MonthlyIncome',
                           'Nivel_Riesgo', 'Probabilidad_%', 'Recomendacion']]

            st.subheader("👥 Empleados con riesgo de renuncia")
            st.markdown("Haz clic en una fila para ver la recomendación completa 👇")

            # Usar st.data_editor con expandibles
            st.data_editor(
                df_tabla,
                column_config={
                    "Recomendacion": st.column_config.TextColumn(
                        "Recomendación personalizada", help="Sugerencias de retención"
                    )
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )

            # Gráficos por departamento
            st.subheader("📊 Promedio de probabilidad por departamento")
            dept_avg = df.groupby('Department')['Probabilidad_Renuncia'].mean().reset_index()

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    px.bar(dept_avg, x='Department', y='Probabilidad_Renuncia',
                           color='Probabilidad_Renuncia', color_continuous_scale='Reds',
                           title="Probabilidad promedio de renuncia por departamento"),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    px.pie(dept_avg, names='Department', values='Probabilidad_Renuncia',
                           title="Distribución general del riesgo por departamento",
                           hole=0.4),
                    use_container_width=True
                )


# ==========================
# 5. Ejecutar
# ==========================
if __name__ == "__main__":
    main()



















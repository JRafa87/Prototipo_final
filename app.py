import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

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

    except Exception as e:
        st.error(f"Error al cargar artefactos o data de referencia: {e}")
        return None, None, None, None, None


# ================================
# 2. Funciones de Preprocesamiento
# ================================
def preprocess_data(df, model_columns, categorical_mapping, scaler):
    df_processed = df.copy().drop_duplicates()
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    cols_to_fill = list(set(numeric_cols) & set(model_columns))
    df_processed[cols_to_fill] = df_processed[cols_to_fill].fillna(df_processed[cols_to_fill].mean())

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

    try:
        df_processed[model_columns] = scaler.transform(df_processed[model_columns])
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

    return " • ".join(recomendaciones)


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
        st.dataframe(df.head())

        processed_df = preprocess_data(df.drop(columns=['Attrition'], errors='ignore'), model_columns, categorical_mapping, scaler)
        if processed_df is None:
            return

        if st.button("🚀 Ejecutar Predicción"):
            prob = model.predict_proba(processed_df)[:, 1]
            df['Probabilidad_Renuncia'] = prob
            df['Prediction_Renuncia'] = (prob > 0.5).astype(int)
            df['Recomendacion'] = df.apply(generar_recomendacion_personalizada, axis=1)

            # ALERTA PRINCIPAL
            high_risk = df[df['Probabilidad_Renuncia'] > 0.5]
            if len(high_risk) > 0:
                st.error(f"🚨 Se detectaron **{len(high_risk)} empleados** con ALTA probabilidad de renuncia (>50%).")
            else:
                st.success("🎉 No se detectaron empleados con alto riesgo.")

            # SEMAFORIZACIÓN SUAVIZADA
            def color_prob(val):
                if val > 0.5:
                    return 'background-color: #FF9999; color: black;'  # rojo pastel
                elif 0.4 <= val <= 0.5:
                    return 'background-color: #FFE699; color: black;'  # amarillo pastel
                else:
                    return 'background-color: #B6D7A8; color: black;'  # verde pastel

            st.subheader("👥 Top 10 empleados con mayor probabilidad de renuncia")
            df_top10 = df.sort_values('Probabilidad_Renuncia', ascending=False).head(10)
            st.dataframe(
                df_top10[['EmployeeNumber', 'Department', 'JobRole', 'MonthlyIncome', 'Probabilidad_Renuncia']]
                .style.applymap(color_prob, subset=['Probabilidad_Renuncia'])
                .format({'Probabilidad_Renuncia': "{:.2%}"})
            )

            # RECOMENDACIONES EN MODAL
            with st.expander("💬 Ver recomendaciones personalizadas"):
                st.write(df_top10[['EmployeeNumber', 'JobRole', 'Recomendacion']])

            # GRÁFICO PIE POR DEPARTAMENTO
            st.subheader("🏢 Distribución de probabilidad promedio por departamento (%)")
            dept_avg = df.groupby('Department')['Probabilidad_Renuncia'].mean().reset_index()
            dept_avg['Porcentaje'] = 100 * dept_avg['Probabilidad_Renuncia'] / dept_avg['Probabilidad_Renuncia'].sum()
            fig_pie = px.pie(dept_avg, names='Department', values='Porcentaje',
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                             hole=0.4, title="Distribución total (100%)")
            st.plotly_chart(fig_pie, use_container_width=True)

            # GRÁFICO DE IMPORTANCIA DE VARIABLES
            if hasattr(model, "feature_importances_"):
                st.subheader("📈 Variables con mayor influencia en la predicción")
                feat_imp = pd.DataFrame({
                    'Variable': model_columns,
                    'Importancia': model.feature_importances_
                }).sort_values(by='Importancia', ascending=False).head(10)
                fig_bar = px.bar(feat_imp, x='Importancia', y='Variable', orientation='h',
                                 color='Importancia', color_continuous_scale='Peach')
                st.plotly_chart(fig_bar, use_container_width=True)

            # DESCARGA
            st.download_button(
                "⬇️ Descargar resultados (Excel)",
                data=export_results_to_excel(df),
                file_name="predicciones_resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ============================
# Inicio
# ============================
if __name__ == "__main__":
    main()
















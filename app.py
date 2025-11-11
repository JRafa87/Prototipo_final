import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

# ==========================
# 1. Cargar modelo y artefactos
# ==========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/xgboost_model.pkl')
        categorical_mapping = joblib.load('models/categorical_mapping.pkl')
        scaler = joblib.load('models/scaler.pkl')
        df_reference = pd.read_csv('data/reference_data.csv')
        df_reference['Attrition'] = df_reference['Attrition'].replace({'Yes': 1, 'No': 0})
        return model, categorical_mapping, scaler, df_reference
    except Exception as e:
        st.error(f"Error al cargar modelo o artefactos: {e}")
        return None, None, None, None


# ============================
# 2. Preprocesamiento
# ============================
def preprocess_data(df, model_columns, categorical_mapping, scaler):
    df_processed = df.copy()
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
            df_processed[col] = df_processed[col].fillna(-1)

    try:
        df_processed[model_columns] = scaler.transform(df_processed[model_columns])
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}")
        return None
    return df_processed[model_columns]


# ============================
# 3. Recomendaciones
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
        recomendaciones.append("Fomentar la transparencia y confianza.")
    if row.get('NumeroTardanzas', 0) > 3 or row.get('NumeroFaltas', 0) > 1:
        recomendaciones.append("Revisar causas de ausentismo y ofrecer apoyo.")
    if not recomendaciones:
        recomendaciones.append("Sin alertas relevantes. Mantener seguimiento preventivo.")
    return " ".join(recomendaciones)


# ============================
# 4. Exportar resultados
# ============================
def export_results_to_excel(df, filename="predicciones_resultados.xlsx"):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Predicciones', index=False)
    return filename


# ============================
# 5. Interfaz principal
# ============================
def main():
    st.set_page_config(page_title="Predicción de Renuncia", layout="wide")
    st.title("📊 Modelo de Predicción de Renuncia de Empleados")

    model, categorical_mapping, scaler, df_reference = load_model()
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
        st.info(f"✅ Archivo cargado correctamente. Filas: {len(df)}")
        st.dataframe(df.head())

        processed_df = preprocess_data(df.drop(columns=['Attrition'], errors='ignore'), model_columns, categorical_mapping, scaler)
        if processed_df is None:
            return

        if st.button("🚀 Ejecutar Predicción"):
            prob = model.predict_proba(processed_df)[:, 1]
            df['Probabilidad_Renuncia'] = prob
            df['Prediction_Renuncia'] = (prob > 0.5).astype(int)
            df['Recomendacion'] = df.apply(generar_recomendacion_personalizada, axis=1)
            st.session_state.df_resultados = df
            st.session_state.show_modal = False

    # ===============================
    # Mostrar resultados
    # ===============================
    if "df_resultados" in st.session_state:
        df = st.session_state.df_resultados

        # ==== Gráficos ====
        st.subheader("📊 Análisis general por departamento")
        col1, col2 = st.columns(2)

        dept_avg = df.groupby('Department')['Probabilidad_Renuncia'].mean().reset_index()

        with col1:
            fig_bar = px.bar(
                dept_avg, x='Department', y='Probabilidad_Renuncia',
                text_auto='.2%', color='Probabilidad_Renuncia',
                color_continuous_scale=['#6DD47E', '#FFD966', '#E57373'],
                title="Probabilidad promedio por Departamento"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_pie = px.pie(
                dept_avg, names='Department', values='Probabilidad_Renuncia',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4, title="Distribución general por Departamento"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ==== Tabla ====
        st.subheader("👥 Empleados con mayor probabilidad de renuncia")

        def color_prob(val):
            if val > 0.61:
                return 'background-color: #F28B82; color:black;'
            elif 0.50 <= val <= 0.60:
                return 'background-color: #FFF176; color:black;'
            else:
                return 'background-color: #C8E6C9; color:black;'

        df_top10 = df.sort_values('Probabilidad_Renuncia', ascending=False).head(10)

        for i, row in df_top10.iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.5, 1.8, 1.5, 1, 1])
            with col1:
                st.write(f"**{row['EmployeeNumber']}**")
            with col2:
                st.write(row['Department'])
            with col3:
                st.write(row['JobRole'])
            with col4:
                st.write(f"S/. {row['MonthlyIncome']}")
            with col5:
                color = color_prob(row['Probabilidad_Renuncia'])
                st.markdown(f"<div style='{color}; text-align:center; border-radius:8px; padding:4px;'>{row['Probabilidad_Renuncia']:.2%}</div>", unsafe_allow_html=True)
            with col6:
                if st.button("👁️ Ver", key=f"ver_{i}"):
                    st.session_state.modal = {"id": row["EmployeeNumber"], "texto": row["Recomendacion"]}
                    st.session_state.show_modal = True

        # ==== MODAL FUNCIONAL ====
        if st.session_state.get("show_modal", False):
            modal = st.session_state.modal
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                        background-color: rgba(0, 0, 0, 0.6);
                        display: flex; justify-content: center; align-items: center;
                        z-index: 1000;">
                        <div style="
                            background-color: white;
                            padding: 30px;
                            border-radius: 12px;
                            width: 50%;
                            box-shadow: 0 4px 25px rgba(0,0,0,0.3);
                            text-align: left;">
                            <h4>💬 Recomendaciones para el empleado {modal["id"]}</h4>
                            <p style='font-size:15px; text-align:justify;'>{modal["texto"]}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if st.button("❌ Cerrar ventana"):
                st.session_state.show_modal = False
                st.experimental_rerun()

        # ==== DESCARGA ====
        st.subheader("📥 Descargar resultados")
        output_filename = export_results_to_excel(df)
        with open(output_filename, "rb") as file:
            st.download_button(
                label="⬇️ Descargar resultados en Excel",
                data=file,
                file_name="predicciones_resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ============================
# Ejecutar
# ============================
if __name__ == "__main__":
    main()

























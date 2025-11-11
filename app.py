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

        st.success("✅ Modelo y artefactos cargados correctamente.")
        st.toast("Modelo cargado correctamente 🎯", icon="✅")
        return model, categorical_mapping, scaler
    except Exception as e:
        st.error(f"❌ Error al cargar modelo o artefactos: {e}")
        return None, None, None


# ==========================
# 2. Preprocesamiento
# ==========================
def preprocess_data(df, model_columns, categorical_mapping, scaler):
    df_processed = df.copy()

    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    cols_to_fill = list(set(numeric_cols) & set(model_columns))
    if cols_to_fill:
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
        present_model_columns = [c for c in model_columns if c in df_processed.columns]
        df_processed[present_model_columns] = scaler.transform(df_processed[present_model_columns])
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}")
        return None

    return df_processed[model_columns]


# ==========================
# 3. Recomendaciones
# ==========================
def generar_recomendacion_personalizada(row):
    recomendaciones = []
    if row.get('IntencionPermanencia', 3) <= 2:
        recomendaciones.append("Reforzar desarrollo profesional.")
    if row.get('CargaLaboralPercibida', 3) >= 4:
        recomendaciones.append("Revisar carga laboral.")
    if row.get('SatisfaccionSalarial', 3) <= 2:
        recomendaciones.append("Evaluar ajustes salariales.")
    if row.get('ConfianzaEmpresa', 3) <= 2:
        recomendaciones.append("Fomentar la confianza y comunicación.")
    if row.get('NumeroTardanzas', 0) > 3 or row.get('NumeroFaltas', 0) > 1:
        recomendaciones.append("Analizar causas de ausentismo.")
    if not recomendaciones:
        recomendaciones.append("Sin alertas relevantes. Seguimiento preventivo.")
    return " | ".join(recomendaciones)


# ==========================
# 4. Exportar resultados
# ==========================
@st.cache_data
def export_results_to_excel(df):
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Predicciones', index=False)
    return output.getvalue()


# ==========================
# 5. Interfaz principal
# ==========================
def main():
    st.set_page_config(page_title="Predicción de Renuncia", layout="wide")
    st.markdown("<h1 style='text-align:center; color:#1f77b4;'>📊 Modelo de Predicción de Renuncia</h1>", unsafe_allow_html=True)

    model, categorical_mapping, scaler = load_model()
    if not model:
        return

    model_columns = [
        'Age','BusinessTravel','DailyRate','Department','DistanceFromHome',
        'Education','EducationField','EnvironmentSatisfaction','Gender','HourlyRate',
        'JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus',
        'MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike',
        'PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
        'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
        'YearsSinceLastPromotion','YearsWithCurrManager',
        'IntencionPermanencia','CargaLaboralPercibida','SatisfaccionSalarial',
        'ConfianzaEmpresa','NumeroTardanzas','NumeroFaltas'
    ]

    uploaded_file = st.file_uploader("📂 Sube tu archivo CSV o Excel", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.info(f"✅ Archivo cargado correctamente. Filas: **{len(df)}** | Columnas: **{df.shape[1]}**")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error al leer archivo: {e}")
            return

        if st.button("🚀 Ejecutar Predicción", use_container_width=True):
            processed = preprocess_data(df.drop(columns=['Attrition'], errors='ignore'), model_columns, categorical_mapping, scaler)
            if processed is None:
                return
            prob = model.predict_proba(processed)[:, 1]
            df['Probabilidad_Renuncia'] = prob
            df['Prediction_Renuncia'] = (prob > 0.5).astype(int)
            df['Recomendacion'] = df.apply(generar_recomendacion_personalizada, axis=1)
            st.session_state.df_resultados = df

    if "df_resultados" in st.session_state:
        df = st.session_state.df_resultados

        # === ALERTA general ===
        total_altos = (df["Probabilidad_Renuncia"] > 0.5).sum()
        if total_altos > 0:
            st.error(f"🔴 **ALERTA:** {total_altos} empleados ({total_altos/len(df):.1%}) tienen probabilidad > 50%.")
        else:
            st.success("🟢 Ningún empleado supera el 50% de probabilidad de renuncia.")

        # === TABLA PRINCIPAL ===
        st.subheader("👥 Top 10 empleados con mayor probabilidad de renuncia")
        st.markdown("---")

        def color_prob(val):
            if val >= 0.5:
                return 'background-color:#FFCDD2; color:black; font-weight:bold;'
            elif 0.4 <= val < 0.5:
                return 'background-color:#FFF59D; color:black;'
            else:
                return 'background-color:#C8E6C9; color:black;'

        df_top10 = df.sort_values('Probabilidad_Renuncia', ascending=False).head(10)
        for i, row in df_top10.iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.5, 1.8, 1.5, 1, 1])
            with col1: st.write(f"**{row['EmployeeNumber']}**")
            with col2: st.write(row['Department'])
            with col3: st.write(row['JobRole'])
            with col4: st.write(f"S/. {row['MonthlyIncome']:,.2f}")
            with col5:
                st.markdown(f"<div style='{color_prob(row['Probabilidad_Renuncia'])}; text-align:center; border-radius:8px; padding:4px;'>{row['Probabilidad_Renuncia']:.1%}</div>", unsafe_allow_html=True)
            with col6:
                with st.popover("👁️ Ver detalles", use_container_width=True):
                    st.markdown(f"**Empleado {row['EmployeeNumber']}** — Probabilidad: {row['Probabilidad_Renuncia']:.1%}")
                    st.markdown("### 🧭 Recomendaciones:")
                    recs = row["Recomendacion"].split(" | ")
                    st.markdown("<ul>" + "".join(f"<li>{r}</li>" for r in recs) + "</ul>", unsafe_allow_html=True)

        # === GRÁFICOS ===
        st.subheader("📊 Análisis por Departamento")
        dept_avg = df.groupby('Department')['Probabilidad_Renuncia'].mean().reset_index()

        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(dept_avg, x='Department', y='Probabilidad_Renuncia',
                             color='Probabilidad_Renuncia', text_auto='.1%',
                             color_continuous_scale=['#8BC34A','#FFEB3B','#E57373'],
                             title="Probabilidad Promedio por Departamento")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_pie = px.pie(dept_avg, names='Department', values='Probabilidad_Renuncia',
                             hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold,
                             title="Distribución de Probabilidades por Departamento")
            st.plotly_chart(fig_pie, use_container_width=True)

        # === DESCARGA ===
        st.subheader("📥 Descargar Resultados")
        excel_data = export_results_to_excel(df)
        st.download_button(
            label="⬇️ Descargar reporte Excel",
            data=excel_data,
            file_name="predicciones_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


if __name__ == "__main__":
    main()































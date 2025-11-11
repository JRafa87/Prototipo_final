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
        return model, categorical_mapping, scaler
    except Exception as e:
        st.error(f"❌ Error al cargar modelo o artefactos: {e}")
        return None, None, None


# ==========================
# 2. Preprocesamiento
# ==========================
def preprocess_data(df, model_columns, categorical_mapping, scaler):
    df_processed = df.copy()

    # Asegurar que todas las columnas estén presentes
    for col in model_columns:
        if col not in df_processed.columns:
            df_processed[col] = np.nan

    # Normalización y mapeo de variables categóricas
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

    # Rellenar numéricos y escalar
    numeric_cols = [col for col in model_columns if col not in categorical_cols]
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0)

    try:
        df_processed[model_columns] = scaler.transform(df_processed[model_columns])
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

    # ==========================
    # 🔹 Pestañas principales
    # ==========================
    tab1, tab2 = st.tabs(["📂 Predicción desde archivo", "🧮 Simulación manual"])

    # === TAB 1: ARCHIVO ===
    with tab1:
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

            total_altos = (df["Probabilidad_Renuncia"] > 0.5).sum()
            if total_altos > 0:
                st.error(f"🔴 **ALERTA:** {total_altos} empleados ({total_altos/len(df):.1%}) tienen probabilidad > 50%.")
            else:
                st.success("🟢 Ningún empleado supera el 50% de probabilidad de renuncia.")

            st.subheader("👥 Top 10 empleados con mayor probabilidad de renuncia")
            st.markdown("---")

            df_top10 = df.sort_values('Probabilidad_Renuncia', ascending=False).head(10)
            for i, row in df_top10.iterrows():
                col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.5, 1.8, 1.5, 1, 1])
                with col1: st.write(f"**{row['EmployeeNumber']}**")
                with col2: st.write(row['Department'])
                with col3: st.write(row['JobRole'])
                with col4: st.write(f"S/. {row['MonthlyIncome']:,.2f}")
                with col5:
                    color = "#FFCDD2" if row['Probabilidad_Renuncia'] >= 0.5 else "#C8E6C9"
                    st.markdown(
                        f"<div style='background-color:{color}; text-align:center; border-radius:8px; padding:4px;'>{row['Probabilidad_Renuncia']:.1%}</div>",
                        unsafe_allow_html=True)
                with col6:
                    with st.popover("🔍 Ver"):
                        st.markdown("### 🧭 Recomendaciones")
                        for rec in row["Recomendacion"].split(" | "):
                            st.write(f"- {rec}")

    # === TAB 2: SIMULACIÓN MANUAL ===
    with tab2:
        st.subheader("🧮 Simulación manual de un empleado")
        st.write("Completa los campos para predecir la probabilidad de renuncia de un empleado específico:")

        col1, col2 = st.columns(2)
        with col1:
            Age = st.number_input("Edad", 18, 65, 30)
            Gender = st.selectbox("Género", ["M", "F"])
            Department = st.selectbox("Departamento", ["VENTAS", "RRHH", "TECNOLOGÍA", "FINANZAS"])
            JobRole = st.text_input("Puesto de trabajo", "Analista")
            MonthlyIncome = st.number_input("Ingreso mensual (S/.)", 1000, 20000, 3500)
            BusinessTravel = st.selectbox("Frecuencia de viaje", ["NUNCA", "RARA VEZ", "FRECUENTE"])
            OverTime = st.selectbox("¿Hace horas extra?", ["SI", "NO"])
        with col2:
            IntencionPermanencia = st.slider("Intención de permanencia (1-5)", 1, 5, 3)
            CargaLaboralPercibida = st.slider("Carga laboral percibida (1-5)", 1, 5, 3)
            SatisfaccionSalarial = st.slider("Satisfacción salarial (1-5)", 1, 5, 3)
            ConfianzaEmpresa = st.slider("Confianza en la empresa (1-5)", 1, 5, 3)
            NumeroTardanzas = st.number_input("Número de tardanzas", 0, 20, 0)
            NumeroFaltas = st.number_input("Número de faltas", 0, 10, 0)

        if st.button("🔮 Predecir", use_container_width=True):
            # Crear registro con todas las columnas del modelo
            input_data = {col: np.nan for col in model_columns}
            input_data.update({
                'Age': Age,
                'Gender': Gender,
                'Department': Department,
                'JobRole': JobRole,
                'MonthlyIncome': MonthlyIncome,
                'BusinessTravel': BusinessTravel,
                'OverTime': OverTime,
                'IntencionPermanencia': IntencionPermanencia,
                'CargaLaboralPercibida': CargaLaboralPercibida,
                'SatisfaccionSalarial': SatisfaccionSalarial,
                'ConfianzaEmpresa': ConfianzaEmpresa,
                'NumeroTardanzas': NumeroTardanzas,
                'NumeroFaltas': NumeroFaltas
            })

            df_input = pd.DataFrame([input_data])
            processed_input = preprocess_data(df_input, model_columns, categorical_mapping, scaler)

            if processed_input is not None:
                prob = model.predict_proba(processed_input)[:, 1][0]
                recomendacion = generar_recomendacion_personalizada(df_input.iloc[0])

                st.markdown(f"""
                    <div style='background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;'>
                        <h3>🔎 Resultado de la simulación</h3>
                        <p style='font-size:20px;'>Probabilidad de renuncia: 
                        <b style='color:{"red" if prob>0.5 else "green"}'>{prob:.1%}</b></p>
                        <p><b>Recomendación:</b> {recomendacion}</p>
                    </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()



































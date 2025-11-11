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
# 2. Preprocesamiento
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
        recomendaciones.append("Fomentar la transparencia y la confianza.")
    if row.get('NumeroTardanzas', 0) > 3 or row.get('NumeroFaltas', 0) > 1:
        recomendaciones.append("Revisar causas de ausentismo y ofrecer apoyo.")

    if not recomendaciones:
        recomendaciones.append("Sin alertas relevantes. Mantener seguimiento preventivo.")

    return " ".join(recomendaciones)


# ============================
# 4. Exportar resultados
# ============================
def export_results_to_excel(df, filename="predicciones_resultados.xlsx"):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as output:
        df.to_excel(output, sheet_name='Predicciones', index=False)
    return filename


# ============================
# 5. Interfaz Streamlit
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

            # SEMAFORIZACIÓN por umbrales nuevos
            def color_prob(val):
                if val > 0.50:
                    return 'background-color: #F28B82; color:black;'  # rojo suave
                elif 0.40 <= val <= 0.50:
                    return 'background-color: #FFF4A3; color:black;'  # amarillo pastel
                else:
                    return 'background-color: #B7E1CD; color:black;'  # verde suave

            st.subheader("👥 Top 10 empleados con mayor probabilidad de renuncia")
            df_top10 = df.sort_values('Probabilidad_Renuncia', ascending=False).head(10).copy()

            # Estado del empleado seleccionado
            if "empleado_sel" not in st.session_state:
                st.session_state.empleado_sel = None

            for i, row in df_top10.iterrows():
                col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.2, 1.5, 1.2, 1, 1])
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
                    if st.button(f"👁️ Ver", key=f"ver_{i}"):
                        st.session_state.empleado_sel = i

            # Mostrar ventana modal si hay selección
            if st.session_state.empleado_sel is not None:
                emp = df_top10.loc[st.session_state.empleado_sel]
                st.markdown(f"""
                    <div style="
                        background-color:white;
                        border:2px solid #ccc;
                        border-radius:12px;
                        padding:20px;
                        position:fixed;
                        top:20%;
                        left:30%;
                        width:40%;
                        box-shadow:0 0 20px rgba(0,0,0,0.3);
                        z-index:9999;">
                        <h4>💬 Recomendaciones para el empleado {emp['EmployeeNumber']}</h4>
                        <p style='font-size:15px;'>{emp['Recomendacion']}</p>
                        <br>
                        <form action="" method="get">
                            <input type="submit" value="Cerrar" style="
                                background-color:#4CAF50;
                                color:white;
                                padding:8px 16px;
                                border:none;
                                border-radius:6px;
                                cursor:pointer;">
                        </form>
                    </div>
                """, unsafe_allow_html=True)

            # GRÁFICOS
            st.subheader("📊 Análisis por Departamento y Variables Clave")
            col1, col2 = st.columns(2)
            with col1:
                dept_avg = df.groupby('Department')['Probabilidad_Renuncia'].mean().reset_index()
                fig_bar = px.bar(dept_avg, x='Department', y='Probabilidad_Renuncia',
                                 text_auto='.2%', color='Probabilidad_Renuncia',
                                 color_continuous_scale='Reds',
                                 title="Probabilidad promedio por departamento")
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                fig_pie = px.pie(dept_avg, names='Department', values='Probabilidad_Renuncia',
                                 color_discrete_sequence=px.colors.qualitative.Pastel,
                                 hole=0.4, title="Distribución total por departamento")
                st.plotly_chart(fig_pie, use_container_width=True)

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





















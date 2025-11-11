import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import os

# ==========================
# 1. Cargar modelo y artefactos
# ==========================
@st.cache_resource
def load_model():
    try:
        # Nota: Asumiendo que estos archivos existen en el entorno de ejecución
        # En un entorno real, estos archivos deben estar disponibles.
        model = joblib.load('models/xgboost_model.pkl')
        categorical_mapping = joblib.load('models/categorical_mapping.pkl')
        scaler = joblib.load('models/scaler.pkl')
        # La línea de carga de df_reference no se usa en la lógica de UI/predicción
        # df_reference = pd.read_csv('data/reference_data.csv')
        # df_reference['Attrition'] = df_reference['Attrition'].replace({'Yes': 1, 'No': 0})
        st.success("✅ Modelo y artefactos cargados correctamente.")
        # Retornamos solo lo necesario para la predicción
        return model, categorical_mapping, scaler
    except Exception as e:
        st.error(f"Error al cargar modelo o artefactos: Asegúrate de que los archivos 'xgboost_model.pkl', 'categorical_mapping.pkl' y 'scaler.pkl' están en la carpeta 'models/'. Error: {e}")
        return None, None, None


# ============================
# 2. Preprocesamiento
# ============================
def preprocess_data(df, model_columns, categorical_mapping, scaler):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    
    # Rellenar solo las columnas numéricas relevantes para el modelo que falten
    cols_to_fill = list(set(numeric_cols) & set(model_columns))
    if cols_to_fill:
        # Calcular la media solo de las columnas a rellenar
        mean_values = df_processed[cols_to_fill].mean()
        df_processed[cols_to_fill] = df_processed[cols_to_fill].fillna(mean_values)

    categorical_cols = [
        'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
        'MaritalStatus', 'OverTime'
    ]
    for col in categorical_cols:
        if col in df_processed.columns:
            # Limpieza y mapeo de categóricas
            df_processed[col] = df_processed[col].astype(str).str.strip().str.upper()
            if col in categorical_mapping:
                df_processed[col] = df_processed[col].map(categorical_mapping[col])
            # Rellenar valores no mapeados con -1 (o el valor usado en el entrenamiento)
            df_processed[col] = df_processed[col].fillna(-1)

    try:
        # Asegurar que todas las columnas necesarias están presentes para el escalado
        # Si faltan columnas, Streamlit mostrará un error.
        
        # Filtrar solo las columnas presentes en el DataFrame para evitar errores
        present_model_columns = [col for col in model_columns if col in df_processed.columns]

        if not present_model_columns:
            st.error("Error: El archivo de datos no contiene ninguna de las columnas esperadas por el modelo.")
            return None

        # Asegurar que las columnas están en el orden esperado si es crítico para el modelo
        # (Aunque el escalador no debería ser sensible al orden si solo se transforman las columnas)
        df_to_scale = df_processed[present_model_columns] 
        
        # Escalar los datos
        df_processed[present_model_columns] = scaler.transform(df_to_scale)
        
    except Exception as e:
        st.error(f"Error al escalar los datos o al alinear columnas. Asegúrate de que el CSV contiene las columnas esperadas. Detalle: {e}")
        return None
        
    # Devolver solo las columnas procesadas que el modelo espera
    return df_processed[model_columns]


# ============================
# 3. Recomendaciones
# ============================
def generar_recomendacion_personalizada(row):
    # Se extraen los valores usando .get() con un valor por defecto que no activa la alerta (e.g., 3)
    recomendaciones = []
    # Si la intención de permanencia (escala 1-5, donde 1 es baja) es baja
    if row.get('IntencionPermanencia', 3) <= 2:
        recomendaciones.append("Reforzar conversaciones de desarrollo profesional y revisar oportunidades de crecimiento.")
    # Si la carga laboral percibida (escala 1-5, donde 5 es alta) es alta
    if row.get('CargaLaboralPercibida', 3) >= 4:
        recomendaciones.append("Revisar la carga laboral y redistribuir tareas o asignar recursos de apoyo.")
    # Si la satisfacción salarial (escala 1-5, donde 1 es baja) es baja
    if row.get('SatisfaccionSalarial', 3) <= 2:
        recomendaciones.append("Evaluar ajustes salariales o beneficios, y asegurar la competitividad de la compensación.")
    # Si la confianza en la empresa (escala 1-5, donde 1 es baja) es baja
    if row.get('ConfianzaEmpresa', 3) <= 2:
        recomendaciones.append("Fomentar la transparencia, la comunicación de liderazgo y la confianza en las decisiones.")
    # Revisión de ausentismo (valores inventados, ajusta según tu data real)
    if row.get('NumeroTardanzas', 0) > 3 or row.get('NumeroFaltas', 0) > 1:
        recomendaciones.append("Revisar causas de ausentismo y ofrecer apoyo o planes de acción.")
    
    # Indicadores de modelo base que a menudo son importantes
    if row.get('OverTime', 'No') == 'Yes':
        recomendaciones.append("El tiempo extra es un factor de riesgo. Revisar la gestión de horarios y equilibrio vida-trabajo.")
    if row.get('JobSatisfaction', 3) <= 2:
        recomendaciones.append("Indagación sobre el nivel de satisfacción con el puesto actual (factores intrínsecos).")


    if not recomendaciones:
        recomendaciones.append("Sin alertas inmediatas relevantes. Mantener seguimiento preventivo y motivacional.")
        
    return " | ".join(recomendaciones)


# ============================
# 4. Exportar resultados
# ============================
@st.cache_data
def export_results_to_excel(df):
    output = pd.io.common.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Predicciones', index=False)
    processed_data = output.getvalue()
    return processed_data


# ============================
# 5. Interfaz principal
# ============================
def main():
    st.set_page_config(page_title="Predicción de Renuncia", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>📊 Modelo de Predicción de Renuncia de Empleados</h1>", unsafe_allow_html=True)

    # Solo cargar el modelo, mapeo y scaler (df_reference no se usa en esta app)
    model, categorical_mapping, scaler = load_model()
    if model is None or categorical_mapping is None or scaler is None:
        return

    # Definir las columnas esperadas por el modelo (incluyendo las nuevas variables de la encuesta)
    model_columns = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
        'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager',
        # Variables de encuesta (asumiendo que son las últimas columnas procesadas)
        'IntencionPermanencia', 'CargaLaboralPercibida', 'SatisfaccionSalarial',
        'ConfianzaEmpresa', 'NumeroTardanzas', 'NumeroFaltas'
    ]

    # --- Carga de Archivo ---
    uploaded_file = st.file_uploader("📂 Sube un archivo CSV o Excel con los datos de los empleados", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            # Cargar el DataFrame
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.info(f"✅ Archivo cargado correctamente. Filas: **{len(df)}** | Columnas: **{df.shape[1]}**")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return
            
        # --- Preprocesamiento y Predicción ---
        if st.button("🚀 Ejecutar Predicción y Análisis", use_container_width=True):
            # Preprocesar sin la columna Attrition si existe
            processed_df = preprocess_data(df.drop(columns=['Attrition'], errors='ignore'), model_columns, categorical_mapping, scaler)
            
            if processed_df is None:
                st.warning("No se pudo completar el preprocesamiento. Revisa los mensajes de error anteriores.")
                return

            # Ejecutar Predicción
            prob = model.predict_proba(processed_df)[:, 1]
            df['Probabilidad_Renuncia'] = prob
            df['Prediction_Renuncia'] = (prob > 0.5).astype(int)
            df['Recomendacion'] = df.apply(generar_recomendacion_personalizada, axis=1)
            
            # Guardar resultados en el estado de sesión
            st.session_state.df_resultados = df

    # ===============================
    # Mostrar resultados
    # ===============================
    if "df_resultados" in st.session_state:
        df = st.session_state.df_resultados

        st.markdown("---")

        # ==== ALERTA general ====
        total_altos = (df["Probabilidad_Renuncia"] > 0.5).sum()
        if total_altos > 0:
            st.error(f"🔴 **¡ALERTA!** {total_altos} empleados ({total_altos/len(df):.1%}) presentan una probabilidad de renuncia superior al 50%.")
        else:
            st.success("🟢 No hay empleados con probabilidad de renuncia superior al 50%.")


        # ==== Tabla de TOP Empleados ====
        st.subheader("👥 Empleados con mayor probabilidad de renuncia (Top 10)")
        st.markdown("---")

        # Encabezados de la tabla (usando columnas de Streamlit)
        col1_h, col2_h, col3_h, col4_h, col5_h, col6_h = st.columns([1.2, 1.5, 1.8, 1.5, 1, 1])
        col1_h.markdown("**ID Empleado**")
        col2_h.markdown("**Departamento**")
        col3_h.markdown("**Rol**")
        col4_h.markdown("**Ingreso Mensual**")
        col5_h.markdown("**Probabilidad**")
        col6_h.markdown("**Acción**")


        def color_prob(val):
            if val >= 0.5:
                return 'background-color: #F28B82; color:black; font-weight:bold;' # Rojo
            elif 0.4 <= val < 0.5:
                return 'background-color: #FFF59D; color:black;' # Amarillo
            else:
                return 'background-color: #C8E6C9; color:black;' # Verde

        df_top10 = df.sort_values('Probabilidad_Renuncia', ascending=False).head(10)

        for i, row in df_top10.iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.5, 1.8, 1.5, 1, 1])
            
            # --- Fila de la tabla ---
            with col1:
                st.write(f"**{row['EmployeeNumber']}**")
            with col2:
                st.write(row['Department'])
            with col3:
                st.write(row['JobRole'])
            with col4:
                # Usar formato de moneda local con dos decimales
                st.write(f"S/. {row['MonthlyIncome']:,.2f}")
            with col5:
                color_style = color_prob(row['Probabilidad_Renuncia'])
                st.markdown(f"<div style='{color_style}; text-align:center; border-radius:8px; padding:4px; margin:0;'>{row['Probabilidad_Renuncia']:.2%}</div>", unsafe_allow_html=True)
            with col6:
                # Usamos st.popover para mostrar los detalles y recomendaciones. Es nativo y se cierra solo al hacer clic fuera.
                with st.popover("👁️ Ver", use_container_width=True):
                    st.markdown(f"**Recomendaciones para el empleado {row['EmployeeNumber']}**", help=f"Probabilidad: {row['Probabilidad_Renuncia']:.2%}")
                    
                    # Dividir las recomendaciones en una lista para mejor lectura
                    recs = row["Recomendacion"].split(" | ")
                    st.markdown("<ul>" + "".join(f"<li>{rec}</li>" for rec in recs) + "</ul>", unsafe_allow_html=True)

        st.markdown("---") # Separador visual

        # ==== Gráficos (MOVIDOS DEBAJO DE LA TABLA) ====
        st.subheader("📊 Análisis general por departamento")
        col1, col2 = st.columns(2)

        dept_avg = df.groupby('Department')['Probabilidad_Renuncia'].mean().reset_index()

        with col1:
            fig_bar = px.bar(
                dept_avg, x='Department', y='Probabilidad_Renuncia',
                text_auto='.2%', color='Probabilidad_Renuncia',
                color_continuous_scale=['#8BC34A', '#FFEB3B', '#E57373'], # Verde a Rojo
                title="Probabilidad promedio por Departamento"
            )
            fig_bar.update_layout(xaxis_title="", yaxis_title="Probabilidad Promedio", title_x=0.5)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_pie = px.pie(
                dept_avg, names='Department', values='Probabilidad_Renuncia',
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.4, title="Distribución de la Probabilidad de Renuncia por Departamento"
            )
            fig_pie.update_layout(title_x=0.5)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---") # Separador visual

        # ==== DESCARGA ====
        st.subheader("📥 Descargar resultados")
        excel_data = export_results_to_excel(df)
        
        st.download_button(
            label="⬇️ Descargar informe completo en Excel",
            data=excel_data,
            file_name="predicciones_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.markdown("---")


if __name__ == "__main__":
    main()






























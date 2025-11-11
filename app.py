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
# Funciones de Recomendaciones
# ============================
def generar_recomendacion_personalizada(row):
    recomendaciones = []

    if row.get('IntencionPermanencia', 3) <= 2:
        recomendaciones.append("💬 Reforzar conversaciones de desarrollo profesional para mejorar la intención de permanencia.")
    if row.get('CargaLaboralPercibida', 3) >= 4:
        recomendaciones.append("🕒 Revisar la carga laboral percibida y redistribuir tareas si es necesario.")
    if row.get('SatisfaccionSalarial', 3) <= 2:
        recomendaciones.append("💰 Evaluar ajustes salariales o beneficios complementarios.")
    if row.get('ConfianzaEmpresa', 3) <= 2:
        recomendaciones.append("🤝 Implementar acciones de transparencia y fortalecimiento de la confianza.")
    if row.get('NumeroTardanzas', 0) > 3 or row.get('NumeroFaltas', 0) > 1:
        recomendaciones.append("📅 Reunirse para entender causas de ausentismo y ofrecer apoyo personalizado.")

    if not recomendaciones:
        recomendaciones.append("✅ No se detectan alertas críticas adicionales. Mantener seguimiento preventivo.")

    return "\n".join(recomendaciones)


def display_recommendations(df_results):
    st.markdown("---")
    st.header("💡 Recomendaciones Estratégicas")

    high_risk_threshold = 0.70
    df_high_risk = df_results[df_results['Probabilidad_Renuncia'] > high_risk_threshold]
    
    if df_high_risk.empty:
        st.success("El riesgo de deserción es bajo. Mantenga las políticas actuales.")
        return

    st.subheader("1. Intervención Individual Prioritaria")
    top_risk_count = min(10, len(df_high_risk))
    st.markdown(f"**Enfocarse en los {top_risk_count} empleados con mayor probabilidad de renuncia** (>{high_risk_threshold*100:.0f}%).")
    st.info("Realizar entrevistas de retención confidenciales para conocer sus motivaciones y preocupaciones.")

    if 'Department' in df_high_risk.columns:
        st.subheader("2. Foco Departamental")
        risk_by_dept = df_high_risk.groupby('Department').size().sort_values(ascending=False)
        top_risk_dept = risk_by_dept.index[0]
        st.warning(f"El departamento de **{top_risk_dept}** tiene la mayor concentración de empleados en riesgo alto.")

    st.subheader("3. Estrategia Global de Prevención")
    if 'MonthlyIncome' in df_high_risk.columns:
        avg_high_risk_income = df_high_risk['MonthlyIncome'].mean()
        avg_total_income = df_results['MonthlyIncome'].mean()
        if avg_high_risk_income < avg_total_income * 0.9:
            st.info("💰 Revisar las bandas salariales: los empleados de mayor riesgo tienen ingresos promedio más bajos.")


# ============================
# Funciones de Exportación
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
    st.markdown("Carga tu archivo de datos para obtener predicciones.")

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

    uploaded_file = st.file_uploader("Sube un archivo CSV o Excel (.csv, .xlsx) para PREDICCIÓN", type=["csv", "xlsx"])
    
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
            st.error("No se puede continuar con la predicción debido a un error de preprocesamiento.")
            return 

        st.header("1. Predicción con Datos Cargados")
        
        if st.button("🚀 Ejecutar Predicción"):
            st.info("Ejecutando el modelo sobre los datos cargados...")

            probabilidad_renuncia = model.predict_proba(processed_df)[:, 1]
            predictions = (probabilidad_renuncia > 0.5).astype(int)
            
            df_original['Prediction_Renuncia'] = predictions
            df_original['Probabilidad_Renuncia'] = probabilidad_renuncia

            st.success("✅ Predicción completada correctamente")

            # ========================
            # 🔹 Análisis de Alto Riesgo
            # ========================
            high_risk_threshold = 0.70
            df_high_risk = df_original[df_original['Probabilidad_Renuncia'] > high_risk_threshold]
            count_high_risk = len(df_high_risk)

            if count_high_risk > 0:
                st.warning(f"⚠️ Se detectaron **{count_high_risk} empleados** con alta probabilidad de renuncia (>70%).")
            else:
                st.success("🎉 No se detectaron empleados con alto riesgo de renuncia.")

            # ========================
            # 🔹 Tabla de Top 10 empleados con mayor riesgo
            # ========================
            st.subheader("👥 Top 10 empleados con mayor probabilidad de renuncia")

            df_top10 = df_original.sort_values(by='Probabilidad_Renuncia', ascending=False).head(10).copy()

            def color_prob(val):
                if val >= 0.8:
                    color = 'background-color: #ff4d4d; color: white;'
                elif val >= 0.6:
                    color = 'background-color: #ffcc00; color: black;'
                else:
                    color = 'background-color: #70db70; color: black;'
                return color

            st.dataframe(
                df_top10[['EmployeeNumber', 'Department', 'JobRole', 'MonthlyIncome', 'Probabilidad_Renuncia']]
                .style.applymap(color_prob, subset=['Probabilidad_Renuncia'])
                .format({'Probabilidad_Renuncia': "{:.2%}"})
            )

            # ========================
            # 🔹 Recomendaciones individuales
            # ========================
            st.subheader("🔍 Recomendaciones Individuales (Top 10 Riesgo)")
            for _, row in df_top10.iterrows():
                with st.expander(f"👁️ Ver recomendación para empleado #{row['EmployeeNumber']} ({row['JobRole']})"):
                    st.markdown(f"**Probabilidad de renuncia:** {row['Probabilidad_Renuncia']:.2%}")
                    st.markdown(f"**Departamento:** {row['Department']}")
                    st.markdown(f"**Ingreso mensual:** ${row['MonthlyIncome']:.2f}")
                    st.markdown("---")
                    st.markdown("**Recomendación personalizada:**")
                    st.write(generar_recomendacion_personalizada(row))

            # ========================
            # 🔹 Descarga y análisis global
            # ========================
            st.download_button(
                label="⬇️ Descargar Resultados de Predicción (Excel)",
                data=export_results_to_excel(df_original),
                file_name="predicciones_resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            display_recommendations(df_original)


# ============================
#  Inicio de la Aplicación
# ============================
if __name__ == "__main__":
    main()















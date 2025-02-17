import sys
import os
import streamlit as st
import pandas as pd
import pickle
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import load_and_clean_data

st.title("Pitchers Metrics")

model_path = "models/model.pkl"

if not os.path.exists(model_path):
    st.warning("El modelo no se ha generado. Creando modelo...")
    try:
        subprocess.run(["python", "src/train.py"], check=True)
        st.success("Modelo generado exitosamente.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error al ejecutar: {e}")
        st.stop()

model = pickle.load(open(model_path, "rb"))

df = pd.read_csv("data/BateadorVsPitcher.csv")

st.header("Predicción de rendimiento de Pitchers")

bateadores = st.multiselect("Selecciona un máximo de 3 bateadores: ", df["Bateador"].unique(), max_selections=3)

if len(bateadores) > 0:
    df_filtrado = df[df["Bateador"].isin(bateadores)]
    pitchers = df_filtrado["Pitcher"].unique()

    df_filtrado = pd.get_dummies(df, columns=["Bateador", "Etapa", "Equipo", "Pitcher"], drop_first=True)

    results = []
    for pitcher in pitchers:
        columna_pitcher = f"Pitcher_{pitcher}"
        if columna_pitcher in df_filtrado.columns:
            df_test = df_filtrado[df_filtrado[columna_pitcher] == 1]
        else:
            print(f"La columna {columna_pitcher} no existe en df_filtrado")
            continue

        #Generando las columnas que el modelo va a consultar
        if hasattr(model, "feature_names_in_"):
            feature_columns = model.feature_names_in_.tolist()
        else:
            feature_columns = [col for col in df_test.columns if col not in ["Pitcher", "Bateador", "Resultado"]]

        df_test = df_test[feature_columns]

        if not df_test.empty:
            prediction = model.predict_proba(df_test)[:, 1].mean()
            results.append(("Pitcher: " + pitcher, prediction))

    if results:
        df_results = pd.DataFrame(results, columns=["Pitcher", "Probabilidad del bateador embasarse"])
        df_results = df_results.sort_values("Probabilidad del bateador embasarse", ascending=True)
        st.write(df_results)
    else:
        st.write("No hay suficientes datos para generar predicciones.")

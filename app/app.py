import sys
import os
import streamlit as st
import pandas as pd
import pickle
import subprocess
import plotly.graph_objects as go
import random
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import load_and_clean_data

# Cachear carga de datos y modelo
@st.cache_data
def cargar_datos():
    return pd.read_csv("data/BateadorVsPitcher.csv")

@st.cache_resource
def cargar_modelo():
    if os.path.exists("models/model.pkl"):
        return pickle.load(open("models/model.pkl", "rb"))
    else:
        return None

@st.cache_resource
def cargar_columnas():
    if os.path.exists("models/feature_columns.pkl"):
        return pickle.load(open("models/feature_columns.pkl", "rb"))
    else:
        return None

# Función para entrenar el modelo si no existe
def entrenar_modelo_si_no_existe():
    model_path = "models/model.pkl"
    if not os.path.exists(model_path):
        try:
            subprocess.run(["python", "src/train.py"], check=True)
            st.success("Modelo generado exitosamente.")
        except subprocess.CalledProcessError as e:
            st.error(f"Error al generar el modelo: {e}")
            st.stop()

    if 'outs' not in st.session_state:
        st.session_state.outs = 0
        st.session_state.bolas = 0
        st.session_state.strikes = 0
        st.session_state.en_primera = False
        st.session_state.en_segunda = False
        st.session_state.en_tercera = False
        st.session_state.turno = 0
        st.session_state.inning_activo = False
        st.session_state.bateadores_turno = []
        st.session_state.bateador_actual = ""

def dibujar_campo():
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.2, 1.2)

    # Bases
    ax.plot([0, 0.7], [0, 0.7], color='black')
    ax.plot([0, -0.7], [0, 0.7], color='black')
    ax.plot([-0.7, 0.7], [0.7, 0.7], color='black')

    # Bases markers
    ax.plot(0, 0, 's', markersize=20, color='white', markeredgecolor='black')  # Home
    ax.plot(0.7, 0.7, 's', markersize=20, color='white', markeredgecolor='black')  # 1B
    ax.plot(-0.7, 0.7, 's', markersize=20, color='white', markeredgecolor='black')  # 2B
    ax.plot(0, 1.0, 's', markersize=20, color='white', markeredgecolor='black')  # 3B

    # Jugadores en bases
    if st.session_state.en_primera:
        ax.plot(0.7, 0.7, 'o', markersize=20, color='blue')
    if st.session_state.en_segunda:
        ax.plot(-0.7, 0.7, 'o', markersize=20, color='blue')
    if st.session_state.en_tercera:
        ax.plot(0, 1.0, 'o', markersize=20, color='blue')

    ax.axis('off')
    st.pyplot(fig)

def simular_lanzamiento():
    evento = random.choices(
        ["strike", "bola", "hit", "out"],
        weights=[0.4, 0.3, 0.2, 0.1],
        k=1
    )[0]

    if evento == "strike":
        st.session_state.strikes += 1
        st.success("Strike!")
    elif evento == "bola":
        st.session_state.bolas += 1
        st.info("Bola!")
    elif evento == "hit":
        st.success("¡Hit!")
        if not st.session_state.en_primera:
            st.session_state.en_primera = True
        elif not st.session_state.en_segunda:
            st.session_state.en_segunda = True
        elif not st.session_state.en_tercera:
            st.session_state.en_tercera = True
        else:
            st.session_state.outs += 1
            st.warning("Out forzado por congestion de bases!")
    elif evento == "out":
        st.session_state.outs += 1
        st.warning("Out!")

    if st.session_state.strikes >= 3:
        st.session_state.outs += 1
        st.session_state.strikes = 0
        st.session_state.bolas = 0
        st.warning("Ponche!")

    if st.session_state.bolas >= 4:
        st.session_state.bolas = 0
        st.session_state.strikes = 0
        st.session_state.en_primera = True
        st.success("Boleto a primera!")


def recomendar_pitchers(bateador_actual, df, model, feature_columns):
    bateador_data = df[df["Bateador"] == bateador_actual]
    pitchers = bateador_data["Pitcher"].unique()

    df_modelo = load_and_clean_data(df, from_dataframe=True)
    recomendaciones = []

    for pitcher in pitchers:
        columna_pitcher = f"Pitcher_{pitcher}"
        if columna_pitcher in df_modelo.columns:
            df_test = df_modelo[df_modelo[columna_pitcher] == 1]
            if "Embasado" in df_test.columns:
                df_test = df_test.drop(columns=["Embasado"])
            df_test = df_test.reindex(columns=feature_columns, fill_value=0)

            if not df_test.empty:
                prediction = model.predict_proba(df_test)[:, 1].mean()
                prediction = min(max(prediction, 0.05), 0.95)
                recomendaciones.append((pitcher, prediction))

    recomendaciones = sorted(recomendaciones, key=lambda x: x[1])[:3]
    return recomendaciones

# Función para generar resumen
def generar_resumen(pitcher_name, efectividades, bateadores):
    cantidad = len(efectividades)
    promedio = sum(efectividades) / cantidad

    if promedio >= 0.75:
        nivel = "Excelente"
        consejo = "Altamente recomendado para enfrentar a estos bateadores."
    elif promedio >= 0.6:
        nivel = "Bueno"
        consejo = "Buena opción para este enfrentamiento."
    elif promedio >= 0.45:
        nivel = "Moderado"
        consejo = "Puede ser una opción viable, aunque con cierto riesgo."
    else:
        nivel = "Riesgoso"
        consejo = "No se recomienda enfrentar a estos bateadores con este pitcher."

    if cantidad == 1:
        resumen = (
            f"***Resumen del match***\n\n"
            f"El pitcher **{pitcher_name}** tiene una efectividad de **{efectividades[0]:.2f}** "
            f"frente al bateador **{bateadores[0]}**.\n\n"
            f"Nivel de confianza: **{nivel}**.\n\n"
            f"Recomendación: {consejo}"
        )
    else:
        resumen = (
            f"***Resumen del match***\n\n"
            f"El pitcher **{pitcher_name}** tiene una efectividad promedio de **{promedio:.2f}** "
            f"frente a los bateadores seleccionados: **{', '.join(bateadores)}**.\n\n"
            f"Nivel de confianza: **{nivel}**.\n\n"
            f"Recomendación: {consejo}"
        )

    return resumen

# Inicio de la app
st.sidebar.title("Navegación")
page = st.sidebar.radio("Selecciona una vista", [
    "🏠 Home",
    "⚾ Predicción de rendimiento",
    "🥎 Simulación de Partidos",
    "📊 Estadísticas por Pitcher",
    "📝 Registro"
])

#vista de inicio
if page == "🏠 Home":
    st.title("Pitchers Metrics ⚾")
    st.image("images/BG/details-ball-sport.jpg", use_container_width=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("images/BG/pitcher.png", width=150)

    with col2:
        st.write("""
        Bienvenido a Pitchers Metrics, tu asistente para analizar y predecir el desempeño de los pitchers contra bateadores específicos usando Machine Learning.

        🟢 **¿Qué puedes hacer aquí?**  
        - Predecir probabilidad de embasado.
        - Simular innings de partidos reales.
        - Ver estadísticas históricas de enfrentamientos.
        - Registrar nuevos datos para mejorar el modelo.

        ¡Optimiza tu estrategia y toma mejores decisiones en el juego!
        """)

#vista de predicciones
elif page == "⚾ Predicción de rendimiento":
    st.title("⚾ Predicción de rendimiento")

    entrenar_modelo_si_no_existe()
    model = cargar_modelo()
    feature_columns = cargar_columnas()
    df = cargar_datos()

    st.header("Predicción de rendimiento de Pitchers")

    bateadores = st.multiselect("Selecciona un máximo de 3 bateadores:", df["Bateador"].unique(), max_selections=3)

    if bateadores:
        df_filtrado = df[df["Bateador"].isin(bateadores)]
        pitchers = df_filtrado["Pitcher"].unique()

        df_modelo = load_and_clean_data(df, from_dataframe=True)

        results = []
        for pitcher in pitchers:
            columna_pitcher = f"Pitcher_{pitcher}"
            if columna_pitcher in df_modelo.columns:
                df_test = df_modelo[df_modelo[columna_pitcher] == 1]
            else:
                continue

            if "Embasado" in df_test.columns:
                df_test = df_test.drop(columns=["Embasado"])

            df_test = df_test.reindex(columns=feature_columns, fill_value=0)

            if not df_test.empty:
                if df_test.shape[0] >= 3:  # mínimo 3 datos para predecir
                    # Ajustar df_test para que tenga exactamente las columnas que espera el modelo
                    feature_names_modelo = model.feature_names_in_

                    # Agregar columnas faltantes con ceros
                    for col in feature_names_modelo:
                        if col not in df_test.columns:
                            df_test[col] = 0

                    # Eliminar columnas sobrantes que no estén en el modelo
                    df_test = df_test[feature_names_modelo]

                    prediction = model.predict_proba(df_test)[:, 1].mean()
                    prediction = min(max(prediction, 0.05), 0.95)  # limitar entre 5% y 95%
                    results.append((pitcher, prediction))
                else:
                    print(f"El pitcher {pitcher} tiene muy pocos enfrentamientos para predicción confiable.")


        if results:
            df_results = pd.DataFrame(results, columns=["Pitcher", "Probabilidad del bateador embasarse"])
            df_results["Efectividad del Pitcher (%)"] = (1 - df_results["Probabilidad del bateador embasarse"]) * 100
            df_results = df_results.sort_values("Probabilidad del bateador embasarse")

            st.write("### Resultados")
            st.dataframe(df_results.round(2))
            st.bar_chart(df_results.set_index("Pitcher")["Efectividad del Pitcher (%)"])

            # Mostrar selector de pitcher basado en resultados
            pitcher_seleccionado = st.selectbox(
                "Selecciona un pitcher para ver su resumen de efectividad:",
                options=df_results["Pitcher"].tolist()
            )

            # Filtrar el pitcher seleccionado
            efectividades = df_results[df_results["Pitcher"] == pitcher_seleccionado]["Efectividad del Pitcher (%)"].tolist()

            # Generar resumen actualizado
            resumen = generar_resumen(
                pitcher_seleccionado,
                efectividades,
                bateadores
            )
            st.markdown(resumen)

            # Guardar en session_state por si quieres usar en otras vistas
            st.session_state.efectividades = efectividades
            st.session_state.bateadores_seleccionados = bateadores
            st.session_state.pitcher_seleccionado = pitcher_seleccionado

        else:
            st.warning("No hay suficientes datos para generar predicciones.")

#vista de simulacion de partidos
elif page == "🥎 Simulación de Partidos":
    st.title("🥎 Simulación de Partidos")

    entrenar_modelo_si_no_existe()
    model = cargar_modelo()
    feature_columns = cargar_columnas()
    df = cargar_datos()

    if not st.session_state.inning_activo:
        bateadores = st.multiselect("Selecciona el orden de bateadores para el inning:", df["Bateador"].unique(), max_selections=9)
        if st.button("Iniciar Inning"):
            if bateadores:
                st.session_state.bateadores_turno = bateadores
                st.session_state.bateador_actual = bateadores[0]
                st.session_state.inning_activo = True
            else:
                st.warning("Debes seleccionar al menos un bateador para iniciar.")
    else:
        st.subheader(f"Turno de: {st.session_state.bateador_actual}")
        st.text(f"Outs: {st.session_state.outs} | Bolas: {st.session_state.bolas} | Strikes: {st.session_state.strikes}")

        dibujar_campo()

        recomendaciones = recomendar_pitchers(st.session_state.bateador_actual, df, model, feature_columns)
        st.write("### 🧢 Recomendaciones de Pitchers para este bateador:")
        for pitcher, prob in recomendaciones:
            st.write(f"- {pitcher}: Probabilidad de que el bateador se embasase: {prob:.2f}")

        if st.button("Lanzar Pitch"):
            simular_lanzamiento()

            if st.session_state.outs >= 3:
                st.success("Inning Terminado!")
                st.session_state.inning_activo = False
            else:
                st.session_state.turno += 1
                if st.session_state.turno < len(st.session_state.bateadores_turno):
                    st.session_state.bateador_actual = st.session_state.bateadores_turno[st.session_state.turno % len(st.session_state.bateadores_turno)]
                else:
                    st.session_state.turno = 0
                    st.session_state.bateador_actual = st.session_state.bateadores_turno[0]

#vista de estadisticas
elif page == "📊 Estadísticas por Pitcher":
    st.title("📊 Estadísticas por Pitcher")

    df = cargar_datos()

    pitcher_seleccionado = st.selectbox("Selecciona un pitcher para ver sus estadísticas:", options=["Seleccionar..."] + list(df["Pitcher"].unique()), index=0)

    if pitcher_seleccionado != "Seleccionar...":
        img_path = f"images/pitchers/{pitcher_seleccionado.replace(' ', '_')}.jfif"
        if os.path.exists(img_path):
            st.image(img_path, caption=pitcher_seleccionado, width=180)
        else:
            st.warning("Imagen no disponible para este pitcher.")
        stats_pitcher = df[df["Pitcher"] == pitcher_seleccionado]
        st.write(stats_pitcher)
    else:
        st.info("Seleccione un pitcher para ver sus estadísticas.")

#vista de registro
elif page == "📝 Registro":
    st.title("📝 Registro de estadísticas")

    csv_path = "data/BateadorVsPitcher.csv"

    def actualizar_estadisticas(pitcher, bateador, nueva_data):
        df = pd.read_csv(csv_path)
        mask = (df['Pitcher'] == pitcher) & (df['Bateador'] == bateador)
        if mask.any():
            for col in nueva_data.columns:
                if col in df.columns and col not in ["Bateador", "Etapa", "Equipo", "Pitcher"]:
                    df.loc[mask, col] += nueva_data[col].values[0]
        else:
            df = pd.concat([df, nueva_data], ignore_index=True)
        df.to_csv(csv_path, index=False)
        st.success("Estadísticas actualizadas correctamente")

    df = cargar_datos()

    pitcher = st.selectbox("Selecciona el pitcher:", df['Pitcher'].unique())
    bateador = st.selectbox("Seleccione el bateador:", df['Bateador'].unique())

    st.header("Registrar estadísticas")
    col1, col2, col3 = st.columns(3)

    with col1:
        AB = st.number_input("Turnos al bate (AB)", min_value=0)
        R = st.number_input("Carreras (R)", min_value=0)
        H = st.number_input("Hits (H)", min_value=0)
        HR = st.number_input("Home Runs (HR)", min_value=0)

    with col2:
        _2B = st.number_input("Dobles (2B)", min_value=0)
        _3B = st.number_input("Triples (3B)", min_value=0)
        RBI = st.number_input("Carreras impulsadas (RBI)", min_value=0)
        SO = st.number_input("Strikeouts (SO)", min_value=0)

    with col3:
        BB = st.number_input("Bases por bolas (BB)", min_value=0)
        IBB = st.number_input("BB Intencionales (IBB)", min_value=0)
        Equipo = st.selectbox("Equipo", ["Selecciona un equipo", "Tigres del Licey", "Leones del Escogido"])

    if st.button("Guardar estadísticas"):
        nueva_data = pd.DataFrame({
            "Bateador": [bateador], "Etapa": ["Serie Regular"], "Equipo": [Equipo], "Pitcher": [pitcher],
            "AB": [AB], "R": [R], "H": [H], "2B": [_2B], "3B": [_3B], "HR": [HR],
            "RBI": [RBI], "BB": [BB], "IBB": [IBB], "SO": [SO]
        })
        actualizar_estadisticas(pitcher, bateador, nueva_data)

    st.header("Estadísticas actuales")
    st.dataframe(df)

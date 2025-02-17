import pandas as pd

def load_and_clean_data(data, from_dataframe=False):
    if not from_dataframe:
        df = pd.read_csv(data)
    else:
        df = data.copy()

    #Creando la variable embasado donde se evalúa los hits, base por bola y bases por bolas intencionadas
    df["Embasado"] = ((df["H"] > 0) | (df["BB"] > 0) | (df["IBB"] > 0)).astype(int)

    df.drop(columns=["AVG", "OBP", "SLG", "OPS", "R", "RBI"], inplace=True, errors="ignore")
    df = pd.get_dummies(df, columns=["Bateador", "Etapa", "Equipo", "Pitcher"], drop_first=True)

    return df

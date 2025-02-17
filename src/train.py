import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from preprocessing import load_and_clean_data

df = load_and_clean_data("data/BateadorVsPitcher.csv")
print("Columnas despues de limpieza: ", df.columns)

#Guardando las columnas antes de entrenar el modelo
feature_columns = df.drop(columns=["Embasado"]).columns
pickle.dump(feature_columns, open("models/feature_columns.pkl", "wb"))

X = df.drop(columns=["Embasado"])
y = df["Embasado"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo: ", accuracy)

#Guardando el modelo
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
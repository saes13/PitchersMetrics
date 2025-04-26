import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from src.preprocessing import load_and_clean_data

# Cargar y limpiar datos
df = load_and_clean_data("data/BateadorVsPitcher.csv")
print("Columnas después de limpieza: ", df.columns)

# Definir características y objetivo
X = df.drop(columns=["Embasado"])
y = df["Embasado"]

# Guardar columnas de características
pickle.dump(X.columns, open("models/feature_columns.pkl", "wb"))

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Random Forest con class_weight balanced
model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_features="sqrt",
    bootstrap=True,
    max_samples=2/3,
    oob_score=True,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo: ", accuracy)
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# Guardar el modelo entrenado
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
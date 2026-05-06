import pandas as pd
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature

# =========================
# 1. Cargar / Descargar datos
# =========================

# Crear carpeta data si no existe
os.makedirs("data", exist_ok=True)

# URL del dataset Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# Leer dataset desde internet
df = pd.read_csv(url)

# Guardarlo localmente
df.to_csv("data/titanic.csv", index=False)

print("Dataset cargado correctamente")

# =========================
# 2. Preprocesamiento
# =========================

# Selección de variables relevantes
df = df[["Survived", "Pclass", "Age", "Fare"]]

# Eliminar nulos
df = df.dropna()

# Separar variables
X = df.drop("Survived", axis=1)
y = df["Survived"]

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Preprocesamiento completado")

# =========================
# 3. MLflow setup
# =========================

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("titanic_experiment")

# =========================
# 4. Entrenamiento
# =========================

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # =========================
    # 5. Evaluación
    # =========================

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("Accuracy:", acc)
    print("F1 Score:", f1)

    # =========================
    # 6. Logging MLflow
    # =========================

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Firma del modelo
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train[:5]
    )

    print("Modelo registrado en MLflow correctamente")
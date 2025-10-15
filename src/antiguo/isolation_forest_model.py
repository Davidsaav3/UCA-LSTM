import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# CARGA DE PARAMETROS
with open("parameters_config.json") as f:
    config = json.load(f)
params = config["isolation_forest_model"]

df = pd.read_csv("../results/intermediate/dataset_with_anomalies.csv")
data = df[[params["target_col"]]].values

# =========================
# ESCALADO OPCIONAL DE LOS DATOS
# =========================
# Si se indica en JSON que se quiere escalar, usamos StandardScaler
# para normalizar los datos (media 0, desviación típica 1)
if params.get("scaler") == "StandardScaler":
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
else:
    # Si no se desea escalado, usamos los datos tal cual
    data_scaled = data

# =========================
# PARÁMETROS DEL MODELO ISOLATION FOREST
# =========================
# n_estimators: número de árboles en el bosque (por defecto 100)
# max_samples: número de muestras por árbol (por defecto 'auto'=mismo tamaño del dataset)
# contamination: proporción esperada de anomalías (por defecto 0.05)
# max_features: número máximo de características usadas por árbol (por defecto 1.0 = todas)
# bootstrap: si se muestrean con reemplazo (por defecto False)
# random_state: semilla para reproducibilidad (por defecto 42)
clf_params = {
    "n_estimators": params.get("n_estimators", 100),
    "max_samples": params.get("max_samples", "auto"),
    "contamination": params.get("contamination", 0.05),
    "max_features": params.get("max_features", 1.0),
    "bootstrap": params.get("bootstrap", False),
    "random_state": params.get("random_state", 42)
}

# =========================
# CREACION Y ENTRENAMIENTO DEL MODELO
# =========================
# Se instancia IsolationForest con los parámetros definidos y se entrena
clf = IsolationForest(**clf_params)
clf.fit(data_scaled)

# =========================
# OBTENCIÓN DE SCORES Y PREDICCIONES
# =========================
# scores: puntuación de anomalía (valores negativos => más anómalo)
# predictions: -1 = anómalo, 1 = normal
scores = clf.decision_function(data_scaled)
predictions = clf.predict(data_scaled)

# Creamos columnas para los resultados en el DataFrame
df["score"] = scores
df["is_anomaly_detected"] = np.where(predictions == -1, 1, 0)

# =========================
# GUARDAR RESULTADOS
# =========================
# Creamos carpeta si no existe y guardamos CSV con scores y detecciones
os.makedirs(os.path.dirname(params["output_path"]), exist_ok=True)
df.to_csv(params["output_path"], index=False)

# GUARDAR PARAMETROS
with open("../results/parameters.txt", "a") as f:
    f.write("\n===== isolation_forest_model.py =====\n")
    for k,v in clf_params.items():
        f.write(f"{k}: {v}\n")

print("Paso 3 completado: Isolation Forest ejecutado y resultados guardados ✅")

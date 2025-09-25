import pandas as pd
import numpy as np
import json
import os

# =========================
# CARGAR PARAMETROS DESDE JSON
# =========================
with open("parameters_config.json") as f:
    params = json.load(f)["add_anomalies"]

# CARGAR DATASET ORIGINAL
df = pd.read_csv("../results/intermediate/original_dataset.csv")

# FIJAR SEMILLA PARA REPRODUCIBILIDAD
np.random.seed(params["random_state"])

# ===================================================
# DECIDIR SI USAR ANOMALIAS EXISTENTES O GENERAR SINTETICAS
# ===================================================
if params.get("use_existing_anomalies", False):
    # ✅ USAR ANOMALIAS EXISTENTES
    # Si la columna 'is_anomaly_real' no existe, la creamos e inicializamos a 0
    if "is_anomaly_real" not in df.columns:
        df["is_anomaly_real"] = 0

    # Obtenemos los índices de las filas que ya están marcadas como anomalías
    anomaly_indices = df.index[df["is_anomaly_real"] == 1].tolist()

else:
    # ❌ GENERAR ANOMALIAS SINTETICAS
    n_samples = len(df)  # Número total de filas del dataset
    n_anomalies = int(n_samples * params["anomaly_fraction"])  # Número de anomalías a introducir

    # Seleccionamos índices aleatorios sin reemplazo
    anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)

    # Inicializamos columna de anomalías reales
    df["is_anomaly_real"] = 0

    # Introducir anomalías
    for idx in anomaly_indices:
        if np.issubdtype(df[params["target_col"]].dtype, np.number):
            # Para columnas numéricas, multiplicar por un factor aleatorio grande
            df.at[idx, params["target_col"]] *= np.random.uniform(5, 10)
        else:
            # Para columnas no numéricas, añadir un sufijo para marcar la anomalía
            df.at[idx, params["target_col"]] = str(df.at[idx, params["target_col"]]) + "_anom"

        # Marcar la fila como anomalía real
        df.at[idx, "is_anomaly_real"] = 1

# =========================
# GUARDAR RESULTADOS
# =========================
# Guardar CSV con las anomalías procesadas
df.to_csv(params["output_path"], index=False)

# Guardar índices de anomalías en archivo separado
np.savetxt(params["anomaly_indices_path"], anomaly_indices, fmt="%d")

# Guardar parámetros usados
with open("../results/parameters.txt", "a") as f:
    f.write("\n===== add_anomalies.py =====\n")
    for k, v in params.items():
        f.write(f"{k}: {v}\n")

print("Paso 2 completado: Anomalías procesadas y guardadas ✅")

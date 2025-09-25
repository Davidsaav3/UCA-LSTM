import pandas as pd
import numpy as np
import json
import os

# CARGAR PARAMETROS
with open("parameters_config.json") as f:
    params = json.load(f)["add_anomalies"]

df = pd.read_csv("../results/intermediate/original_dataset.csv")
np.random.seed(params["random_state"])

if params.get("use_existing_anomalies", False):
    # ✅ USAR ANOMALIAS EXISTENTES
    if "is_anomaly_real" not in df.columns:
        # Si no existe la columna, la inicializamos a 0
        df["is_anomaly_real"] = 0
    anomaly_indices = df.index[df["is_anomaly_real"]==1].tolist()
else:
    # ❌ INTRODUCIR ANOMALIAS SINTETICAS
    n_samples = len(df)
    n_anomalies = int(n_samples * params["anomaly_fraction"])
    anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)
    df["is_anomaly_real"] = 0
    for idx in anomaly_indices:
        if np.issubdtype(df[params["target_col"]].dtype, np.number):
            df.at[idx, params["target_col"]] *= np.random.uniform(5, 10)
        else:
            df.at[idx, params["target_col"]] = str(df.at[idx, params["target_col"]]) + "_anom"
        df.at[idx, "is_anomaly_real"] = 1

# GUARDAR RESULTADOS
df.to_csv(params["output_path"], index=False)
np.savetxt(params["anomaly_indices_path"], anomaly_indices, fmt="%d")

# GUARDAR PARAMETROS
with open("../results/parameters.txt", "a") as f:
    f.write("\n===== add_anomalies.py =====\n")
    for k,v in params.items():
        f.write(f"{k}: {v}\n")

print("Paso 2 completado: Anomalías procesadas y guardadas ✅")

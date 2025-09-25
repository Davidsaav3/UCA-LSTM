import pandas as pd
import os
import json

# CARGA DE DATASET ORIGINAL Y GUARDADO INTERMEDIO
with open("parameters_config.json") as f:
    params = json.load(f)["load_data"]

os.makedirs(os.path.dirname(params["output_path_intermediate"]), exist_ok=True)

df = pd.read_csv(params["csv_path"])
df.to_csv(params["output_path_intermediate"], index=False)

# GUARDAR PARAMETROS
with open("../results/parameters.txt", "a") as f:
    f.write("\n===== load_data.py =====\n")
    for k,v in params.items():
        f.write(f"{k}: {v}\n")

print("Paso 1 completado: Dataset cargado y guardado ✅")

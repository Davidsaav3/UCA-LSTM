import pandas as pd
import os
import json

def main():
    # === CARGA DE PARAMETROS ===
    with open("parameters_config.json") as f:
        params = json.load(f)["load_data"]

    # Crear carpetas si no existen
    os.makedirs(os.path.dirname(params["output_path_intermediate"]), exist_ok=True)
    os.makedirs(os.path.dirname(params["output_path_null"]), exist_ok=True)

    # === CARGA DEL DATASET ORIGINAL ===
    df = pd.read_csv(params["csv_path"])

    # Guardar dataset intermedio para uso posterior
    df.to_csv(params["output_path_intermediate"], index=False)

    print(f"Dataset original guardado en {params['output_path_intermediate']} ✅")

    # === ELIMINAR FILAS CON VALORES VACÍOS ===
    n_filas_original = len(df)
    df_clean = df.dropna()
    n_filas_eliminadas = n_filas_original - len(df_clean)

    # Guardar dataset limpio
    df_clean.to_csv(params["output_path_null"], index=False)
    print(f"Dataset limpio guardado en {params['output_path_null']} ✅")
    print(f"Filas eliminadas por contener valores vacíos: {n_filas_eliminadas}")

    # === GUARDAR PARAMETROS USADOS ===
    os.makedirs("../results", exist_ok=True)
    with open("../results/parameters.txt", "a") as f:
        f.write("\n===== load_data.py =====\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"filas_eliminadas_por_nulos: {n_filas_eliminadas}\n")

if __name__ == "__main__":
    main()

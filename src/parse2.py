import json
import pandas as pd

# Cargar JSON
with open("autoconsumo2025.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraer columnas y valores
columnas = data["columns"]
valores = [v for v in data["values"] if v]  # eliminar listas vacías

# Crear DataFrame
df = pd.DataFrame(valores, columns=columnas)

# Obtener lista de UIDs únicos
uids = df['uid'].unique()

# Crear un diccionario para almacenar DataFrames pivotados por uid
dfs = []

for uid in uids:
    df_uid = df[df['uid'] == uid].copy()
    # Renombrar columnas agregando el UID como prefijo
    df_uid = df_uid.add_prefix(f"{uid}_")
    dfs.append(df_uid.reset_index(drop=True))

# Concatenar todos los DataFrames por columnas
df_final = pd.concat(dfs, axis=1)

# Guardar a Excel
df_final.to_excel("autoconsumo2025_uid.xlsx", index=False, engine='openpyxl')

print("Archivo Excel pivotado por UID creado correctamente.")

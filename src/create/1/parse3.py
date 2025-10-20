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

# Función para limpiar nombres de columnas
def limpiar_nombre(nombre):
    return nombre.replace(" ", "_").replace("-", "_").replace("/", "_")

# Pivotar los datos por UID y time
dfs = []
uids = df['uid'].unique()

for uid in uids:
    df_uid = df[df['uid'] == uid].copy()
    prefijo = limpiar_nombre(uid)
    df_uid = df_uid.set_index('time')  # usar time como índice
    # Renombrar columnas agregando prefijo
    df_uid = df_uid.add_prefix(f"{prefijo}_")
    dfs.append(df_uid)

# Concatenar todos los DataFrames por índice (time)
df_final = pd.concat(dfs, axis=1)

# Guardar a Excel
df_final.to_excel("autoconsumo2025_uid_time.xlsx", engine='openpyxl')

print("Archivo Excel pivotado por UID y alineado por time creado correctamente.")

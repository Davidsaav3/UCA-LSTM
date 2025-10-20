import json
import pandas as pd
import unicodedata

# ------------------------
# Función para limpiar nombres
# ------------------------
def limpiar_nombre(nombre):
    if not isinstance(nombre, str):
        nombre = str(nombre)
    nombre = nombre.lower()
    nombre = ''.join(
        c for c in unicodedata.normalize('NFD', nombre)
        if unicodedata.category(c) != 'Mn'
    )
    nombre = nombre.replace(" ", "_").replace("-", "_").replace("/", "_")
    return nombre

# ------------------------
# Cargar JSON
# ------------------------
with open("energia_21-25.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraer columnas y valores
columnas = data["columns"]
valores = [v for v in data["values"] if v and len(v) == len(columnas)]  # filtrar filas mal formadas

# Crear DataFrame
df = pd.DataFrame(valores, columns=columnas)

# Limpiar nombres de columnas
df.columns = [limpiar_nombre(c) for c in df.columns]

# ------------------------
# Pivotar datos por UID
# ------------------------
dfs = []
uids = df["uid"].unique()
total = len(uids)

for i, uid in enumerate(uids, start=1):
    df_uid = df[df["uid"] == uid].copy()
    prefijo = limpiar_nombre(uid)
    df_uid = df_uid.set_index("time")  # usar 'time' como índice
    df_uid = df_uid.add_prefix(f"{prefijo}_")  # añadir prefijo a columnas
    dfs.append(df_uid)

    # Mostrar progreso
    print(f"Procesado {i}/{total} UIDs ({(i/total)*100:.2f}%)")

# Concatenar todos los DataFrames por índice (time)
df_final = pd.concat(dfs, axis=1)

# ------------------------
# Guardar resultado en CSV
# ------------------------
df_final.to_csv("energia_21-25.csv", index=True, encoding="utf-8")
print("✅ Archivo CSV creado correctamente: energia_21-25.csv")

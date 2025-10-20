import json
import pandas as pd
import unicodedata

# Cargar JSON
with open("fotovoltaica_22-25.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraer columnas y valores
columnas = data["columns"]
valores = [v for v in data["values"] if v]  # eliminar listas vacías

# Crear DataFrame
df = pd.DataFrame(valores, columns=columnas)

# Función para limpiar nombres de columnas
def limpiar_nombre(nombre):
    # Convertir a minúsculas
    nombre = nombre.lower()
    # Quitar acentos
    nombre = ''.join(c for c in unicodedata.normalize('NFD', nombre)
                     if unicodedata.category(c) != 'Mn')
    # Reemplazar espacios y caracteres especiales por guion bajo
    nombre = nombre.replace(" ", "_").replace("-", "_").replace("/", "_")
    return nombre

# Aplicar limpieza a todas las columnas
df.columns = [limpiar_nombre(c) for c in df.columns]

# Pivotar los datos por UID y time
dfs = []
uids = df['uid'].unique()
total = len(uids)

for i, uid in enumerate(uids, start=1):
    df_uid = df[df['uid'] == uid].copy()
    prefijo = limpiar_nombre(uid)
    df_uid = df_uid.set_index('time')  # usar time como índice
    # Renombrar columnas agregando prefijo
    df_uid = df_uid.add_prefix(f"{prefijo}_")
    dfs.append(df_uid)

    # Mostrar progreso
    print(f"Procesado {i}/{total} UIDs ({(i/total)*100:.2f}%)")

# Concatenar todos los DataFrames por índice (time)
df_final = pd.concat(dfs, axis=1)

# Guardar a Excel
df_final.to_excel("fotovoltaica_22-25.xlsx", engine='openpyxl')

print("✅ Archivo Excel pivotado por UID y alineado por time creado correctamente.")

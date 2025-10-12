import json
import pandas as pd
import unicodedata
import re
import math

# Cargar JSON
with open("agua_22-25.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraer columnas y valores
columnas = data["columns"]
valores = [v for v in data["values"] if v]  # eliminar listas vacías

# Crear DataFrame
df = pd.DataFrame(valores, columns=columnas)

# Función para limpiar nombres (robusta ante None/NaN y tipos no-string)
def limpiar_nombre(nombre):
    # Normalizar None / NaN
    if nombre is None:
        nombre = "sin_uid"
    else:
        try:
            if isinstance(nombre, float) and math.isnan(nombre):
                nombre = "sin_uid"
        except Exception:
            pass
    # Forzar string y minusculas
    nombre = str(nombre).lower()
    # Quitar acentos
    nombre = ''.join(c for c in unicodedata.normalize('NFD', nombre)
                     if unicodedata.category(c) != 'Mn')
    # Reemplazar cualquier caracter no alfanumérico por guion bajo
    nombre = re.sub(r'[^0-9a-z]+', '_', nombre)
    # Colapsar guiones bajos múltiples y quitar inicial/final
    nombre = re.sub(r'_+', '_', nombre).strip('_')
    if nombre == "":
        nombre = "sin_uid"
    return nombre

# Aplicar limpieza a todas las columnas (por si hay acentos/espacios en los nombres de columna)
df.columns = [limpiar_nombre(c) for c in df.columns]

# Comprobar que existen las columnas necesarias
if 'uid' not in df.columns:
    raise KeyError("No se encontró la columna 'uid' en el JSON original.")
if 'time' not in df.columns:
    raise KeyError("No se encontró la columna 'time' en el JSON original.")

# Normalizar valores de uid (rellenar NaN y forzar string)
df['uid'] = df['uid'].fillna('sin_uid').astype(str)

# Pivotar los datos por UID y time
dfs = []
uids = df['uid'].unique()
total = len(uids)

for i, uid in enumerate(uids, start=1):
    df_uid = df[df['uid'] == uid].copy()
    prefijo = limpiar_nombre(uid)
    # usar time como índice
    df_uid = df_uid.set_index('time')
    # eliminar columna uid (es redundante una vez agrupado por uid)
    df_uid = df_uid.drop(columns=['uid'], errors='ignore')
    # Renombrar columnas agregando prefijo
    df_uid = df_uid.add_prefix(f"{prefijo}_")
    dfs.append(df_uid)

    # Mostrar progreso
    print(f"Procesado {i}/{total} UIDs ({(i/total)*100:.2f}%) - prefijo: {prefijo}")

# Concatenar todos los DataFrames por índice (time)
if not dfs:
    print("No se ha generado ningún DataFrame. Revisa el archivo JSON.")
else:
    df_final = pd.concat(dfs, axis=1).sort_index()
    # Guardar a Excel
    df_final.to_excel("agua_22-25.xlsx", engine='openpyxl')
    print("✅ Archivo Excel pivotado por UID y alineado por time creado correctamente: agua_22-25.xlsx")

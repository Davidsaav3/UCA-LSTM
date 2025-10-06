import os
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
# Carpeta con archivos
# ------------------------
carpeta = "wifi"

# Archivos que vamos a procesar (2019_H1 hasta 2025_H2)
archivos = [
    f"datos_{anio}_H{sem}.json"
    for anio in range(2019, 2026)
    for sem in (1, 2)
]

# ------------------------
# Procesar archivos uno por uno
# ------------------------
for archivo in archivos:
    ruta = os.path.join(carpeta, archivo)
    if not os.path.exists(ruta):
        print(f"⚠️ No se encontró {archivo}, se salta...")
        continue

    print(f"📂 Procesando {archivo}...")

    # Cargar JSON
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)

    columnas = data["columns"]
    valores = [v for v in data["values"] if v and len(v) == len(columnas)]

    # Crear DataFrame
    df = pd.DataFrame(valores, columns=columnas)
    df.columns = [limpiar_nombre(c) for c in df.columns]

    # Obtener UIDs únicos
    uids = df["uid"].unique()
    total = len(uids)

    # Crear CSV completo
    df_final = pd.concat(
        [df[df["uid"] == uid].set_index("time").add_prefix(f"{limpiar_nombre(uid)}_")
         for uid in uids],
        axis=1
    )
    salida_csv = archivo.replace(".json", ".csv")
    df_final.to_csv(os.path.join(carpeta, salida_csv), index=True, encoding="utf-8")
    print(f"✅ CSV generado: {salida_csv}")

    # Crear Excel con hojas por UID
    salida_xlsx = archivo.replace(".json", ".xlsx")
    with pd.ExcelWriter(os.path.join(carpeta, salida_xlsx), engine="openpyxl") as writer:
        for i, uid in enumerate(uids, start=1):
            df_uid = df[df["uid"] == uid].copy()
            df_uid = df_uid.set_index("time")
            hoja = limpiar_nombre(uid)[:31]  # Excel limita nombre a 31 chars
            df_uid.to_excel(writer, sheet_name=hoja)
            print(f"   - UID {i}/{total} guardado en hoja {hoja}")

    print(f"✅ Excel generado: {salida_xlsx}\n")

print("🎉 Conversión terminada para todos los archivos.")

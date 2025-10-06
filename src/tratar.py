import pandas as pd
import os

# Carpeta donde están los archivos
carpeta = r"C:\Users\UA\Documents\anomaly-detection-lstm\src"

# Generar lista de archivos automáticamente
archivos = []
for año in range(2019, 2026):  # 2019 a 2025
    for semestre in ["H1", "H2"]:
        nombre_archivo = f"datos_{año}_{semestre}.xlsx"
        ruta_completa = os.path.join(carpeta, nombre_archivo)
        if os.path.exists(ruta_completa):
            archivos.append(ruta_completa)

# Procesar cada archivo
for archivo_excel in archivos:
    print(f"Procesando {archivo_excel}...")

    xls = pd.ExcelFile(archivo_excel)
    hojas = xls.sheet_names

    df_base = None  # contendrá 'time' y las columnas finales

    for hoja in hojas:
        df = pd.read_excel(archivo_excel, sheet_name=hoja, usecols=["time", "value"])

        # Convertir time a formato datetime (seguridad ante strings)
        df["time"] = pd.to_datetime(df["time"], utc=True)

        # Redondear a precisión de minutos para garantizar alineación exacta
        df["time"] = df["time"].dt.floor("min")

        # Renombrar columna value según la hoja
        df = df.rename(columns={"value": f"wifi_{hoja}"})

        # Alinear por time
        if df_base is None:
            df_base = df
        else:
            df_base = pd.merge(df_base, df, on="time", how="outer")

    # Ordenar y limpiar duplicados si existen
    df_base = df_base.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    # Guardar resultado
    nombre_salida = archivo_excel.replace(".xlsx", "_trat.xlsx")
    df_base.to_excel(nombre_salida, index=False)

    print(f"Archivo combinado creado: {nombre_salida}")

print("Todos los archivos han sido procesados.")

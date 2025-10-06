import pandas as pd
import os

# Ruta del archivo de entrada
archivo_entrada = r"C:\Users\UA\Documents\anomaly-detection-lstm\src\procesar.xlsx"
archivo_salida = archivo_entrada.replace(".xlsx", "_alineado.csv")

# Leer todo el archivo Excel
df = pd.read_excel(archivo_entrada)

# Detectar automáticamente las columnas de tiempo por sufijo "_time"
columnas_time = [c for c in df.columns if c.endswith("_time")]
print("Columnas de tiempo detectadas:", columnas_time)

# Lista para guardar los DataFrames parciales
dfs = []

# Procesar cada bloque de datos
for col_time in columnas_time:
    prefijo = col_time.replace("_time", "")
    cols_prefijo = [c for c in df.columns if c.startswith(prefijo + "_") and c != col_time]

    if not cols_prefijo:
        continue

    # Extraer bloque de datos
    bloque = df[[col_time] + cols_prefijo].copy()

    # Convertir a datetime (detecta ambos formatos automáticamente)
    bloque[col_time] = pd.to_datetime(bloque[col_time], utc=True, errors='coerce')
    bloque = bloque.dropna(subset=[col_time])

    # Redondear a minuto exacto
    bloque[col_time] = bloque[col_time].dt.floor("min")

    # Renombrar columna de tiempo a 'time' para unificar
    bloque = bloque.rename(columns={col_time: "time"})

    dfs.append(bloque)

# Crear rango temporal unificado (todos los minutos desde el menor al mayor)
todos_tiempos = pd.concat([b["time"] for b in dfs])
rango_tiempo = pd.date_range(todos_tiempos.min(), todos_tiempos.max(), freq="min", tz="UTC")
df_final = pd.DataFrame({"time": rango_tiempo})

# Unir todos los bloques por 'time'
for bloque in dfs:
    df_final = pd.merge(df_final, bloque, on="time", how="outer")

# Ordenar y resetear índice
df_final = df_final.sort_values("time").reset_index(drop=True)

# Guardar en CSV
df_final.to_csv(archivo_salida, index=False)
print(f"\nArchivo alineado creado en CSV: {archivo_salida}")

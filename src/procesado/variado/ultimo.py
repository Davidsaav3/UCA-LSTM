import pandas as pd
import os

# Carpeta donde están los archivos CSV
carpeta = r"C:\Users\UA\Documents\anomaly-detection-lstm\src"

# Años a procesar
años = range(2021, 2026)

for año in años:
    print(f"Procesando {año}...")

    # Archivos CSV
    archivo_todo = os.path.join(carpeta, f"todo_{año}.csv")
    archivo_wifi = os.path.join(carpeta, f"wifi_{año}.csv")
    
    # Leer CSVs evitando DtypeWarning
    df_todo = pd.read_csv(archivo_todo, sep=';', decimal='.', parse_dates=['time'], low_memory=False)
    df_wifi = pd.read_csv(archivo_wifi, sep=';', decimal='.', parse_dates=['time'], low_memory=False)
    
    # Alineación por fecha usando merge externo
    df_merge = pd.merge(df_todo, df_wifi, on='time', how='outer', suffixes=('_todo', '_wifi'))
    
    # --- Archivo con columna time de "wifi" ---
    df_wifi_time = df_merge.copy()
    # Reemplazar la columna time por la de wifi original si existe
    if 'time' in df_wifi.columns:
        df_wifi_time['time'] = df_wifi_time['time']
    df_wifi_time.to_csv(os.path.join(carpeta, f"unido_wifi_time_{año}.csv"), sep=';', index=False)
    
    # --- Archivo con columna time de "todo" ---
    df_todo_time = df_merge.copy()
    # Reemplazar la columna time por la de todo original si existe
    if 'time' in df_todo.columns:
        df_todo_time['time'] = df_todo_time['time']
    df_todo_time.to_csv(os.path.join(carpeta, f"unido_todo_time_{año}.csv"), sep=';', index=False)

print("Procesamiento completado.")

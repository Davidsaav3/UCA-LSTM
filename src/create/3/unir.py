import pandas as pd

# Cargar los CSV
df1 = pd.read_csv("a2024.csv")
df2 = pd.read_csv("b2024.csv")

# Convertir las columnas de fecha a datetime con zona horaria
df1['datetime-a'] = pd.to_datetime(df1['datetime-a'], utc=True)
df2['datetime-b'] = pd.to_datetime(df2['datetime-b'], utc=True)

# Renombrar la columna de df2 para que coincida con df1 temporalmente
df2 = df2.rename(columns={'datetime-b': 'datetime-a'})

# Fusionar ambos dataframes por la columna de fecha
df_merged = pd.merge(df1, df2, on='datetime-a', how='outer')  # 'outer' mantiene todas las fechas

# Guardar el resultado en un nuevo CSV
df_merged.to_csv("csv_fusionado.csv", index=False)

print("CSV fusionado creado: csv_fusionado.csv")

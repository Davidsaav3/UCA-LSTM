import pandas as pd

# Leer CSV
df = pd.read_csv("todo_2025.csv", parse_dates=['time'])

# Filtrar solo filas con minutos 0, 15, 30, 45
df_filtrado = df[df['time'].dt.minute.isin([0, 15, 30, 45])]

# Guardar resultado en nuevo CSV
df_filtrado.to_csv("todo_2025_filtrado.csv", index=False)

print("Filtrado completado. Se guardó 'archivo_filtrado.csv'.")

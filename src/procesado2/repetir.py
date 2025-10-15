import pandas as pd

# Ruta del archivo CSV original
input_csv = "aemet.csv"

# Ruta del archivo CSV de salida
output_csv = "aemet2.csv"

# Leer el CSV original
df = pd.read_csv(input_csv)

# Repetir cada fila 96 veces
df_repetido = df.loc[df.index.repeat(96)].reset_index(drop=True)

# Guardar el nuevo CSV
df_repetido.to_csv(output_csv, index=False)

print(f"Se ha generado '{output_csv}' con cada fila repetida 96 veces.")

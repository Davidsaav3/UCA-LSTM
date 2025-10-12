import json
import pandas as pd

# Cargar el JSON desde un archivo
with open("autoconsumo2025.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraer columnas y valores
columnas = data["columns"]
valores = data["values"]

# Filtrar listas vacías
valores = [v for v in valores if v]  # elimina listas vacías

# Crear DataFrame
df = pd.DataFrame(valores, columns=columnas)

# Guardar a Excel
df.to_excel("autoconsumo2025.xlsx", index=False, engine='openpyxl')

print("Archivo Excel creado correctamente.")

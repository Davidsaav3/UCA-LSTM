import json

# Nombres de los dos archivos
archivo1 = "wifi2018.json"
archivo2 = "wifi2019-ext.json"
salida = "wifi_18-25.json"

# Cargar primer archivo
with open(archivo1, "r", encoding="utf-8") as f1:
    data1 = json.load(f1)

# Cargar segundo archivo
with open(archivo2, "r", encoding="utf-8") as f2:
    data2 = json.load(f2)

# Mantener las columnas del primero
columnas = data1["columns"]

# Combinar los valores de ambos
valores = data1["values"] + data2["values"]

# Crear el nuevo JSON
resultado = {
    "columns": columnas,
    "values": valores
}

# Guardar en archivo
with open(salida, "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2)

print(f"Archivos combinados en {salida}")

import ijson
import json
import os
from datetime import datetime
from decimal import Decimal

# Función para convertir Decimals a float (o str si prefieres)
def convert(o):
    if isinstance(o, list):
        return [convert(i) for i in o]
    elif isinstance(o, dict):
        return {k: convert(v) for k, v in o.items()}
    elif isinstance(o, Decimal):
        return float(o)  # también podrías usar str(o) si quieres evitar redondeos
    else:
        return o

# Archivo de entrada
input_file = "wifi2019.json"

# Carpeta de salida
output_dir = "salida"
os.makedirs(output_dir, exist_ok=True)

# Abrimos el JSON en streaming para leer columnas
with open(input_file, "r", encoding="utf-8") as f:
    parser = ijson.parse(f)
    columnas = []
    for prefix, event, value in parser:
        if prefix == "columns.item":
            columnas.append(value)
        elif prefix == "values":
            break
    f.seek(0)

# Diccionario de escritores
writers = {}

# Contador
total = 0
report_interval = 1_000_000  # cada millón de filas

with open(input_file, "r", encoding="utf-8") as f:
    objetos = ijson.items(f, "values.item")
    for fila in objetos:
        total += 1
        fila = convert(fila)  # 🔑 convertir Decimals a float/str

        # Extraer fecha
        fecha = datetime.fromisoformat(fila[0].replace("Z", "+00:00"))
        year = fecha.year
        half = 1 if fecha.month <= 6 else 2

        clave = f"{year}_H{half}"
        if clave not in writers:
            output_file = os.path.join(output_dir, f"datos_{clave}.json")
            writers[clave] = {
                "file": open(output_file, "w", encoding="utf-8"),
                "first": True
            }
            writers[clave]["file"].write("{\n")
            writers[clave]["file"].write(f'  "columns": {json.dumps(columnas, ensure_ascii=False)},\n')
            writers[clave]["file"].write('  "values": [\n')

        w = writers[clave]
        if not w["first"]:
            w["file"].write(",\n")
        w["file"].write("    " + json.dumps(fila, ensure_ascii=False))
        w["first"] = False

        if total % report_interval == 0:
            print(f"Procesadas {total:,} filas...")

# Cerrar archivos
for clave, w in writers.items():
    w["file"].write("\n  ]\n}")
    w["file"].close()

print(f"✅ Terminado. Total filas procesadas: {total:,}")

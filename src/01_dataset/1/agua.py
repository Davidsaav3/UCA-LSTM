import requests
import json
from datetime import datetime, timedelta

# -------------------------
# CONFIGURACIÓN
# -------------------------
YEAR = 2022   # <<-- CAMBIA AQUÍ EL AÑO QUE QUIERAS

# URL de la API
url = "https://openapi.kunna.es/01_dataset/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NTIxMzU4Njd9.gM49pEOaIMq7NJdwsX2jcI3DXl5AIbKaWoQ0HsYCIZo"

# Headers de la petición
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

# Fechas de inicio y fin
start_date = datetime(YEAR, 1, 1)
end_date = datetime(YEAR, 12, 31)

# Intervalo por petición
delta = timedelta(minutes=30)

# Inicializamos el JSON final
final_data = {
    "columns": [],
    "values": []
}

# Set para evitar duplicados (uid + time)
seen = set()

# Nombre del archivo final según el año
output_file = f"agua{YEAR}.json"

# -------------------------
# RECORRER BLOQUES DE FECHAS
# -------------------------
current_start = start_date
while current_start < end_date:
    current_end = min(current_start + delta, end_date)
    
    payload = {
        "time_start": current_start.isoformat() + "Z",
        "time_end": current_end.isoformat() + "Z",
        "filters": [],
        "limit": 3000,
        "count": False,
        "order": "DESC"
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        
        # Guardamos las columnas si aún no se han guardado
        if not final_data["columns"] and "columns" in data.get("result", {}):
            final_data["columns"] = data["result"]["columns"]
        
        # Agregamos los valores obtenidos sin duplicados
        values = data.get("result", {}).get("values", [])
        added = 0
        for row in values:
            # Evitar filas vacías o incompletas
            if len(row) < 2:
                print(f"Fila inválida descartada: {row}")
                continue
            
            unique_id = f"{row[1]}_{row[0]}"  # uid + time
            if unique_id not in seen:
                final_data["values"].append(row)
                seen.add(unique_id)
                added += 1
        
        print(f"Recogidos {len(values)} registros ({added} nuevos) de {payload['time_start']} a {payload['time_end']}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error en el bloque {payload['time_start']} a {payload['time_end']}: {e}")
    
    current_start = current_end

# -------------------------
# GUARDAR RESULTADOS
# -------------------------
with open(output_file, "w") as f:
    json.dump(final_data, f, indent=2)

print(f"✅ Total de registros recogidos: {len(final_data['values'])}")
print(f"📂 Datos guardados en: {output_file}")

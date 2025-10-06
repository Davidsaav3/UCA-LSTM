import pandas as pd

# --- Cargar Excel ---
df = pd.read_excel("clima_24-25_prep.xlsx")

# --- Convertir a datetime (maneja formatos mixtos) ---
df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

# --- Truncar a minutos ---
df["time_min"] = df["time"].dt.floor("min")

# --- Quitar zona horaria de la columna ---
df["time_min"] = df["time_min"].dt.tz_localize(None)

# --- Pivotar ---
df_pivot = df.pivot_table(index="time_min", columns="type", values="value", aggfunc="first")

# --- Quitar zona horaria del índice si existe ---
if pd.api.types.is_datetime64tz_dtype(df_pivot.index):
    df_pivot.index = df_pivot.index.tz_localize(None)

# --- Guardar a Excel ---
df_pivot.reset_index().to_excel("datos_pivotados.xlsx", index=False)

print("✅ Archivo generado correctamente: datos_pivotados.xlsx")

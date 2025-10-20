import pandas as pd  # PARA MANEJO DE DATAFRAMES
import numpy as np   # PARA MANIPULACIÓN DE ARRAYS

# PARÁMETROS
INPUT_CSV = '../../results/preparation/05_variance.csv'  # DATASET FINAL TRAS VARIANZA
OUTPUT_CSV = '../../results/preparation/06_sequences.csv'  # CSV CON SECUENCIAS GENERADAS
TIMESTEPS = 10  # NÚMERO DE PASOS TEMPORALES (VENTANA)
TARGET_COLUMN = None  # COLUMNA OBJETIVO PARA PREDICCIÓN, NONE = TODAS
SHOW_INFO = True  # MOSTRAR MENSAJES INFORMATIVOS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV, low_memory=False)  # LEER CSV
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# CREAR SECUENCIAS
sequence_list = []  # LISTA PARA ALMACENAR SECUENCIAS
n_features = df.shape[1] if TARGET_COLUMN is None else df.drop(columns=[TARGET_COLUMN]).shape[1]  # NÚMERO DE FEATURES

for i in range(len(df) - TIMESTEPS + 1):
    # EXTRAER VENTANA DE TIMESTEPS
    if TARGET_COLUMN is None:
        seq = df.iloc[i:i+TIMESTEPS].values  # INCLUYE TODAS LAS COLUMNAS
        seq = seq.flatten()  # 🔹 CORRECCIÓN: aplanar 2D a 1D para poder crear DataFrame 2D
    else:
        seq = df.drop(columns=[TARGET_COLUMN]).iloc[i:i+TIMESTEPS].values  # EXCLUYE TARGET DE FEATURES
        target = df[TARGET_COLUMN].iloc[i+TIMESTEPS-1]  # VALOR TARGET AL FINAL DE LA SECUENCIA
        seq = np.hstack([seq.flatten(), target])  # CONCATENAR FEATURES + TARGET
    sequence_list.append(seq)

# CONVERTIR LISTA DE SECUENCIAS A DATAFRAME
df_sequences = pd.DataFrame(sequence_list)  # 🔹 CORRECCIÓN: todas las secuencias ahora tienen la misma longitud 1D
if SHOW_INFO:
    print(f"[ INFO ] SECUENCIAS GENERADAS: {df_sequences.shape[0]} FILAS, {df_sequences.shape[1]} COLUMNAS")

# GUARDAR CSV CON SECUENCIAS
df_sequences.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] DATASET DE SECUENCIAS EN '{OUTPUT_CSV}'")

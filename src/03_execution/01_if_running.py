import pandas as pd 
import numpy as np
import os
import json
from sklearn.ensemble import IsolationForest  
from sklearn.preprocessing import StandardScaler  
import time  
import sys
import warnings

# === SILENCIAR WARNINGS DE NUMPY (runtime warnings internos de IsolationForest) ===
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.stdout.reconfigure(encoding='utf-8')

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/01_classification'           # CARPETA PRINCIPAL DE RESULTADOS
INPUT_CSV = '../../results/02_preparation/infrastructure/realtime/if/05_variance.csv'  # CSV SIMULADO TIEMPO REAL
PARAMS_JSON = os.path.join(RESULTS_FOLDER, '01_if_params.json')     # ARCHIVO HIPERPARÁMETROS
OUTPUT_CSV = os.path.join(RESULTS_FOLDER, '01_if.csv')                    # CSV FINAL COMPLETO
OUTPUT_IF_CSV = os.path.join(RESULTS_FOLDER, '01_if_anomaly.csv')         # CSV SOLO ANOMALÍAS

# FLAGS DE CONTROL
SAVE_ANOMALY_CSV = True  # GUARDAR CSV SOLO ANOMALÍAS
SHOW_INFO = True         # MOSTRAR INFO EN CONSOLA
SLEEP_TIME = 0.05        # RETARDO ENTRE REGISTROS (SIMULACIÓN TIEMPO REAL)

# CARGAR HIPERPARÁMETROS DESDE JSON
with open(PARAMS_JSON, 'r') as f:
    params = json.load(f)
if SHOW_INFO:
    print(f"[ INFO ] HIPERPARÁMETROS CARGADOS DESDE '{PARAMS_JSON}'")

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV, low_memory=False)  # LEER CSV

# === (1) DETECTAR Y CORREGIR NaNs e Infs ===
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))

# === (2) ELIMINAR COLUMNAS DE VARIANZA CERO O MUY PEQUEÑA ===
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
df_num = df[num_cols]

# Columnas exactamente cero
zero_var_cols = df_num.columns[df_num.var() == 0].tolist()
# Columnas con varianza extremadamente pequeña
small_var_cols = df_num.columns[df_num.var() < 1e-8].tolist()

# Unir listas y eliminar duplicados
drop_cols = list(set(zero_var_cols + small_var_cols))
if len(drop_cols) > 0 and SHOW_INFO:
    print(f"[ AVISO ] {len(drop_cols)} COLUMNAS ELIMINADAS POR VARIANZA CERO O MUY PEQUEÑA")

df = df.drop(columns=drop_cols)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# === (3) ESCALADO ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_cols])

# Limitar valores extremos para evitar posibles warnings internos
X_scaled = np.clip(X_scaled, -10, 10)

if SHOW_INFO:
    print(f"[ INFO ] DATASET LIMPIO Y ESCALADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# INICIALIZAR MODELO IF CON PARÁMETROS ÓPTIMOS
clf = IsolationForest(
    n_estimators=params['n_estimators'],
    max_samples=params['max_samples'],
    contamination=params['contamination'],
    max_features=params['max_features'],
    bootstrap=params['bootstrap'],
    n_jobs=-1,
    random_state=42
)
clf.fit(X_scaled)  # ENTRENAR MODELO
if SHOW_INFO:
    print("[ INFO ] MODELO IF ENTRENADO CON PARÁMETROS ÓPTIMOS")

# SIMULACIÓN DE ENTRADA EN TIEMPO REAL
results = []
for i in range(len(X_scaled)):
    sample = X_scaled[i].reshape(1, -1)  # SELECCIONAR REGISTRO
    pred = clf.predict(sample)[0]         # PREDICCIÓN ANOMALÍA
    score = clf.decision_function(sample)[0] * -1  # SCORE ANOMALÍA (MÁS POSITIVO = MÁS ANÓMALO)
    label = 1 if pred == -1 else 0       # CONVERTIR A 0/1
    results.append((i, label, score))    # GUARDAR RESULTADO

    if SHOW_INFO and i % 50 == 0:
        print(f"[ INFO ] REGISTRO {i}: SCORE={score:.4f} ANOMALÍA={label}")

    time.sleep(SLEEP_TIME)  # SIMULAR RETARDO TIEMPO REAL

# CONVERTIR RESULTADOS A DATAFRAME
df_results = pd.DataFrame(results, columns=['index', 'anomaly', 'anomaly_score'])
df_final = pd.concat([df.reset_index(drop=True), df_results[['anomaly', 'anomaly_score']]], axis=1)  # UNIR CON DATAFRAME ORIGINAL

# GUARDAR RESULTADOS COMPLETOS
df_final.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] RESULTADOS COMPLETOS EN '{OUTPUT_CSV}'")

# GUARDAR ANOMALÍAS DETECTADAS
if SAVE_ANOMALY_CSV:
    df_anomalies = df_final[df_final['anomaly'] == 1].copy()  # FILTRAR SOLO ANOMALÍAS
    df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False)  # ORDENAR POR SCORE
    df_anomalies.to_csv(OUTPUT_IF_CSV, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] CSV SOLO ANOMALÍAS EN '{OUTPUT_IF_CSV}'")

# SIMULACIÓN DE FLUJO EN TIEMPO REAL Y DETECCIÓN DE ANOMALÍAS CON ISOLATION FOREST

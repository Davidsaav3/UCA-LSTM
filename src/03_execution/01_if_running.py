import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/01_classification'           # CARPETA PRINCIPAL DE RESULTADOS
INPUT_CSV = '../../results/02_preparation/infrastructure/realtime/if/05_variance.csv'  # DATASET SIMULADO EN TIEMPO REAL
PARAMS_JSON = os.path.join(RESULTS_FOLDER, '01_if_best_params.json')           # ARCHIVO DE HIPERPARÁMETROS
OUTPUT_CSV = os.path.join(RESULTS_FOLDER, '01_if.csv')                    # CSV FINAL COMPLETO
OUTPUT_IF_CSV = os.path.join(RESULTS_FOLDER, '01_if_anomaly.csv')         # CSV SOLO ANOMALÍAS

# FLAGS DE CONTROL
SAVE_ANOMALY_CSV = True
SHOW_INFO = True
SLEEP_TIME = 0.05  # RETARDO ENTRE REGISTROS (SIMULACIÓN TIEMPO REAL)

# CARGAR HIPERPARÁMETROS DESDE JSON
with open(PARAMS_JSON, 'r') as f:
    params = json.load(f)
if SHOW_INFO:
    print(f"[ INFO ] HIPERPARÁMETROS CARGADOS DESDE '{PARAMS_JSON}'")

# CARGAR Y ESCALAR DATASET
df = pd.read_csv(INPUT_CSV, low_memory=False)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_cols])
if SHOW_INFO:
    print(f"[ INFO ] DATASET ESCALADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# INICIALIZAR MODELO CON HIPERPARÁMETROS ÓPTIMOS
clf = IsolationForest(
    n_estimators=params['n_estimators'],
    max_samples=params['max_samples'],
    contamination=params['contamination'],
    max_features=params['max_features'],
    bootstrap=params['bootstrap'],
    n_jobs=-1,
    random_state=42
)
clf.fit(X_scaled)
if SHOW_INFO:
    print("[ INFO ] MODELO IF ENTRENADO CON PARÁMETROS ÓPTIMOS")

# SIMULACIÓN DE ENTRADA EN TIEMPO REAL
results = []
for i in range(len(X_scaled)):
    sample = X_scaled[i].reshape(1, -1)
    pred = clf.predict(sample)[0]
    score = clf.decision_function(sample)[0] * -1  # MÁS POSITIVO = MÁS ANÓMALO
    label = 1 if pred == -1 else 0
    results.append((i, label, score))

    if SHOW_INFO and i % 50 == 0:
        print(f"[ INFO ] REGISTRO {i}: SCORE={score:.4f} ANOMALÍA={label}")

    time.sleep(SLEEP_TIME)  # SIMULAR RETARDO TEMPORAL

# CONVERTIR RESULTADOS A DATAFRAME
df_results = pd.DataFrame(results, columns=['index', 'anomaly', 'anomaly_score'])
df_final = pd.concat([df.reset_index(drop=True), df_results[['anomaly', 'anomaly_score']]], axis=1)

# GUARDAR RESULTADOS
df_final.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] RESULTADOS COMPLETOS EN '{OUTPUT_CSV}'")

# GUARDAR ANOMALÍAS DETECTADAS
if SAVE_ANOMALY_CSV:
    df_anomalies = df_final[df_final['anomaly'] == 1].copy()
    df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False)
    df_anomalies.to_csv(OUTPUT_IF_CSV, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] CSV SOLO ANOMALÍAS EN '{OUTPUT_IF_CSV}'")

# COMENTARIO FINAL
# SE SIMULA ENTRADA EN TIEMPO REAL Y SE DETECTAN ANOMALÍAS CON IF

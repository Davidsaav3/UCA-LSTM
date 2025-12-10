import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import IsolationForest  
from sklearn.preprocessing import StandardScaler  
import time  
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.stdout.reconfigure(encoding='utf-8')


# CONFIGURACIÓN
RESULTS_FOLDER = '../../results/03_execution/01_classification'
INPUT_CSV = '../../results/02_preparation/infrastructure/realtime/if/05_variance.csv'
PARAMS_JSON = os.path.join(RESULTS_FOLDER, '01_if_params.json')
OUTPUT_CSV = os.path.join(RESULTS_FOLDER, '01_if.csv')
OUTPUT_IF_CSV = os.path.join(RESULTS_FOLDER, '01_if_anomaly.csv')

SAVE_ANOMALY_CSV = True
SHOW_INFO = True
SLEEP_TIME = 0.05

# CARGAR HIPERPARÁMETROS
with open(PARAMS_JSON, 'r') as f:
    params = json.load(f)
if SHOW_INFO:
    print(f"[ INFO ] HIPERPARÁMETROS CARGADOS DESDE '{PARAMS_JSON}'")

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV, low_memory=False)

# === (1) LIMPIEZA DE NaNs e Infs ===
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))

# === (2) ELIMINAR COLUMNAS DE VARIANZA CERO/MUY BAJA ===
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
df_num = df[num_cols]

zero_var_cols = df_num.columns[df_num.var() == 0].tolist()
small_var_cols = df_num.columns[df_num.var() < 1e-8].tolist()
drop_cols = list(set(zero_var_cols + small_var_cols))

if len(drop_cols) > 0 and SHOW_INFO:
    print(f"[ AVISO ] {len(drop_cols)} COLUMNAS ELIMINADAS POR VARIANZA BAJA")

df = df.drop(columns=drop_cols)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()


# (3) AJUSTE ROBUSTO DE BASELINE Y TRAIN_SKIP

N = len(df)
BASELINE = min(500, N // 2)   # nunca más de la mitad del dataset
TRAIN_SKIP = min(300, N // 4) # nunca más de 1/4 del dataset

if SHOW_INFO:
    print(f"[ INFO ] BASELINE={BASELINE}, TRAIN_SKIP={TRAIN_SKIP}")


# (4) ESCALADO CON BASELINE

scaler = StandardScaler()
scaler.fit(df.iloc[:BASELINE][num_cols])
X_scaled = scaler.transform(df[num_cols])
X_scaled = np.clip(X_scaled, -10, 10)

if SHOW_INFO:
    print(f"[ INFO ] ESCALADO COMPLETADO")
    print(f"[ INFO ] DATASET FINAL: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")


# (5) ENTRENAR IF SALTANDO TRAIN_SKIP

clf = IsolationForest(
    n_estimators=params['n_estimators'],
    max_samples=params['max_samples'],
    contamination=params['contamination'],
    max_features=params['max_features'],
    bootstrap=params['bootstrap'],
    n_jobs=-1,
    random_state=42
)

clf.fit(X_scaled[TRAIN_SKIP:])  # ENTRENAR SIN ARRANQUE
if SHOW_INFO:
    print(f"[ INFO ] MODELO ENTRENADO SALTANDO LAS PRIMERAS {TRAIN_SKIP} FILAS")


# (6) DETECCIÓN EN FLUJO TIEMPO REAL

results = []
for i in range(len(X_scaled)):
    sample = X_scaled[i].reshape(1, -1)
    pred = clf.predict(sample)[0]
    score = clf.decision_function(sample)[0] * -1
    label = 1 if pred == -1 else 0

    results.append((i, label, score))

    if SHOW_INFO and i % 50 == 0:
        print(f"[ INFO ] REGISTRO {i}: SCORE={score:.4f} ANOMALÍA={label}")

    time.sleep(SLEEP_TIME)


# (7) GUARDADO DE RESULTADOS

df_results = pd.DataFrame(results, columns=['index', 'anomaly', 'anomaly_score'])
df_final = pd.concat([df.reset_index(drop=True), df_results[['anomaly', 'anomaly_score']]], axis=1)

df_final.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] RESULTADOS COMPLETOS EN '{OUTPUT_CSV}'")

if SAVE_ANOMALY_CSV:
    df_anomalies = df_final[df_final['anomaly'] == 1].copy()
    df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False)
    df_anomalies.to_csv(OUTPUT_IF_CSV, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] CSV SOLO ANOMALÍAS EN '{OUTPUT_IF_CSV}'")

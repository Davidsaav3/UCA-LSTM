import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from itertools import product

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/01_classification'
INPUT_CSV = os.path.join(RESULTS_FOLDER, '01_contaminated.csv')
OUTPUT_JSON = os.path.join(RESULTS_FOLDER, '01_if_best_params.json')  # PARAMS ÓPTIMOS

SHOW_INFO = True  # MOSTRAR INFO EN CONSOLA

# CONFIGURACIÓN DEL GRID
PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_samples': ['auto', 0.8],
    'contamination': [0.005, 0.01],
    'max_features': [0.8, 1.0],
    'bootstrap': [False]
}

# CARGAR Y ESCALAR DATASET
df = pd.read_csv(INPUT_CSV, low_memory=False)
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_cols])
if SHOW_INFO:
    print("[ INFO ] FEATURES ESCALADAS CON STANDARDSCALER")

# FUNCION PARA EVALUAR MODELO
def evaluate_if(params, X):
    clf = IsolationForest(
        n_estimators=params['n_estimators'],
        max_samples=params['max_samples'],
        contamination=params['contamination'],
        max_features=params['max_features'],
        bootstrap=params['bootstrap'],
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X)
    scores = clf.decision_function(X) * -1  # positivo = más anómalo
    return np.mean(scores)  # MÉTRICA: promedio del score

# EXPLORAR GRID
all_combinations = list(product(
    PARAM_GRID['n_estimators'],
    PARAM_GRID['max_samples'],
    PARAM_GRID['contamination'],
    PARAM_GRID['max_features'],
    PARAM_GRID['bootstrap']
))

best_score = -np.inf
best_params = None

for comb in all_combinations:
    params = {
        'n_estimators': comb[0],
        'max_samples': comb[1],
        'contamination': comb[2],
        'max_features': comb[3],
        'bootstrap': comb[4]
    }
    score = evaluate_if(params, X_scaled)
    if SHOW_INFO:
        print(f"[ INFO ] Probando {params} → Score: {score:.6f}")
    if score > best_score:
        best_score = score
        best_params = params

# GUARDAR PARAMETROS ÓPTIMOS
with open(OUTPUT_JSON, 'w') as f:
    json.dump(best_params, f, indent=4)
if SHOW_INFO:
    print(f"[ GUARDADO ] HIPERPARÁMETROS ÓPTIMOS EN '{OUTPUT_JSON}'")
    print(f"[ INFO ] MEJOR COMBINACIÓN: {best_params}")

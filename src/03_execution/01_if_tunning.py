import pandas as pd 
import numpy as np
import os
import json
from sklearn.ensemble import IsolationForest 
from sklearn.preprocessing import StandardScaler  
from itertools import product  
import sys
sys.stdout.reconfigure(encoding='utf-8')

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/01_classification'  # CARPETA RESULTADOS
INPUT_CSV = os.path.join(RESULTS_FOLDER, '01_contaminated.csv')  # CSV CONTAMINADO
OUTPUT_JSON = os.path.join(RESULTS_FOLDER, '01_if_params.json')  # PARAMS ÓPTIMOS

SHOW_INFO = True  # MOSTRAR INFO EN CONSOLA

# CONFIGURACIÓN DEL GRID DE HIPERPARÁMETROS
PARAM_GRID = {
    'n_estimators': [100, 200],          # NÚMERO DE ÁRBOLES
    'max_samples': ['auto', 0.8],        # MUESTRAS POR ÁRBOL
    'contamination': [0.005, 0.01],      # FRACCIÓN DE ANOMALÍAS
    'max_features': [0.8, 1.0],          # FRACCIÓN DE FEATURES POR ÁRBOL
    'bootstrap': [False]                  # BOOTSTRAP DESACTIVADO
}

# CARGAR Y ESCALAR DATASET
df = pd.read_csv(INPUT_CSV, low_memory=False)  # LEER CSV
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()  # SELECCIONAR COLUMNAS NUMÉRICAS
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(df[num_cols])  # ESCALAR FEATURES
if SHOW_INFO:
    print("[ INFO ] FEATURES ESCALADAS CON STANDARDSCALER")

# FUNCION PARA EVALUAR MODELO CON DETERMINADOS PARÁMETROS
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
    clf.fit(X)  # ENTRENAR MODELO
    scores = clf.decision_function(X) * -1  # SCORE POSITIVO = MÁS ANÓMALO
    return np.mean(scores)  # MÉTRICA PROMEDIO DEL SCORE

# EXPLORAR GRID DE HIPERPARÁMETROS
all_combinations = list(product(
    PARAM_GRID['n_estimators'],
    PARAM_GRID['max_samples'],
    PARAM_GRID['contamination'],
    PARAM_GRID['max_features'],
    PARAM_GRID['bootstrap']
))

best_score = -np.inf
best_params = None

# PROBAR TODAS LAS COMBINACIONES
for comb in all_combinations:
    params = {
        'n_estimators': comb[0],
        'max_samples': comb[1],
        'contamination': comb[2],
        'max_features': comb[3],
        'bootstrap': comb[4]
    }
    score = evaluate_if(params, X_scaled)  # EVALUAR MODELO
    if SHOW_INFO:
        print(f"[ INFO ] Probando {params} → Score: {score:.6f}")
    if score > best_score:  # ACTUALIZAR MEJOR COMBINACIÓN
        best_score = score
        best_params = params

# GUARDAR PARAMETROS ÓPTIMOS EN JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(best_params, f, indent=4)
if SHOW_INFO:
    print(f"[ GUARDADO ] HIPERPARÁMETROS ÓPTIMOS EN '{OUTPUT_JSON}'")
    print(f"[ INFO ] MEJOR COMBINACIÓN: {best_params}")

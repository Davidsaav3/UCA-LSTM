import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
from sklearn.preprocessing import StandardScaler  
from tensorflow.keras.models import load_model   
import sys
sys.stdout.reconfigure(encoding='utf-8')

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/02_prediction'  # CARPETA RESULTADOS
INPUT_CSV_1 = '../../results/02_preparation/infrastructure/realtime/lstm/05_variance.csv'  # CSV INFRA
INPUT_CSV_2 = '../../results/02_preparation/context/realtime/lstm/05_variance.csv'         # CSV CONTEXTO
MODEL_PATH = os.path.join(RESULTS_FOLDER, '02_lstm_model.keras')  # RUTA MODELO
OUTPUT_PRED_CSV = os.path.join(RESULTS_FOLDER, '02_lstm_predictions.csv')  # CSV SALIDA PREDICCIONES

TIMESTEPS = 10
SCALE_DATA = True
SHOW_INFO = True
TARGET_COLUMN = 'wifi_inal_sf_1_39'  # COLUMNA OBJETIVO

# CARGA DEL MODELO
model = load_model(MODEL_PATH)  # CARGAR MODELO ENTRENADO
if SHOW_INFO:
    print(f"[ INFO ] MODELO CARGADO DESDE '{MODEL_PATH}'")

expected_features = model.input_shape[-1]  # NÚMERO DE FEATURES QUE ESPERA EL MODELO
if SHOW_INFO:
    print(f"[ INFO ] EL MODELO ESPERA {expected_features} FEATURES")

# CARGAR LOS CSV
df_1 = pd.read_csv(INPUT_CSV_1, low_memory=False)  # LEER CSV INFRA
df_2 = pd.read_csv(INPUT_CSV_2, low_memory=False)  # LEER CSV CONTEXTO

# VALORES DE LA COLUMNA TARGET
target_col_values = df_1[TARGET_COLUMN] if TARGET_COLUMN in df_1.columns else pd.Series(range(df_1.shape[0]))

# LIMPIAR NaN e infinitos
df_1 = df_1.replace([np.inf, -np.inf], np.nan).fillna(0)  # REEMPLAZAR INF Y NAN
df_2 = df_2.replace([np.inf, -np.inf], np.nan).fillna(0)

# VERIFICAR filas
if df_1.shape[0] != df_2.shape[0]:
    raise ValueError(f"Los CSV no tienen el mismo número de filas: df_1={df_1.shape[0]}, df_2={df_2.shape[0]}")

# CONCATENAR CSV HORIZONTALMENTE
df = pd.concat([df_1, df_2], axis=1)
if SHOW_INFO:
    print(f"[ INFO ] DATASET COMBINADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# AJUSTE DE DIMENSIONES
current_features = df.shape[1]
if current_features < expected_features:
    for i in range(expected_features - current_features):
        df[f"missing_{i}"] = 0  # AGREGAR COLUMNAS FALTANTES
elif current_features > expected_features:
    df = df.iloc[:, :expected_features]  # RECORTAR COLUMNAS EXTRA

# ESCALADO
if SCALE_DATA:
    scaler = StandardScaler()                  # CREAR OBJETO ESCALADOR
    df_scaled = scaler.fit_transform(df.values)  # ESCALAR FEATURES
else:
    df_scaled = df.values.copy()  # USAR DATOS ORIGINALES

# PREPARAR CSV SALIDA
if os.path.exists(OUTPUT_PRED_CSV):
    os.remove(OUTPUT_PRED_CSV)  # ELIMINAR CSV EXISTENTE

with open(OUTPUT_PRED_CSV, 'w') as f:
    f.write(f'{TARGET_COLUMN},prediction\n')  # ESCRIBIR ENCABEZADO

# INICIALIZAR lista de predicciones con NaN para los primeros TIMESTEPS-1
predictions = [np.nan] * (TIMESTEPS - 1)

# GENERAR PREDICCIONES
for start_idx in range(df_scaled.shape[0] - TIMESTEPS + 1):
    end_idx = start_idx + TIMESTEPS
    X_seq = df_scaled[start_idx:end_idx].reshape(1, TIMESTEPS, expected_features)  # FORMATO 3D PARA LSTM
    pred_value = model.predict(X_seq, verbose=0)[0, 0]  # PREDICCIÓN
    predictions.append(pred_value)

# GUARDAR CSV
for idx in range(len(target_col_values)):
    with open(OUTPUT_PRED_CSV, 'a') as f:
        f.write(f"{target_col_values.iloc[idx]},{predictions[idx]}\n")  # GUARDAR VALOR Y PREDICCIÓN

if SHOW_INFO:
    print(f"[ GUARDADO ] Predicciones generadas con {len(predictions)} filas, igual que la entrada.")  # INFO FINAL

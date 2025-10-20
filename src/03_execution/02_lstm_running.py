import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/02_prediction'   # CARPETA DE RESULTADOS
INPUT_CSV = '../../results/02_preparation/infrastructure/realtime/lstm/05_variance.csv'  # CSV TEMPORAL
MODEL_PATH = os.path.join(RESULTS_FOLDER, '02_lstm_model.keras')  # MODELO ENTRENADO
OUTPUT_PRED_CSV = os.path.join(RESULTS_FOLDER, '02_lstm_predictions.csv')  # SALIDA DE PREDICCIONES

TIMESTEPS = 10              # LONGITUD DE SECUENCIA
SCALE_DATA = True           # ESCALAR FEATURES
SHOW_INFO = True            # MOSTRAR MENSAJES

# CARGAR MODELO LSTM
model = load_model(MODEL_PATH)
if SHOW_INFO:
    print(f"[ INFO ] MODELO CARGADO DESDE '{MODEL_PATH}'")

# OBTENER NÚMERO DE FEATURES ESPERADAS
expected_features = model.input_shape[-1]
if SHOW_INFO:
    print(f"[ INFO ] EL MODELO ESPERA {expected_features} FEATURES")

# CARGAR DATASET EN TIEMPO REAL
df = pd.read_csv(INPUT_CSV, low_memory=False)
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# AJUSTAR DIMENSIONES SI SON DISTINTAS
current_features = df.shape[1]
if current_features < expected_features:
    # AÑADIR COLUMNAS NULAS SI FALTAN FEATURES
    diff = expected_features - current_features
    for i in range(diff):
        df[f"missing_{i}"] = 0
    if SHOW_INFO:
        print(f"[ WARNING ] FALTABAN {diff} FEATURES → SE AÑADIERON COLUMNAS NULAS")
elif current_features > expected_features:
    # RECORTAR COLUMNAS SOBRANTES
    df = df.iloc[:, :expected_features]
    if SHOW_INFO:
        print(f"[ WARNING ] SOBRABAN {current_features - expected_features} FEATURES → SE RECORTARON")

# ESCALADO DE FEATURES
if SCALE_DATA:
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.values)
    if SHOW_INFO:
        print("[ INFO ] FEATURES ESCALADAS CON STANDARDSCALER")
else:
    df_scaled = df.values.copy()

# SIMULACIÓN DE ENTRADA EN TIEMPO REAL
predictions_list = []  # ALMACENA PREDICCIONES

# RECORRER EL DATASET SECUENCIA A SECUENCIA
for start_idx in range(0, df_scaled.shape[0] - TIMESTEPS + 1):
    end_idx = start_idx + TIMESTEPS
    X_seq = df_scaled[start_idx:end_idx].reshape(1, TIMESTEPS, expected_features)  # FORMATO 3D
    pred = model.predict(X_seq, verbose=0)  # PREDICCIÓN
    predictions_list.append(pred[0, 0])      # GUARDAR PREDICCIÓN
    if SHOW_INFO and start_idx % 100 == 0:
        print(f"[ INFO ] SECUENCIA {start_idx} → PREDICCIÓN: {pred[0,0]:.4f}")

# GUARDAR PREDICCIONES EN CSV
df_pred = pd.DataFrame(predictions_list, columns=['prediction'])
df_pred.to_csv(OUTPUT_PRED_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] PREDICCIONES EN '{OUTPUT_PRED_CSV}'")

# COMENTARIO FINAL
# EL SCRIPT SIMULA UNA ENTRADA TEMPORAL EN TIEMPO REAL
# AJUSTA AUTOMÁTICAMENTE LAS DIMENSIONES Y GENERA PREDICCIONES SECUENCIALES

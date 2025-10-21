import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/02_prediction'
INPUT_CSV_1 = '../../results/02_preparation/infrastructure/realtime/lstm/05_variance.csv'
INPUT_CSV_2 = '../../results/02_preparation/context/realtime/lstm/05_variance.csv'
MODEL_PATH = os.path.join(RESULTS_FOLDER, '02_lstm_model.keras')
OUTPUT_PRED_CSV = os.path.join(RESULTS_FOLDER, '02_lstm_predictions.csv')

TIMESTEPS = 10
SCALE_DATA = True
SHOW_INFO = True
TARGET_COLUMN = 'agua_map07020001'

# CARGA DEL MODELO
model = load_model(MODEL_PATH)
if SHOW_INFO:
    print(f"[ INFO ] MODELO CARGADO DESDE '{MODEL_PATH}'")

expected_features = model.input_shape[-1]
if SHOW_INFO:
    print(f"[ INFO ] EL MODELO ESPERA {expected_features} FEATURES")

# CARGAR LOS CSV
df_1 = pd.read_csv(INPUT_CSV_1, low_memory=False)
df_2 = pd.read_csv(INPUT_CSV_2, low_memory=False)

# VALORES DE LA COLUMNA TARGET
target_col_values = df_1[TARGET_COLUMN] if TARGET_COLUMN in df_1.columns else pd.Series(range(df_1.shape[0]))

# LIMPIAR NaN e infinitos
df_1 = df_1.replace([np.inf, -np.inf], np.nan).fillna(0)
df_2 = df_2.replace([np.inf, -np.inf], np.nan).fillna(0)

# VERIFICAR filas
if df_1.shape[0] != df_2.shape[0]:
    raise ValueError(f"Los CSV no tienen el mismo número de filas: df_1={df_1.shape[0]}, df_2={df_2.shape[0]}")

# CONCATENAR
df = pd.concat([df_1, df_2], axis=1)
if SHOW_INFO:
    print(f"[ INFO ] DATASET COMBINADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# AJUSTE DE DIMENSIONES
current_features = df.shape[1]
if current_features < expected_features:
    for i in range(expected_features - current_features):
        df[f"missing_{i}"] = 0
elif current_features > expected_features:
    df = df.iloc[:, :expected_features]

# ESCALADO
if SCALE_DATA:
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.values)
else:
    df_scaled = df.values.copy()

# PREPARAR CSV
if os.path.exists(OUTPUT_PRED_CSV):
    os.remove(OUTPUT_PRED_CSV)

with open(OUTPUT_PRED_CSV, 'w') as f:
    f.write(f'{TARGET_COLUMN},prediction\n')

# INICIALIZAR lista de predicciones con NaN para los primeros TIMESTEPS-1
predictions = [np.nan] * (TIMESTEPS - 1)

# GENERAR PREDICCIONES
for start_idx in range(df_scaled.shape[0] - TIMESTEPS + 1):
    end_idx = start_idx + TIMESTEPS
    X_seq = df_scaled[start_idx:end_idx].reshape(1, TIMESTEPS, expected_features)
    pred_value = model.predict(X_seq, verbose=0)[0, 0]
    predictions.append(pred_value)

# GUARDAR CSV
for idx in range(len(target_col_values)):
    with open(OUTPUT_PRED_CSV, 'a') as f:
        f.write(f"{target_col_values.iloc[idx]},{predictions[idx]}\n")

if SHOW_INFO:
    print(f"[ GUARDADO ] Predicciones generadas con {len(predictions)} filas, igual que la entrada.")

import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/02_prediction'
INPUT_CSV_1 = '../../results/02_preparation/infrastructure/realtime/lstm/05_variance.csv'
INPUT_CSV_2 = '../../results/02_preparation/context/realtime/lstm/05_variance.csv'
MODEL_PATH = os.path.join(RESULTS_FOLDER, '02_lstm_model.keras')
SCALER_PATH = os.path.join(RESULTS_FOLDER, '02_lstm_scaler.gz')
TARGET_STATS_PATH = os.path.join(RESULTS_FOLDER, '02_target_stats.npz')
OUTPUT_PRED_CSV = os.path.join(RESULTS_FOLDER, '02_lstm_predictions.csv')
OUTPUT_DIAGNOSIS_PLOT = os.path.join(RESULTS_FOLDER, '02_realtime_diagnosis.png')

TIMESTEPS = 120  # Igual que entrenamiento
FORECAST_HORIZON = 10
SHOW_INFO = True
TARGET_COLUMN = 'wifi_inal_sf_1_39'

# CARGA
model = load_model(MODEL_PATH)
print(f"[ INFO ] MODELO CARGADO DESDE '{MODEL_PATH}'")

expected_features = model.input_shape[-1]
print(f"[ INFO ] EL MODELO ESPERA {expected_features} FEATURES")

scaler = joblib.load(SCALER_PATH)
print(f"[ INFO ] Scaler cargado desde '{SCALER_PATH}'")

stats = np.load(TARGET_STATS_PATH)
target_mean = stats['mean']
target_std = stats['std']
print(f"[ INFO ] Estadísticas del target: mean={target_mean:.4f}, std={target_std:.4f}")

# DATOS
df_1 = pd.read_csv(INPUT_CSV_1, low_memory=False)
df_2 = pd.read_csv(INPUT_CSV_2, low_memory=False)
target_col_values = df_1[TARGET_COLUMN].values if TARGET_COLUMN in df_1.columns else np.zeros(df_1.shape[0])

df_1 = df_1.replace([np.inf, -np.inf], np.nan).fillna(0)
df_2 = df_2.replace([np.inf, -np.inf], np.nan).fillna(0)

if df_1.shape[0] != df_2.shape[0]:
    raise ValueError("Filas diferentes")

df = pd.concat([df_1, df_2], axis=1)
print(f"[ INFO ] DATASET COMBINADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# AJUSTE FEATURES
current_features = df.shape[1]
if current_features < expected_features:
    for i in range(expected_features - current_features):
        df[f"missing_{i}"] = 0
elif current_features > expected_features:
    df = df.iloc[:, :expected_features]

df_scaled = scaler.transform(df.values.astype(float))

# CSV SALIDA
if os.path.exists(OUTPUT_PRED_CSV):
    os.remove(OUTPUT_PRED_CSV)
header = f'{TARGET_COLUMN}' + ''.join([f',pred_step{j+1}' for j in range(FORECAST_HORIZON)])
with open(OUTPUT_PRED_CSV, 'w') as f:
    f.write(header + '\n')

# PREDICCIONES
predictions_original = np.full((len(df_scaled), FORECAST_HORIZON), np.nan)

n_samples = len(df_scaled) - TIMESTEPS + 1
if n_samples > 0:
    X_windows = np.zeros((n_samples, TIMESTEPS, expected_features))
    for i in range(n_samples):
        X_windows[i] = df_scaled[i:i + TIMESTEPS]
    predicted_scaled = model.predict(X_windows, verbose=0)  # (n_samples, 10)
    predicted_original_temp = predicted_scaled * target_std + target_mean

    # Rellenar predicciones sin errores de shape
    for i in range(n_samples):
        start_idx = i + TIMESTEPS
        num_steps = min(FORECAST_HORIZON, len(predictions_original) - start_idx)
        if num_steps > 0:
            predictions_original[start_idx:start_idx + num_steps, :num_steps] = predicted_original_temp[i, :num_steps]

# Primer paso para diagnóstico
pred_first_step = np.full(len(target_col_values), np.nan)
for i in range(n_samples):
    idx = i + TIMESTEPS
    if idx < len(pred_first_step):
        pred_first_step[idx] = predictions_original[idx, 0]

# MÉTRICAS
valid_mask = ~np.isnan(pred_first_step)
if np.any(valid_mask):
    mse_lstm = np.mean((target_col_values[valid_mask] - pred_first_step[valid_mask]) ** 2)
    mse_baseline = np.mean((target_col_values[valid_mask] - target_mean) ** 2)
    print(f"[ DIAGNÓSTICO REALTIME ] MSE LSTM (primer paso): {mse_lstm:.6f}")
    print(f"[ DIAGNÓSTICO REALTIME ] MSE Baseline: {mse_baseline:.6f}")

# GRÁFICA
plot_n = min(500, len(target_col_values))
plt.figure(figsize=(14, 6))
plt.plot(target_col_values[:plot_n], label='Real (realtime)', color='blue', linewidth=2)
plt.plot(pred_first_step[:plot_n], label='Predicción LSTM (primer paso)', color='red', alpha=0.8, linewidth=2)
plt.axhline(y=target_mean, color='green', linestyle='--', label=f'Baseline: media = {target_mean:.4f}')
plt.title('Diagnóstico en Realtime: Real vs Predicción LSTM (primer paso) vs Baseline')
plt.xlabel('Índice de Registro')
plt.ylabel('Valor original')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIAGNOSIS_PLOT, dpi=150)
print(f"[ GUARDADO ] Gráfica realtime en '{OUTPUT_DIAGNOSIS_PLOT}'")
plt.close()

# GUARDAR CSV
for idx in range(len(target_col_values)):
    row = [target_col_values[idx]] + predictions_original[idx].tolist()
    row_str = ','.join([str(v) if not np.isnan(v) else '' for v in row])
    with open(OUTPUT_PRED_CSV, 'a') as f:
        f.write(row_str + '\n')

print(f"[ GUARDADO ] Predicciones multi-step en '{OUTPUT_PRED_CSV}'")
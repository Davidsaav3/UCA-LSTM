import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import joblib  # Para guardar el scaler
import matplotlib.pyplot as plt  # Para diagnóstico
import tensorflow as tf
import sys
sys.stdout.reconfigure(encoding='utf-8')

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/02_prediction/' # CARPETA PRINCIPAL
INPUT_CSV_1 = '../../results/02_preparation/infrastructure/historical/lstm/05_variance.csv' # DATASET HISTÓRICO INFRA
INPUT_CSV_2 = '../../results/02_preparation/context/historical/lstm/05_variance.csv' # DATASET HISTÓRICO CONTEXTO
OUTPUT_MODEL = os.path.join(RESULTS_FOLDER, '02_lstm_model.keras') # RUTA MODELO ENTRENADO
OUTPUT_SCALER = os.path.join(RESULTS_FOLDER, '02_lstm_scaler.gz')  # Ruta para guardar el scaler
OUTPUT_HISTORY_CSV = os.path.join(RESULTS_FOLDER, '02_lstm_history.csv') # RUTA HISTORIAL ENTRENAMIENTO
OUTPUT_DIAGNOSIS_PLOT = os.path.join(RESULTS_FOLDER, '02_train_diagnosis.png')  # Gráfica de diagnóstico en train
TARGET_STATS_PATH = os.path.join(RESULTS_FOLDER, '02_target_stats.npz')  # Estadísticas del target para inferencia

# PARÁMETROS LSTM
TIMESTEPS = 120  # Ventana más larga para mejor anticipación de cambios
# LONGITUD DE LA SECUENCIA TEMPORAL QUE EL MODELO USARÁ PARA APRENDER PATRONES
# CADA EJEMPLO DE ENTRENAMIENTO CONTENDRÁ 'TIMESTEPS' FILAS CONSECUTIVAS
# VALOR MÁS GRANDE PERMITE CAPTURAR DEPENDENCIAS TEMPORALES MÁS LARGAS,
# PERO AUMENTA LA COMPLEJIDAD Y EL TIEMPO DE ENTRENAMIENTO
FORECAST_HORIZON = 10  # Número de pasos futuros a predecir (multi-step)
FEATURES = None
# NÚMERO DE CARACTERÍSTICAS POR TIMESTEP
# NONE = AUTOMÁTICO, USARÁ TODAS LAS COLUMNAS DEL DATASET
# SI SE ESPECIFICA, SOLO ESA CANTIDAD DE FEATURES SERÁ UTILIZADA
TARGET_COLUMN = 'wifi_inal_sf_1_39'  # Variable objetivo actual (con patrón claro)
# COLUMNA OBJETIVO QUE EL MODELO INTENTA PREDECIR
# -1 = ÚLTIMA COLUMNA DEL DATAFRAME
# SE PUEDE INDICAR EL ÍNDICE DE CUALQUIER COLUMNA SI SE DESEA PREDECIR OTRA
LSTM_UNITS = 64
# NÚMERO DE NEURONAS EN LA CAPA LSTM
# MAYOR NÚMERO = MAYOR CAPACIDAD PARA CAPTURAR PATRONES TEMPORALES COMPLEJOS
# PERO AUMENTA EL RIESGO DE OVERFITTING Y TIEMPO DE ENTRENAMIENTO
DENSE_UNITS = 32
# NÚMERO DE NEURONAS EN LA CAPA DENSAMENTE CONECTADA
# AYUDA A TRANSFORMAR LA REPRESENTACIÓN EXTRAÍDA POR LSTM ANTES DE LA SALIDA
# INFLUYE EN LA CAPACIDAD DEL MODELO PARA AJUSTAR PATRONES NO LINEALES

# HIPERPARÁMETROS ENTRENAMIENTO
BATCH_SIZE = 64
# CANTIDAD DE EJEMPLOS QUE EL MODELO PROCESA ANTES DE ACTUALIZAR LOS PESOS
# BATCH PEQUEÑO = ENTRENAMIENTO MÁS RUIDOSO, MEJOR GENERALIZACIÓN
# BATCH GRANDE = ENTRENAMIENTO MÁS ESTABLE, MENOR VARIABILIDAD, MÁS RECURSOS
EPOCHS = 100  # Aumentado para dar más tiempo al aprendizaje
# NÚMERO DE VECES QUE TODO EL DATASET SE UTILIZA PARA ENTRENAR EL MODELO
# MÁS EPOCHS = MAYOR AJUSTE, PERO POSIBLE OVERFITTING
# MENOS EPOCHS = POSIBLE SUBAJUSTE
VALIDATION_SPLIT = 0.15
# PORCIÓN DEL DATASET QUE SE USA COMO VALIDACIÓN
# 0.0 = TODO EL DATASET PARA ENTRENAMIENTO, NO SE MONITOREA VAL_LOSS
# 0.1-0.2 COMÚN PARA EVALUAR RENDIMIENTO DURANTE EL ENTRENAMIENTO
EARLY_STOPPING = True
# ACTIVAR DETENCIÓN TEMPRANA PARA EVITAR SOBREAJUSTE
# EL ENTRENAMIENTO SE DETIENE SI LA VALIDACIÓN NO MEJORA DURANTE 'PATIENCE' EPOCHS
PATIENCE = 15  # Aumentado para dar más margen
# NÚMERO DE EPOCHS QUE SE ESPERA SIN MEJORA EN VALIDACIÓN ANTES DE DETENER
# SOLO SE USA SI EARLY_STOPPING = TRUE

# FLAGS DE CONTROL
SHOW_INFO = True
SCALE_DATA = True
SAVE_MODEL = True
SAVE_HISTORY = True

# CREAR CARPETA DE RESULTADOS SI NO EXISTE
os.makedirs(RESULTS_FOLDER, exist_ok=True)
print(f"[ INFO ] CARPETA '{RESULTS_FOLDER}' CREADA SI NO EXISTÍA")

# CARGAR LOS DOS DATASETS
df_1 = pd.read_csv(INPUT_CSV_1, low_memory=False) # LEER CSV INFRA
df_2 = pd.read_csv(INPUT_CSV_2, low_memory=False) # LEER CSV CONTEXTO

# VERIFICAR QUE TENGAN MISMO NÚMERO DE FILAS
if df_1.shape[0] != df_2.shape[0]:
    raise ValueError(f"Los CSV no tienen el mismo número de filas: df_1={df_1.shape[0]}, df_2={df_2.shape[0]}")

# CONCATENAR HORIZONTALMENTE
df = pd.concat([df_1.reset_index(drop=True), df_2.reset_index(drop=True)], axis=1) # UNIR DATASETS POR COLUMNAS
print(f"[ INFO ] DATASET COMBINADO HORIZONTALMENTE: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# SELECCIONAR SOLO COLUMNAS NUMÉRICAS PARA FEATURES
df_numeric = df.select_dtypes(include=[np.number]).copy()
dropped_cols = list(set(df.columns) - set(df_numeric.columns))
if len(dropped_cols) > 0:
    print(f"[ AVISO ] Se descartaron columnas no numéricas: {dropped_cols}")

# Si FEATURES is None, usar todas las columnas numéricas
if FEATURES is None:
    FEATURES = df_numeric.shape[1]
    print(f"[ INFO ] FEATURES = {FEATURES} (todas las columnas numéricas)")

# Resolver TARGET_COLUMN
if isinstance(TARGET_COLUMN, int):
    target_idx = TARGET_COLUMN if TARGET_COLUMN >= 0 else df_numeric.shape[1] + TARGET_COLUMN
    target_col_name = df_numeric.columns[target_idx]
else:
    target_col_name = TARGET_COLUMN
print(f"[ INFO ] TARGET_COLUMN resuelto: '{target_col_name}'")

# VARIABLES para ventanas deslizantes
values = df_numeric.values.astype(float)
n_rows = values.shape[0]
n_samples = n_rows - TIMESTEPS - FORECAST_HORIZON + 1
if n_samples <= 0:
    raise ValueError(f"No hay suficientes filas para TIMESTEPS={TIMESTEPS} y horizonte={FORECAST_HORIZON}")

# Construir X (ventanas) e y (secuencia de FORECAST_HORIZON valores futuros - multi-step)
X = np.zeros((n_samples, TIMESTEPS, FEATURES), dtype=float)
y = np.zeros((n_samples, FORECAST_HORIZON), dtype=float)
target_idx_in_numeric = list(df_numeric.columns).index(target_col_name)

for i in range(n_samples):
    X[i] = values[i:i + TIMESTEPS, :FEATURES]
    y[i] = values[i + TIMESTEPS : i + TIMESTEPS + FORECAST_HORIZON, target_idx_in_numeric]

print(f"[ INFO ] Ventanas creadas (multi-step seq2seq): X.shape={X.shape}, y.shape={y.shape}")

# Guardar estadísticas del target
target_mean = np.mean(y)
target_std = np.std(y)
np.savez(TARGET_STATS_PATH, mean=target_mean, std=target_std)
print(f"[ GUARDADO ] Estadísticas del target (mean={target_mean:.4f}, std={target_std:.4f}) en '{TARGET_STATS_PATH}'")

# ESCALADO DE FEATURES
if SCALE_DATA:
    scaler = StandardScaler()
    X_2d = X.reshape(-1, FEATURES)
    X_2d_scaled = scaler.fit_transform(X_2d)
    X_scaled = X_2d_scaled.reshape(n_samples, TIMESTEPS, FEATURES)
    joblib.dump(scaler, OUTPUT_SCALER)
    print(f"[ INFO ] FEATURES escaladas con StandardScaler y scaler guardado en '{OUTPUT_SCALER}'")
else:
    X_scaled = X.copy()

# MODELO LSTM seq2seq (encoder-decoder multi-step)
model = Sequential()
model.add(Input(shape=(TIMESTEPS, FEATURES)))
model.add(Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, recurrent_dropout=0.3, kernel_regularizer=l2(1e-4))))
model.add(Bidirectional(LSTM(64, return_sequences=False, recurrent_dropout=0.3, kernel_regularizer=l2(1e-4))))
model.add(RepeatVector(FORECAST_HORIZON))
model.add(Bidirectional(LSTM(32, return_sequences=True, recurrent_dropout=0.3, kernel_regularizer=l2(1e-4))))
model.add(TimeDistributed(Dense(DENSE_UNITS, activation='relu')))
model.add(TimeDistributed(Dense(1)))
model.add(tf.keras.layers.Reshape((FORECAST_HORIZON,)))  # Salida (batch, 10)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# CALLBACKS
callbacks = []
if EARLY_STOPPING and VALIDATION_SPLIT > 0.0:
    callbacks.append(EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True))
    print(f"[ INFO ] EarlyStopping habilitado con patience={PATIENCE}")

# ENTRENAMIENTO
history = model.fit(
    X_scaled, y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1 if SHOW_INFO else 0
)

# DIAGNÓSTICO EN ENTRENAMIENTO (primer paso)
y_pred_train = model.predict(X_scaled, verbose=0)[:, 0]
y_real_first = y[:, 0]

mse_lstm = np.mean((y_real_first - y_pred_train) ** 2)
mse_mean_baseline = np.mean((y_real_first - target_mean) ** 2)
print(f"[ DIAGNÓSTICO ] MSE LSTM (primer paso) en train: {mse_lstm:.6f}")
print(f"[ DIAGNÓSTICO ] MSE Baseline (predecir media): {mse_mean_baseline:.6f}")

# Gráfica de diagnóstico
plot_n = min(500, len(y_real_first))
plt.figure(figsize=(12, 6))
plt.plot(y_real_first[:plot_n], label='Real (train, primer paso)', color='blue')
plt.plot(y_pred_train[:plot_n], label='Predicción LSTM (train, primer paso)', color='red', alpha=0.8)
plt.axhline(y=target_mean, color='green', linestyle='--', label=f'Baseline: media = {target_mean:.4f}')
plt.title('Diagnóstico en Entrenamiento (primer paso): Real vs Predicción LSTM vs Baseline')
plt.xlabel('Índice')
plt.ylabel('Valor escalado')
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_DIAGNOSIS_PLOT)
print(f"[ GUARDADO ] Gráfica de diagnóstico en entrenamiento: '{OUTPUT_DIAGNOSIS_PLOT}'")
plt.close()

# GUARDAR
if SAVE_HISTORY:
    pd.DataFrame(history.history).to_csv(OUTPUT_HISTORY_CSV, index=False)
    print(f"[ GUARDADO ] HISTORIAL ENTRENAMIENTO EN '{OUTPUT_HISTORY_CSV}'")

if SAVE_MODEL:
    model.save(OUTPUT_MODEL)
    print(f"[ GUARDADO ] MODELO ENTRENADO EN '{OUTPUT_MODEL}'")

print("[ INFO ] ENTRENAMIENTO COMPLETADO (multi-step seq2seq).")
print(f"[ INFO ] El modelo ahora predice los próximos {FORECAST_HORIZON} pasos.")
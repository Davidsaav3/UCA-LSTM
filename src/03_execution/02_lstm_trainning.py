import pandas as pd                  # PARA MANEJO DE DATAFRAMES
import numpy as np                   # PARA MANIPULACIÓN NUMÉRICA
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # DESACTIVA OPTIMIZACIONES ONE-DNN
from sklearn.preprocessing import StandardScaler  # ESCALADO DE FEATURES
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/02_prediction/'                                    # CARPETA PRINCIPAL
INPUT_CSV_1 = '../../results/02_preparation/infrastructure/historical/lstm/05_variance.csv'  # DATASET HISTÓRICO
INPUT_CSV_2 = '../../results/02_preparation/context/historical/lstm/05_variance.csv'  # DATASET HISTÓRICO
OUTPUT_MODEL = os.path.join(RESULTS_FOLDER, '02_lstm_model.keras') # RUTA MODELO ENTRENADO
OUTPUT_HISTORY_CSV = os.path.join(RESULTS_FOLDER, '02_lstm_history.csv') # RUTA HISTORIAL ENTRENAMIENTO

# PARÁMETROS LSTM
TIMESTEPS = 10              # LONGITUD DE SECUENCIA
FEATURES = None             # NÚMERO DE FEATURES, NONE = AUTOMÁTICO
TARGET_COLUMN = -1          # COLUMNA OBJETIVO, -1 = ÚLTIMA
LSTM_UNITS = 50             # NEURONAS CAPTURA PATRONES TEMPORALES
DENSE_UNITS = 32            # NEURONAS CAPA DENSAMENTE CONECTADA
OUTPUT_UNITS = 1            # SALIDA PARA REGRESIÓN

# HIPERPARÁMETROS ENTRENAMIENTO
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.0      # USO COMPLETO DE DATOS PARA ENTRENAMIENTO
EARLY_STOPPING = False
PATIENCE = 5

# FLAGS DE CONTROL
SHOW_INFO = True
SCALE_DATA = True
SAVE_MODEL = True
SAVE_HISTORY = True

# CARGAR LOS DOS DATASETS
df_1 = pd.read_csv(INPUT_CSV_1, low_memory=False)    # LEER PRIMER CSV DE ENTRADA
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df_1.shape[0]} FILAS, {df_1.shape[1]} COLUMNAS")

df_2 = pd.read_csv(INPUT_CSV_2, low_memory=False)  # LEER SEGUNDO CSV DE ENTRADA
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df_2.shape[0]} FILAS, {df_2.shape[1]} COLUMNAS")

# UNIR LOS DATASETS UNO DEBAJO DEL OTRO
df = pd.concat([df_1, df_2], axis=0, ignore_index=True)  # CONCATENA Y REINICIA ÍNDICES
if SHOW_INFO:
    print(f"[ INFO ] DATASET COMBINADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# CONFIGURAR FEATURES Y TARGET
if FEATURES is None:
    FEATURES = df.shape[1]  # USAR TODAS LAS COLUMNAS SI NO SE ESPECIFICA

if TARGET_COLUMN == -1:
    y = df.iloc[TIMESTEPS-1::TIMESTEPS, -1].values  # ÚLTIMA COLUMNA
else:
    y = df.iloc[TIMESTEPS-1::TIMESTEPS, TARGET_COLUMN].values

# REDIMENSIONAR DATASET PARA LSTM
num_sequences = df.shape[0] // TIMESTEPS  # CALCULAR NÚMERO DE SECUENCIAS COMPLETAS
trimmed_rows = num_sequences * TIMESTEPS  # FILAS AJUSTADAS PARA SECUENCIAS
df_trimmed = df.iloc[:trimmed_rows]       # RECORTAR DATASET
if SHOW_INFO:
    print(f"[ INFO ] DATASET RECORTADO A {trimmed_rows} FILAS PARA LSTM")

X = df_trimmed.values.reshape(num_sequences, TIMESTEPS, FEATURES)  # FORMATO 3D PARA LSTM
if SHOW_INFO:
    print(f"[ INFO ] DATOS REDIMENSIONADOS PARA LSTM: {X.shape}")

# ESCALADO DE FEATURES
if SCALE_DATA:
    scaler = StandardScaler()                        # CREAR OBJETO ESCALADOR
    X_reshaped = X.reshape(-1, FEATURES)            # APLANAR PARA ESCALADO
    X_scaled = scaler.fit_transform(X_reshaped)     # ESCALAR FEATURES
    X_scaled = X_scaled.reshape(-1, TIMESTEPS, FEATURES)  # VOLVER A 3D
    if SHOW_INFO:
        print("[ INFO ] FEATURES ESCALADAS CON STANDARDSCALER")
else:
    X_scaled = X.copy()  # USAR DATOS ORIGINALES

# CREAR MODELO LSTM
model = Sequential()
model.add(Input(shape=(TIMESTEPS, FEATURES)))       # INPUT SECUENCIAL
model.add(LSTM(LSTM_UNITS, return_sequences=False)) # CAPA LSTM, ÚLTIMO ESTADO
model.add(Dense(DENSE_UNITS, activation='relu'))    # CAPA DENSAMENTE CONECTADA
model.add(Dense(OUTPUT_UNITS, activation='linear')) # CAPA DE SALIDA LINEAL

model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # COMPILAR MODELO
if SHOW_INFO:
    model.summary()  # RESUMEN DEL MODELO

# CONFIGURAR CALLBACKS
callbacks = []
if EARLY_STOPPING:
    callbacks.append(EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True))

# ENTRENAMIENTO
history = model.fit(
    X_scaled, y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1 if SHOW_INFO else 0
)

# GUARDAR HISTORIAL ENTRENAMIENTO
if SAVE_HISTORY:
    hist_df = pd.DataFrame(history.history)  # CONVERTIR HISTORIAL A DATAFRAME
    hist_df.to_csv(OUTPUT_HISTORY_CSV, index=False)  # GUARDAR CSV
    if SHOW_INFO:
        print(f"[ GUARDADO ] HISTORIAL ENTRENAMIENTO EN '{OUTPUT_HISTORY_CSV}'")

# GUARDAR MODELO ENTRENADO
if SAVE_MODEL:
    model.save(OUTPUT_MODEL)  # GUARDAR MODELO EN FORMATO .keras
    if SHOW_INFO:
        print(f"[ GUARDADO ] MODELO ENTRENADO EN '{OUTPUT_MODEL}'")

# COMENTARIO FINAL
# MODELO ENTRENADO CAPTURA PATRONES TEMPORALES
# SE USARÁ EN 02_lstm_realtime.py PARA PREDICCIONES EN TIEMPO REAL

import pandas as pd           # PARA MANEJO DE DATAFRAMES 
import numpy as np            # PARA MANIPULACIÓN NUMÉRICA
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # DESACTIVA OPTIMIZACIONES ONE-DNN
from sklearn.preprocessing import StandardScaler  # ESCALADO DE FEATURES
from tensorflow.keras.models import Sequential    # MODELO SECUENCIAL
from tensorflow.keras.layers import LSTM, Dense, Input  # CAPAS LSTM, DENSAS E INPUT
from tensorflow.keras.callbacks import EarlyStopping  # PARADA TEMPRANA

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results'                        # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  

INPUT_CSV = '../../results/lstm_context_historical/05_variance.csv'   # CSV DE ENTRADA CON SECUENCIAS
OUTPUT_MODEL = os.path.join(EXECUTION_FOLDER, '02_lstm_model.keras')  # MODELO ENTRENADO
OUTPUT_PRED_CSV = os.path.join(EXECUTION_FOLDER, '02_lstm_predictions.csv')  # CSV PREDICCIONES
OUTPUT_HISTORY_CSV = os.path.join(EXECUTION_FOLDER, '02_lstm_history.csv')   # CSV HISTORIAL

# PARÁMETROS DEL LSTM
TIMESTEPS = 10               # LONGITUD DE SECUENCIA
FEATURES = None              # NÚMERO DE FEATURES, NONE=AUTOMÁTICO
TARGET_COLUMN = -1           # COLUMNA OBJETIVO, -1=ÚLTIMA
LSTM_UNITS = 50              # NEURONAS CAPTURA PATRONES TEMPORALES
DENSE_UNITS = 32             # NEURONAS CAPA DENSAMENTE CONECTADA
OUTPUT_UNITS = 1             # SALIDA PARA REGRESIÓN

# HIPERPARÁMETROS DE ENTRENAMIENTO
BATCH_SIZE = 32              # TAMAÑO DEL LOTE
EPOCHS = 20                  # NÚMERO DE ÉPOCAS
VALIDATION_SPLIT = 0.0       # USO COMPLETO DE DATOS PARA ENTRENAMIENTO
LEARNING_RATE = 0.001        # TASA DE APRENDIZAJE ADAM
EARLY_STOPPING = False       # PARADA TEMPRANA
PATIENCE = 5                 # PACIENCIA SI SE ACTIVA

# FLAGS DE CONTROL
SHOW_INFO = True              # MOSTRAR MENSAJES
SCALE_DATA = True             # ESCALAR FEATURES
SAVE_MODEL = True             # GUARDAR MODELO
SAVE_PREDICTIONS = True       # GUARDAR PREDICCIONES
SAVE_HISTORY = True           # GUARDAR HISTORIAL

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV, low_memory=False)  # LEER CSV
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# CONFIGURAR FEATURES Y TARGET
if FEATURES is None:
    FEATURES = df.shape[1]  # USAR TODAS LAS COLUMNAS

# EXTRAER TARGET DINÁMICO
if TARGET_COLUMN == -1:
    y = df.iloc[TIMESTEPS-1::TIMESTEPS, -1].values  # ÚLTIMA COLUMNA
else:
    y = df.iloc[TIMESTEPS-1::TIMESTEPS, TARGET_COLUMN].values

# REDIMENSIONAMIENTO
num_sequences = df.shape[0] // TIMESTEPS
trimmed_rows = num_sequences * TIMESTEPS
df_trimmed = df.iloc[:trimmed_rows]
if SHOW_INFO:
    print(f"[ INFO ] DATASET RECORTADO A {trimmed_rows} FILAS PARA LSTM")

X = df_trimmed.values.reshape(num_sequences, TIMESTEPS, FEATURES)  # FORMATO 3D
if SHOW_INFO:
    print(f"[ INFO ] DATOS REDIMENSIONADOS PARA LSTM: {X.shape}")

# ESCALADO DE FEATURES
if SCALE_DATA:
    scaler = StandardScaler()                        
    X_reshaped = X.reshape(-1, FEATURES)            # APLANAR PARA ESCALADO
    X_scaled = scaler.fit_transform(X_reshaped)     
    X_scaled = X_scaled.reshape(-1, TIMESTEPS, FEATURES)  # VOLVER A 3D
    if SHOW_INFO:
        print("[ INFO ] FEATURES ESCALADAS CON STANDARDSCALER")
else:
    X_scaled = X.copy()

# CREAR MODELO LSTM
model = Sequential()
model.add(Input(shape=(TIMESTEPS, FEATURES)))          # INPUT SECUENCIAL
model.add(LSTM(LSTM_UNITS, return_sequences=False))    # CAPA LSTM, ÚLTIMO ESTADO
model.add(Dense(DENSE_UNITS, activation='relu'))       # CAPA DENSAMENTE CONECTADA
model.add(Dense(OUTPUT_UNITS, activation='linear'))   # CAPA DE SALIDA LINEAL

model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # COMPILAR MODELO
if SHOW_INFO:
    model.summary()  # RESUMEN DEL MODELO

# CONFIGURAR CALLBACKS
callbacks = []
if EARLY_STOPPING:
    callbacks.append(EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True))

# ENTRENAMIENTO (LSTM TRAINING)
history = model.fit(
    X_scaled, y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1 if SHOW_INFO else 0
)

# GUARDAR HISTORIAL
if SAVE_HISTORY:
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(OUTPUT_HISTORY_CSV, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] HISTORIAL DE ENTRENAMIENTO EN '{OUTPUT_HISTORY_CSV}'")

# GUARDAR MODELO ENTRENADO
if SAVE_MODEL:
    model.save(OUTPUT_MODEL)
    if SHOW_INFO:
        print(f"[ GUARDADO ] MODELO ENTRENADO EN '{OUTPUT_MODEL}'")

# PREDICCIONES (LSTM RUNNING)
predictions = model.predict(X_scaled, verbose=0)
df_pred = pd.DataFrame(predictions, columns=['prediction'])
if SAVE_PREDICTIONS:
    df_pred.to_csv(OUTPUT_PRED_CSV, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] PREDICCIONES EN '{OUTPUT_PRED_CSV}'")

# COMENTARIO FINAL
# MODELO CAPTURA PATRONES TEMPORALES DE INFRAESTRUCTURA Y CONTEXTO
# SALIDAS NORMALIZADAS USADAS EN DIAGNÓSTICO HÍBRIDO CON IF PARA DETECCIÓN XAI/DSSA

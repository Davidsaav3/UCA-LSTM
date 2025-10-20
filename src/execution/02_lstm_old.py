import pandas as pd           # PARA MANEJO DE DATAFRAMES
import numpy as np            # PARA MANIPULACIÓN NUMÉRICA
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.preprocessing import StandardScaler  # PARA ESCALADO DE CARACTERÍSTICAS
from tensorflow.keras.models import Sequential    # MODELO SECUENCIAL
from tensorflow.keras.layers import LSTM, Dense  # CAPAS LSTM Y DENSAS
from tensorflow.keras.callbacks import EarlyStopping  # PARADA TEMPRANA
from tensorflow.keras.layers import Input  # IMPORTAR INPUT

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results'                        # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  

INPUT_CSV = '../../results/lstm_context_historical/05_variance.csv'   # CSV DE ENTRADA CON SECUENCIAS
OUTPUT_MODEL = os.path.join(EXECUTION_FOLDER, '02_lstm_model.keras')  # MODELO ENTRENADO FORMATO NATIVE KERAS
OUTPUT_PRED_CSV = os.path.join(EXECUTION_FOLDER, '02_lstm_predictions.csv')  # CSV CON PREDICCIONES
OUTPUT_HISTORY_CSV = os.path.join(EXECUTION_FOLDER, '02_lstm_history.csv')   # CSV CON HISTORIAL DE ENTRENAMIENTO

# PARÁMETROS DEL LSTM
TIMESTEPS = 10               # LONGITUD DE SECUENCIA PARA LSTM
FEATURES = None              # NÚMERO DE FEATURES, NONE = AUTOMÁTICO
TARGET_COLUMN = -1           # COLUMNA OBJETIVO, -1 = ÚLTIMA COLUMNA
LSTM_UNITS = 64              # NÚMERO DE NEURONAS EN LA CAPA LSTM
DENSE_UNITS = 32             # NÚMERO DE NEURONAS EN CAPA DENSAMENTE CONECTADA
OUTPUT_UNITS = 1             # SALIDA DEL MODELO, 1 PARA REGRESIÓN

# HIPERPARÁMETROS DE ENTRENAMIENTO
BATCH_SIZE = 32              # TAMAÑO DEL LOTE
EPOCHS = 50                  # NÚMERO MÁXIMO DE ÉPOCAS
VALIDATION_SPLIT = 0.2       # FRACCIÓN DE DATOS PARA VALIDACIÓN
LEARNING_RATE = 0.001        # TASA DE APRENDIZAJE (OPCIONAL, DEFAULT ADAM)
EARLY_STOPPING = True         # USAR PARADA TEMPRANA
PATIENCE = 5                 # PACIENCIA DE PARADA TEMPRANA

# FLAGS DE CONTROL
SHOW_INFO = True              # MOSTRAR MENSAJES INFORMATIVOS
SCALE_DATA = True             # ESCALAR FEATURES ANTES DE ENTRENAMIENTO
SAVE_MODEL = True             # GUARDAR MODELO ENTRENADO
SAVE_PREDICTIONS = True       # GUARDAR CSV DE PREDICCIONES
SAVE_HISTORY = True           # GUARDAR HISTORIAL DE ENTRENAMIENTO

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV, low_memory=False)  # LEER CSV DE ENTRADA
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# CONFIGURAR FEATURES Y TARGET
if FEATURES is None:
    FEATURES = df.shape[1]  # SI NO SE ESPECIFICA, USAR TODAS LAS COLUMNAS

# EXTRAER TARGET DINÁMICO
if TARGET_COLUMN == -1:
    y = df.iloc[TIMESTEPS-1::TIMESTEPS, -1].values  # ÚLTIMA COLUMNA
else:
    y = df.iloc[TIMESTEPS-1::TIMESTEPS, TARGET_COLUMN].values

# 🔹 CORREGIR REDIMENSIONAMIENTO PARA QUE EL NÚMERO DE FILAS SEA MÚLTIPLO DE TIMESTEPS
num_sequences = df.shape[0] // TIMESTEPS
trimmed_rows = num_sequences * TIMESTEPS
df_trimmed = df.iloc[:trimmed_rows]
if SHOW_INFO:
    print(f"[ INFO ] DATASET RECORTADO A {trimmed_rows} FILAS PARA REDIMENSIONAMIENTO")

# REDIMENSIONAR A 3D PARA LSTM
X = df_trimmed.values.reshape(num_sequences, TIMESTEPS, FEATURES)  
if SHOW_INFO:
    print(f"[ INFO ] DATOS REDIMENSIONADOS PARA LSTM: {X.shape}")

# ESCALADO DE FEATURES
if SCALE_DATA:
    scaler = StandardScaler()                        # ESCALADO PARA EVITAR DOMINANCIA DE MAGNITUD
    X_reshaped = X.reshape(-1, FEATURES)            # APLANAR PARA ESCALADO
    X_scaled = scaler.fit_transform(X_reshaped)     
    X_scaled = X_scaled.reshape(-1, TIMESTEPS, FEATURES)  # VOLVER A 3D
else:
    X_scaled = X.copy()
if SHOW_INFO and SCALE_DATA:
    print("[ INFO ] FEATURES ESCALADAS CON STANDARDSCALER")

# CREAR MODELO LSTM
model = Sequential()
model.add(Input(shape=(TIMESTEPS, FEATURES)))  # INPUT RECOMENDADO POR KERAS
model.add(LSTM(LSTM_UNITS, return_sequences=False))  # CAPA LSTM
model.add(Dense(DENSE_UNITS, activation='relu'))       # CAPA DENSAMENTE CONECTADA
model.add(Dense(OUTPUT_UNITS, activation='linear'))   # CAPA DE SALIDA LINEAL

model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # COMPILAR MODELO CON MSE Y MAE
if SHOW_INFO:
    model.summary()  # MOSTRAR RESUMEN DEL MODELO

# CONFIGURAR CALLBACKS
callbacks = []
if EARLY_STOPPING:
    callbacks.append(EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True))

# ENTRENAR MODELO
history = model.fit(
    X_scaled, y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1 if SHOW_INFO else 0
)

# GUARDAR HISTORIAL DE ENTRENAMIENTO
if SAVE_HISTORY:
    hist_df = pd.DataFrame(history.history)  # CONVERTIR HISTORIAL A DATAFRAME
    hist_df.to_csv(OUTPUT_HISTORY_CSV, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] HISTORIAL DE ENTRENAMIENTO EN '{OUTPUT_HISTORY_CSV}'")

# GUARDAR MODELO ENTRENADO
if SAVE_MODEL:
    model.save(OUTPUT_MODEL)  # GUARDAR EN FORMATO .keras NATIVE
    if SHOW_INFO:
        print(f"[ GUARDADO ] MODELO ENTRENADO EN '{OUTPUT_MODEL}'")

# REALIZAR PREDICCIONES
predictions = model.predict(X_scaled, verbose=0)
df_pred = pd.DataFrame(predictions, columns=['prediction'])
if SAVE_PREDICTIONS:
    df_pred.to_csv(OUTPUT_PRED_CSV, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] PREDICCIONES EN '{OUTPUT_PRED_CSV}'")

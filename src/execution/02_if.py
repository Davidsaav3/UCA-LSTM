import pandas as pd           # PARA MANEJO DE DATAFRAMES
import numpy as np            # PARA MANIPULACIÓN NUMÉRICA
from sklearn.preprocessing import StandardScaler  # PARA ESCALADO DE CARACTERÍSTICAS
from tensorflow.keras.models import Sequential    # MODELO SECUENCIAL
from tensorflow.keras.layers import LSTM, Dense  # CAPAS LSTM Y DENSAS
from tensorflow.keras.callbacks import EarlyStopping  # PARADA TEMPRANA
import os

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results'                       # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')  
INPUT_CSV = '../../results/lstm_context_historical/06_sequences.csv'  # CSV DE ENTRADA CON SECUENCIAS
OUTPUT_MODEL = os.path.join(EXECUTION_FOLDER, '03_lstm_model.h5')  # MODELO ENTRENADO
OUTPUT_PRED_CSV = os.path.join(EXECUTION_FOLDER, '03_lstm_predictions.csv')  # PREDICCIONES

SHOW_INFO = True            # MOSTRAR MENSAJES INFORMATIVOS EN CONSOLA
TIMESTEPS = 10              # LONGITUD DE LA SECUENCIA PARA LSTM
FEATURES = None             # NÚMERO DE FEATURES, NONE = AUTOMÁTICO
TARGET_COLUMN = -1          # COLUMNA OBJETIVO, -1 = ÚLTIMA COLUMNA
BATCH_SIZE = 32             # TAMAÑO DEL LOTE DE ENTRENAMIENTO
EPOCHS = 50                 # NÚMERO MÁXIMO DE ÉPOCAS
VALIDATION_SPLIT = 0.2      # FRACCIÓN DE DATOS PARA VALIDACIÓN
EARLY_STOPPING = True        # USAR PARADA TEMPRANA
PATIENCE = 5                # PACIENCIA PARA PARADA TEMPRANA

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV, low_memory=False)  # LEER CSV DE ENTRADA
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# SEPARAR FEATURES Y TARGET
if FEATURES is None:
    FEATURES = df.shape[1]  # SI NO SE ESPECIFICA, USAR TODAS LAS COLUMNAS

X = df.values.reshape(-1, TIMESTEPS, FEATURES)  # REDIMENSIONAR A 3D PARA LSTM
y = df.values[:, TARGET_COLUMN]                 # TOMAR COLUMNA OBJETIVO

if SHOW_INFO:
    print(f"[ INFO ] DATOS REDIMENSIONADOS PARA LSTM: {X.shape}")

# ESCALADO DE FEATURES
scaler = StandardScaler()                        # ESCALADO PARA EVITAR DOMINANCIA DE MAGNITUD
X_reshaped = X.reshape(-1, FEATURES)            # APLANAR PARA ESCALADO
X_scaled = scaler.fit_transform(X_reshaped)     
X_scaled = X_scaled.reshape(-1, TIMESTEPS, FEATURES)  # VOLVER A 3D

# CREAR MODELO LSTM
model = Sequential() 
model.add(LSTM(64, input_shape=(TIMESTEPS, FEATURES), return_sequences=False))  # CAPA LSTM
model.add(Dense(32, activation='relu'))                                           # CAPA OCULTA DENSAMENTE CONECTADA
model.add(Dense(1, activation='linear'))                                          # CAPA DE SALIDA LINEAL PARA REGRESIÓN

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

# GUARDAR MODELO ENTRENADO
model.save(OUTPUT_MODEL)
if SHOW_INFO:
    print(f"[ GUARDADO ] MODELO ENTRENADO EN '{OUTPUT_MODEL}'")

# REALIZAR PREDICCIONES
predictions = model.predict(X_scaled, verbose=0)
df_pred = pd.DataFrame(predictions, columns=['prediction'])
df_pred.to_csv(OUTPUT_PRED_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] PREDICCIONES EN '{OUTPUT_PRED_CSV}'")

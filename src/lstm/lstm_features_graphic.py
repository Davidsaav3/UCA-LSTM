import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime

# === 0. Crear carpetas de salida si no existen ===
os.makedirs("../../results/lstm/predictions", exist_ok=True)
os.makedirs("../../results/lstm/plots", exist_ok=True)

# === 0.1. Generar timestamp ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# === 1. Cargar CSV ===
df = pd.read_csv("../../data/raw/23092025 Gandia-AEMET-OpenWeather.csv")

# --- Seleccionamos la columna objetivo ---
target_col = "nivel_plaxiquet"

# --- Filtramos solo columnas numéricas ---
df = df.select_dtypes(include=[np.number]).dropna()

# --- Escalado ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# === 2. Preparar datos para LSTM ===
def create_sequences(data, target_index, seq_length=20):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])              # secuencia de entrada
        y.append(data[i+seq_length, target_index])  # valor siguiente a predecir
    return np.array(X), np.array(y)

# Índice de la columna objetivo
target_index = df.columns.get_loc(target_col)

# Crear secuencias
X, y = create_sequences(scaled_data, target_index, seq_length=20)

# Dividir train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# === 3. Construir modelo LSTM ===
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),   # Entrada: secuencia de 20 pasos
    LSTM(64, return_sequences=True),  # 64 neuronas, devuelve secuencias completas
    Dropout(0.2),                     # 20% neuronas apagadas para regularización
    LSTM(32),                         # segunda capa LSTM con 32 neuronas
    Dropout(0.2),
    Dense(1)                          # salida: un valor
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# === 4. Entrenamiento ===
history = model.fit(
    X_train, y_train,
    epochs=1,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# === 5. Evaluación ===
loss = model.evaluate(X_test, y_test)
print(f"Loss en test: {loss}")

# === 6. Predicciones ===
y_pred = model.predict(X_test)

# Invertir escala de predicciones y valores reales
scaled_target = np.zeros((len(y_pred), scaled_data.shape[1]))
scaled_target[:, target_index] = y_pred[:, 0]
y_pred_rescaled = scaler.inverse_transform(scaled_target)[:, target_index]

scaled_target_real = np.zeros((len(y_test), scaled_data.shape[1]))
scaled_target_real[:, target_index] = y_test
y_test_rescaled = scaler.inverse_transform(scaled_target_real)[:, target_index]

# === 6.1 Guardar resultados en CSV con timestamp ===
csv_path = f"../../results/lstm/predictions/prediction_{timestamp}.csv"
resultados = pd.DataFrame({
    "Real": y_test_rescaled,
    "Predicho": y_pred_rescaled
})
resultados.to_csv(csv_path, index=False)
print(f"Predicciones guardadas en {csv_path} ✅")

# === 7. Gráfico Real vs Predicho ===
plt.figure(figsize=(12,6))
plt.plot(resultados["Real"], label="Real", linewidth=2)
plt.plot(resultados["Predicho"], label="Predicho", linewidth=2, linestyle="dashed")
plt.title("Nivel Plaxiquet - Real vs Predicho", fontsize=14)
plt.xlabel("Tiempo (índice de muestra)")
plt.ylabel("Nivel")
plt.legend()
plt.grid(True, alpha=0.3)
plot_real_path = f"../../results/lstm/plots/prediction_{timestamp}.png"
plt.savefig(plot_real_path, dpi=300)
plt.show()
print(f"Gráfico Real vs Prediction guardado en {plot_real_path} ✅")

# === 8. Gráfico de la función de pérdida ===
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Evolución de la pérdida (MSE)", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, alpha=0.3)
plot_loss_path = f"../../results/lstm/plots/loss_{timestamp}.png"
plt.savefig(plot_loss_path, dpi=300)
plt.show()
print(f"Gráfico de pérdida guardado en {plot_loss_path} ✅")

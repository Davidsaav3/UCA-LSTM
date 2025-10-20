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
import time
import json

# ==========================
# === 0. CONFIGURACIÓN GENERAL ===
# ==========================
target_col = "nivel_plaxiquet"    # Columna objetivo
seq_length = 20                   # Longitud de secuencia para LSTM
test_size = 0.2                   # Proporción test
batch_size = 32
epochs = 10
show_plots = True
auto_params = True                # Para posibles ajustes automáticos futuros

# ==========================
# === 0.1 CREAR CARPETAS DE SALIDA ===
# ==========================
os.makedirs("../../results/lstm/predictions", exist_ok=True)
os.makedirs("../../results/lstm/plots", exist_ok=True)
os.makedirs("../../results/lstm/params", exist_ok=True)

# ==========================
# === 0.2 TIMESTAMP ===
# ==========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ==========================
# === 1. PREPROCESAMIENTO ===
# ==========================
def load_and_preprocess(csv_path):
    start_time = time.time()
    print(f"📥 Cargando CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Convertir fechas a timestamps
    for col in df.select_dtypes(include=['object', 'datetime']):
        if 'date' in col.lower() or 'time' in col.lower() or 'datetime' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                df[col] = df[col].astype(np.int64) // 10**9
                print(f"🗓️ Columna convertida a numérica: {col}")
            except Exception as e:
                print(f"⚠️ No se pudo convertir {col}: {e}")

    # Codificar columnas categóricas
    for col in df.select_dtypes(include=['object']):
        df[col] = pd.Categorical(df[col]).codes
        print(f"🏷️ Columna categórica codificada: {col}")

    # Filtrar solo columnas numéricas
    df_num = df.select_dtypes(include=[np.number]).dropna()
    excluded = set(df.columns) - set(df_num.columns)
    print(f"✅ Columnas numéricas seleccionadas: {len(df_num.columns)}")
    if excluded:
        print(f"⚠️ Columnas excluidas: {excluded}")

    end_time = time.time()
    print(f"⏱️ Preprocesamiento completado en {end_time - start_time:.2f} s")
    return df_num

# ==========================
# === 2. CREACIÓN DE SECUENCIAS ===
# ==========================
def create_sequences(data, target_col, seq_length=20):
    X, y = [], []
    target_index = data.columns.get_loc(target_col)
    values = data.values
    for i in range(len(values) - seq_length):
        X.append(values[i:i+seq_length])
        y.append(values[i+seq_length, target_index])
    return np.array(X), np.array(y)

# ==========================
# === 3. ESCALADO ===
# ==========================
def scale_data(df_num):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_num)
    return scaled_data, scaler

# ==========================
# === 4. CONSTRUCCIÓN DEL MODELO LSTM ===
# ==========================
def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model

# ==========================
# === 5. ENTRENAMIENTO ===
# ==========================
def train_model(model, X_train, y_train, batch_size, epochs):
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    end_time = time.time()
    print(f"⏱️ Entrenamiento completado en {end_time - start_time:.2f} s")
    return history, end_time - start_time

# ==========================
# === 6. PREDICCIÓN Y ESCALADO INVERSO ===
# ==========================
def predict_and_inverse(model, X, y, scaler, target_index):
    y_pred = model.predict(X)
    # Escalar de vuelta
    scaled_target_pred = np.zeros((len(y_pred), scaler.scale_.shape[0]))
    scaled_target_pred[:, target_index] = y_pred[:, 0]
    y_pred_rescaled = scaler.inverse_transform(scaled_target_pred)[:, target_index]

    scaled_target_real = np.zeros((len(y), scaler.scale_.shape[0]))
    scaled_target_real[:, target_index] = y
    y_real_rescaled = scaler.inverse_transform(scaled_target_real)[:, target_index]
    return y_real_rescaled, y_pred_rescaled

# ==========================
# === 7. GUARDAR RESULTADOS Y PARÁMETROS ===
# ==========================
def save_results(y_real, y_pred, timestamp, params_dict, selected_columns):
    # Guardar predicciones
    csv_path = f"../../results/lstm/predictions/prediction_{timestamp}.csv"
    df_res = pd.DataFrame({"Real": y_real, "Predicho": y_pred})
    df_res.to_csv(csv_path, index=False)
    print(f"✅ Predicciones guardadas en {csv_path}")

    # Ampliar parámetros a guardar
    params_dict_full = params_dict.copy()
    params_dict_full.update({
        "timestamp": timestamp,
        "selected_columns": selected_columns,
        "scaler": "MinMaxScaler"
    })

    # Guardar parámetros
    params_path = f"../../results/lstm/params/params_{timestamp}.json"
    with open(params_path, "w") as f:
        json.dump(params_dict_full, f, indent=4)
    print(f"✅ Parámetros guardados en {params_path}")
    return df_res


# ==========================
# === 8. GRAFICOS ===
# ==========================
def plot_results(df_res, timestamp):
    plt.figure(figsize=(12,6))
    plt.plot(df_res["Real"], label="Real", linewidth=2)
    plt.plot(df_res["Predicho"], label="Predicho", linewidth=2, linestyle="dashed")
    plt.title(f"{target_col} - Real vs Predicho", fontsize=14)
    plt.xlabel("Tiempo (índice de muestra)")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = f"../../results/lstm/plots/prediction_{timestamp}.png"
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"✅ Gráfico Real vs Predicho guardado en {plot_path}")

# ==========================
# === 9. EJECUCIÓN PRINCIPAL ===
# ==========================
start_total = time.time()

# 9.1 Cargar y preprocesar
df_num = load_and_preprocess("../../data/dataset.csv")

# 9.2 Escalar
scaled_data, scaler = scale_data(df_num)

# 9.3 Crear secuencias sobre datos escalados
scaled_df = pd.DataFrame(scaled_data, columns=df_num.columns)
X, y = create_sequences(scaled_df, target_col, seq_length)


# 9.4 Dividir train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=False
)

# 9.5 Construir modelo
model = build_lstm((X_train.shape[1], X_train.shape[2]))

# 9.6 Entrenar modelo
history, train_time = train_model(model, X_train, y_train, batch_size, epochs)

# 9.7 Evaluación
loss = model.evaluate(X_test, y_test)
print(f"Loss en test: {loss}")

# 9.8 Predicción y escala inversa
target_index = df_num.columns.get_loc(target_col)
y_real_rescaled, y_pred_rescaled = predict_and_inverse(model, X_test, y_test, scaler, target_index)

# ==========================
# === 9.9 Guardar resultados y parámetros ===
# ==========================
params_dict = {
    "seq_length": seq_length,
    "batch_size": batch_size,
    "epochs": epochs,
    "target_col": target_col
}
selected_columns = df_num.columns.tolist()  # <-- lista de columnas usadas
df_res = save_results(y_real_rescaled, y_pred_rescaled, timestamp, params_dict, selected_columns)

# 9.10 Tiempo total
end_total = time.time()
print(f"⏱️ Tiempo total ejecución: {end_total - start_total:.2f} s (entrenamiento: {train_time:.2f} s)")

# 9.11 Graficar
plot_results(df_res, timestamp)

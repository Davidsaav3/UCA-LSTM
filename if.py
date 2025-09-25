import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import skew, kurtosis
import json
import time
from matplotlib.colors import Normalize

# ==========================
# === 0. CONFIGURACIÓN GENERAL ===
# ==========================
features_to_plot = ["nivel_plaxiquet"]   # Columnas a graficar, None = todas
features_to_save = ["nivel_plaxiquet"]   # Columnas a guardar en CSV, None = todas
show_plots = True                         # True = mostrar gráficos
rolling_window = 5                        # Ventana para features rolling
modo_rapido = True                        # True = rápido, False = lento
auto_params = True                        # True = ajuste automático de parámetros

# Parámetros manuales
iso_n_estimators = 300
iso_max_samples = 0.8
iso_contamination = 0.05
iso_max_features = 0.8
iso_bootstrap = True
iso_random_state = 42

# ==========================
# === 1. CREAR CARPETAS DE SALIDA ===
# ==========================
os.makedirs("../../results/if/predictions", exist_ok=True)
os.makedirs("../../results/if/plots", exist_ok=True)
os.makedirs("../../results/if/params", exist_ok=True)

# ==========================
# === 1.1 TIMESTAMP ===
# ==========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ==========================
# === 2. FUNCIONES DE PREPROCESAMIENTO ===
# ==========================
def load_and_preprocess(csv_path, rolling_window=5, modo_rapido=True):
    start_time = time.time()
    print(f"📥 Cargando CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convertir fechas a timestamp
    for col in df.select_dtypes(include=['object', 'datetime']):
        if 'date' in col.lower() or 'time' in col.lower() or 'datetime' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                df[col] = df[col].astype(np.int64) // 10**9
                print(f"🗓️ Columna convertida a numérica: {col}")
            except Exception as e:
                print(f"⚠️ No se pudo convertir {col}: {e}")
    
    # Convertir columnas categóricas a códigos numéricos
    for col in df.select_dtypes(include=['object']):
        df[col] = pd.Categorical(df[col]).codes
        print(f"🏷️ Columna categórica codificada: {col}")

    # Filtrar columnas numéricas
    df_num = df.select_dtypes(include=[np.number]).dropna()
    excluded = set(df.columns) - set(df_num.columns)
    print(f"✅ Columnas numéricas seleccionadas: {len(df_num.columns)}")
    if excluded:
        print(f"⚠️ Columnas aún excluidas: {excluded}")

    # Crear features derivadas
    print("🔧 Generando features derivadas...")
    features_dict = {}
    features_dict.update(df_num.to_dict('list'))
    
    # Diff
    df_diff = df_num.diff().fillna(0).add_suffix('_diff')
    features_dict.update(df_diff.to_dict('list'))
    
    # Lag
    df_lag = df_num.shift(1).fillna(0).add_suffix('_lag')
    features_dict.update(df_lag.to_dict('list'))
    
    # Rolling mean y std
    df_roll_mean = df_num.rolling(window=rolling_window, min_periods=1).mean().add_suffix('_rollmean')
    df_roll_std = df_num.rolling(window=rolling_window, min_periods=1).std().fillna(0).add_suffix('_rollstd')
    features_dict.update(df_roll_mean.to_dict('list'))
    features_dict.update(df_roll_std.to_dict('list'))
    
    if not modo_rapido:
        # Skew y kurtosis
        df_skew = df_num.rolling(window=rolling_window, min_periods=1).apply(lambda x: skew(x), raw=True).add_suffix('_skew')
        df_kurt = df_num.rolling(window=rolling_window, min_periods=1).apply(lambda x: kurtosis(x), raw=True).add_suffix('_kurt')
        features_dict.update(df_skew.to_dict('list'))
        features_dict.update(df_kurt.to_dict('list'))
    
    df_features = pd.DataFrame(features_dict)
    end_time = time.time()
    print(f"⏱️ Features generadas en {end_time - start_time:.2f} s")
    return df_num, df_features

# ==========================
# === 3. GENERAR ANOMALÍAS SINTÉTICAS ===
# ==========================
def introduce_synthetic_anomalies(df_original, columns, anomaly_fraction=0.01, factor=2.0, seed=None):
    df_anom = df_original.reset_index(drop=True).copy()  # Reset indices
    n_samples = len(df_anom)
    n_anomalies = int(anomaly_fraction * n_samples)
    rng = np.random.default_rng(seed)
    anomaly_idx = rng.choice(n_samples, size=n_anomalies, replace=False)
    
    for col in columns:
        if col in df_anom.columns:
            df_anom.loc[anomaly_idx, col] = df_anom.loc[anomaly_idx, col] * factor
    print(f"✅ Se han introducido {n_anomalies} anomalías en columnas: {columns}")
    return df_anom, anomaly_idx

# ==========================
# === 4. ESCALADO ROBUSTO ===
# ==========================
def scale_features(df_features):
    print("🔧 Escalando features con RobustScaler...")
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df_features)
    return scaled_data

# ==========================
# === 5. AJUSTE AUTOMÁTICO DE PARÁMETROS ===
# ==========================
def auto_adjust_params(scaled_data):
    n_samples, n_features = scaled_data.shape
    n_estimators = min(max(int(n_samples / 500), 50), 500)
    max_samples = min(max(0.1, n_samples / 10000), 1.0)
    max_features = min(max(0.5, n_features / 50), 1.0)
    contamination = min(max(0.01, 0.05), 0.5)
    print(f"✅ Parámetros automáticos ajustados: n_estimators={n_estimators}, max_samples={max_samples:.2f}, max_features={max_features:.2f}, contamination={contamination:.2f}")
    return n_estimators, max_samples, max_features, contamination

# ==========================
# === 6. ENTRENAMIENTO Y PREDICCIÓN ===
# ==========================
def train_iforest(scaled_data, n_estimators, max_samples, max_features, contamination, bootstrap=True, random_state=42):
    start_time = time.time()
    print("🌲 Entrenando Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
        verbose=0
    )
    iso_forest.fit(scaled_data)
    preds = iso_forest.predict(scaled_data)
    scores = iso_forest.decision_function(scaled_data)
    end_time = time.time()
    print(f"⏱️ Entrenamiento completado en {end_time - start_time:.2f} s")
    return iso_forest, preds, scores, end_time - start_time

# ==========================
# === 7. CREAR DATAFRAME DE RESULTADOS ===
# ==========================
def create_results(df_original, preds, scores, features_to_save=None, iso_forest=None):
    results_df = pd.DataFrame(df_original if features_to_save is None else df_original[features_to_save])
    results_df["Anomalia"] = (preds == -1).astype(int)
    results_df["Score"] = scores
    results_df["Score_norm"] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    decision_threshold = iso_forest.offset_ if iso_forest else None
    results_df["Umbral_IForest"] = decision_threshold
    results_df["Ranking"] = results_df["Score"].rank(ascending=True).astype(int)
    return results_df

# ==========================
# === 8. GUARDAR RESULTADOS Y PARÁMETROS ===
# ==========================
def save_results(results_df, timestamp, params_dict):
    csv_path = f"../../results/if/predictions/anomaly_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✅ Resultados guardados en {csv_path}")
    params_path = f"../../results/if/params/anomaly_{timestamp}.json"
    with open(params_path, "w") as f:
        json.dump(params_dict, f, indent=4)
    print(f"✅ Parámetros guardados en {params_path}")

# ==========================
# === 9. GRAFICOS DE ANOMALÍAS ===
# ==========================
def plot_anomalies(df_original, results_df, features_to_plot, timestamp, show_plots=True):
    plot_features = df_original.columns if features_to_plot is None else features_to_plot
    for feature in plot_features:
        plt.figure(figsize=(12,6))
        plt.plot(df_original[feature], label="Valor Real")
        norm = Normalize(vmin=0, vmax=1)
        plt.scatter(
            results_df.index[results_df["Anomalia"]==1],
            df_original.loc[results_df["Anomalia"]==1, feature],
            color=plt.cm.Reds(norm(results_df.loc[results_df["Anomalia"]==1,"Score_norm"])),
            label="Anomalía",
            s=50
        )
        plt.title(f"Detección de anomalías - {feature}", fontsize=14)
        plt.xlabel("Índice de muestra")
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = f"../../results/if/plots/anomaly_{feature}_{timestamp}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        print(f"✅ Gráfico guardado: {plot_path}")
        if show_plots:
            plt.show()
        else:
            plt.close()

# ==========================
# === 10. EJECUCIÓN PRINCIPAL ===
# ==========================
start_total = time.time()

# 10.1 Cargar y preprocesar
df_original, df_features = load_and_preprocess("../../data/raw/23092025 Gandia-AEMET-OpenWeather.csv", rolling_window, modo_rapido)

# 10.2 Introducir anomalías sintéticas
columns_to_alter = ["nivel_plaxiquet"]
df_original, synthetic_anomaly_indices = introduce_synthetic_anomalies(df_original, columns_to_alter, anomaly_fraction=0.02, factor=3.0, seed=42)

# 10.3 Escalar features
scaled_data = scale_features(df_features)

# 10.4 Ajuste de parámetros
if auto_params:
    iso_n_estimators, iso_max_samples, iso_max_features, iso_contamination = auto_adjust_params(scaled_data)
else:
    print("ℹ️ Usando parámetros manuales")

# 10.5 Entrenar Isolation Forest
iso_forest, preds, scores, train_time = train_iforest(
    scaled_data,
    iso_n_estimators,
    iso_max_samples,
    iso_max_features,
    iso_contamination,
    iso_bootstrap,
    iso_random_state
)

# 10.6 Crear resultados
results_df = create_results(df_original, preds, scores, features_to_save, iso_forest)

# 10.7 Guardar resultados
params_dict = {
    "n_estimators": iso_n_estimators,
    "max_samples": iso_max_samples,
    "max_features": iso_max_features,
    "contamination": iso_contamination,
    "bootstrap": iso_bootstrap,
    "random_state": iso_random_state,
    "modo_rapido": modo_rapido,
    "rolling_window": rolling_window
}
save_results(results_df, timestamp, params_dict)

# 10.8 Resumen
n_anom = results_df["Anomalia"].sum()
n_total = len(results_df)
end_total = time.time()
print(f"\n📊 Resumen de anomalías:")
print(f" - Total registros: {n_total}")
print(f" - Anomalías detectadas: {n_anom} ({n_anom/n_total:.2%})")
print(f" - Umbral de decisión IF: {iso_forest.offset_:.4f}")
print(f" - Tiempo total ejecución: {end_total - start_total:.2f} s (entrenamiento: {train_time:.2f} s)")

# 10.9 Graficar anomalías
plot_anomalies(df_original, results_df, features_to_plot, timestamp, show_plots)

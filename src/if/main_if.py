import time
from config import *
from preprocess import load_and_preprocess
from synthetic_anomalies import introduce_synthetic_anomalies
from scaling import scale_features
from auto_params import auto_adjust_params
from train_iforest import train_iforest
from results import create_results, save_results
from plotting import plot_anomalies

start_total = time.time()

# 1️⃣ Preprocesamiento
df_original, df_features = load_and_preprocess("../data/raw/23092025 Gandia-AEMET-OpenWeather.csv", rolling_window, modo_rapido)

# 2️⃣ Introducir anomalías sintéticas
columns_to_alter = ["nivel_plaxiquet"]
df_original, synthetic_anomaly_indices = introduce_synthetic_anomalies(df_original, columns_to_alter, anomaly_fraction=0.02, factor=3.0, seed=42)

# 3️⃣ Escalado
scaled_data = scale_features(df_features)

# 4️⃣ Ajuste de parámetros
if auto_params:
    iso_n_estimators, iso_max_samples, iso_max_features, iso_contamination = auto_adjust_params(scaled_data)
else:
    print("ℹ️ Usando parámetros manuales")

# 5️⃣ Entrenamiento
iso_forest, preds, scores, train_time = train_iforest(
    scaled_data,
    iso_n_estimators,
    iso_max_samples,
    iso_max_features,
    iso_contamination,
    iso_bootstrap,
    iso_random_state
)

# 6️⃣ Crear resultados
results_df = create_results(df_original, preds, scores, features_to_save, iso_forest)

# 7️⃣ Guardar resultados
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
save_results(results_df, timestamp, params_dict, pred_path, params_path)

# 8️⃣ Resumen
n_anom = results_df["Anomalia"].sum()
n_total = len(results_df)
end_total = time.time()
print(f"\n📊 Resumen de anomalías:")
print(f" - Total registros: {n_total}")
print(f" - Anomalías detectadas: {n_anom} ({n_anom/n_total:.2%})")
print(f" - Umbral de decisión IF: {iso_forest.offset_:.4f}")
print(f" - Tiempo total ejecución: {end_total - start_total:.2f} s (entrenamiento: {train_time:.2f} s)")

# 9️⃣ Graficar anomalías
plot_anomalies(df_original, results_df, features_to_plot, timestamp, plot_path, show_plots)

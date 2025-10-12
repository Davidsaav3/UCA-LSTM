from datetime import datetime

# ==========================
# === CONFIGURACIÓN GENERAL ===
# ==========================
features_to_plot = ["nivel_plaxiquet"]   # Columnas a graficar
features_to_save = ["nivel_plaxiquet"]   # Columnas a guardar en CSV
show_plots = True
rolling_window = 5
modo_rapido = True
auto_params = True

# Parámetros manuales Isolation Forest
iso_n_estimators = 300
iso_max_samples = 0.8
iso_contamination = 0.05
iso_max_features = 0.8
iso_bootstrap = True
iso_random_state = 42

# Carpetas de salida
pred_path = "../../results/if/predictions"
plot_path = "../../results/if/plots"
params_path = "../../results/if/params"

# Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

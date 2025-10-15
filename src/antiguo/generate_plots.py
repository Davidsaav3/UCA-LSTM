import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def main():
    # === CARGA DE PARAMETROS DESDE JSON ===
    with open("parameters_config.json") as f:
        params = json.load(f)["generate_plots"]
 
    # === CARGA DE RESULTADOS DEL ISOLATION FOREST ===
    df = pd.read_csv("../results/intermediate/isolation_forest_results.csv")
    target_col = params["target_col"]

    # Crear carpeta de salida si no existe
    os.makedirs(os.path.dirname(params["plot_comparison_path"]), exist_ok=True)

    # =========================
    # CALCULO DE METRICAS BASICAS
    # =========================
    # precision, recall, f1, accuracy sobre detecciones
    y_true = df["is_anomaly_real"]
    y_pred = df["is_anomaly_detected"]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # =========================
    # 1️⃣ Serie temporal de anomalías
    # =========================
    plt.figure(figsize=(14,6))
    # Serie principal
    plt.plot(df[target_col], alpha=params["alpha"], label="Serie")
    # Anomalías reales
    plt.scatter(df.index[df["is_anomaly_real"]==1], 
                df[target_col][df["is_anomaly_real"]==1],
                color=params["colors_anomaly_real"], 
                marker=params["marker_anomaly_real"], 
                label="Anomalias reales")
    # Anomalías detectadas
    plt.scatter(df.index[df["is_anomaly_detected"]==1], 
                df[target_col][df["is_anomaly_detected"]==1],
                color=params["colors_anomaly_detected"], 
                marker=params["marker_anomaly_detected"], 
                label="Anomalias detectadas", 
                facecolors="none")
    plt.title(f"Serie temporal de '{target_col}' con anomalías")
    plt.xlabel("Índice")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(params["plot_comparison_path"])
    plt.close()

    # =========================
    # 2️⃣ Histograma de scores de Isolation Forest
    # =========================
    plt.figure(figsize=(10,5))
    plt.hist(df["score"][df["is_anomaly_detected"]==0], bins=params["bins"], alpha=0.7, label="Normales")
    plt.hist(df["score"][df["is_anomaly_detected"]==1], bins=params["bins"], alpha=0.7, label="Anomalias detectadas")
    plt.title("Distribucion de scores de Isolation Forest")
    plt.xlabel("Score")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(params["plot_scores_hist_path"])
    plt.close()

    # =========================
    # GUARDAR PARAMETROS UTILIZADOS
    # =========================
    with open("../results/parameters.txt", "a") as f:
        f.write("\n===== generate_plots.py =====\n")
        for k,v in params.items():
            f.write(f"{k}: {v}\n")

    print("Paso 4 completado: Graficos generados ✅")

if __name__ == "__main__":
    main()

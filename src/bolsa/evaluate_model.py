import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef

def main():
    # === CARGA DE PARAMETROS DESDE JSON ===
    with open("parameters_config.json") as f:
        params = json.load(f)["evaluate_model"]

    # === CARGA DE RESULTADOS DEL ISOLATION FOREST ===
    df = pd.read_csv("../results/intermediate/isolation_forest_results.csv")
    target_col = params["target_col"]

    # Etiquetas reales y predichas
    y_true = df.get("is_anomaly_real", pd.Series(np.zeros(len(df))))
    y_pred = df["is_anomaly_detected"]

    # =========================
    # CALCULO DE METRICAS DE DESEMPEÑO
    # =========================
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # =========================
    # ESTADISTICAS DE ANOMALIAS
    # =========================
    n_real = int(y_true.sum())
    n_detected = int(y_pred.sum())
    n_correct = int(((y_true==1) & (y_pred==1)).sum())
    false_positives = int(((y_true==0) & (y_pred==1)).sum())
    false_negatives = int(((y_true==1) & (y_pred==0)).sum())
    ratio_detection = n_correct/n_real if n_real>0 else np.nan
    ratio_false_positives = false_positives/n_detected if n_detected>0 else np.nan

    # Estadisticas de scores
    scores = df["score"]
    score_stats = {
        "min": scores.min(),
        "max": scores.max(),
        "mean": scores.mean(),
        "std": scores.std(),
        "25%": scores.quantile(0.25),
        "50%": scores.quantile(0.5),
        "75%": scores.quantile(0.75)
    }

    # Estadisticas de columna objetivo
    col = df[target_col]
    col_stats = {
        "min": col.min(),
        "max": col.max(),
        "mean": col.mean(),
        "std": col.std(),
        "25%": col.quantile(0.25),
        "50%": col.quantile(0.5),
        "75%": col.quantile(0.75)
    }

    # Crear carpeta de metrics si no existe
    os.makedirs(os.path.dirname(params["metrics_path"]), exist_ok=True)

    # =========================
    # GUARDAR METRICAS EN TXT
    # =========================
    with open(params["metrics_path"], "w") as f:
        f.write("===== Metricas de desempeño =====\n")
        f.write(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}\nAccuracy: {accuracy:.4f}\nMCC: {mcc:.4f}\n\n")
        f.write(f"Numero de anomalias reales: {n_real}\nNumero de anomalias detectadas: {n_detected}\n")
        f.write(f"Numero correctamente detectadas: {n_correct}\nFalsos positivos: {false_positives}\nFalsos negativos: {false_negatives}\n")
        f.write(f"Ratio de deteccion: {ratio_detection:.4f}\nRatio de falsos positivos: {ratio_false_positives:.4f}\n\n")

        f.write("===== Estadisticas de scores =====\n")
        for k,v in score_stats.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n===== Estadisticas columna '{}' =====\n".format(target_col))
        for k,v in col_stats.items():
            f.write(f"{k}: {v:.4f}\n")

    # =========================
    # GUARDAR CSVs AUXILIARES
    # =========================
    # Scores completos
    df.to_csv("../results/metrics/scores_complete.csv", index=False)

    # Solo anomalías detectadas
    df[df["is_anomaly_detected"]==1].to_csv("../results/metrics/anomalies_detected.csv", index=False)

    # Comparación real vs detectadas
    if "is_anomaly_real" in df.columns:
        df[["is_anomaly_real","is_anomaly_detected", target_col]].to_csv("../results/metrics/real_vs_detected.csv", index=False)

    # Anomalías detectadas con índice y valor para análisis temporal
    anomalies_time = df[df["is_anomaly_detected"]==1][[target_col]]
    anomalies_time.to_csv("../results/metrics/anomalies_time.csv", index=True)

    # CSV resumen de métricas
    metrics_dict = {**score_stats, **col_stats,
                    "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "mcc": mcc,
                    "n_real": n_real, "n_detected": n_detected, "n_correct": n_correct,
                    "false_positives": false_positives, "false_negatives": false_negatives,
                    "ratio_detection": ratio_detection, "ratio_false_positives": ratio_false_positives}
    pd.DataFrame([metrics_dict]).to_csv(params["metrics_csv_path"], index=False)

    # =========================
    # GUARDAR PARAMETROS UTILIZADOS
    # =========================
    with open("../results/parameters.txt", "a") as f:
        f.write("\n===== evaluate_model.py =====\n")
        for k,v in params.items():
            f.write(f"{k}: {v}\n")

    print("Paso 5 completado: Métricas evaluadas y CSVs auxiliares generados ✅")

if __name__ == "__main__":
    main()

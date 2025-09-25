import pandas as pd
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef

# CARGA DE PARAMETROS
with open("parameters_config.json") as f:
    params = json.load(f)["evaluate_model"]

df = pd.read_csv("../results/intermediate/isolation_forest_results.csv")
y_true = df["is_anomaly_real"]
y_pred = df["is_anomaly_detected"]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

# ESTADISTICAS
n_real = y_true.sum()
n_detected = y_pred.sum()
n_correct = ((y_true==1) & (y_pred==1)).sum()
false_positives = ((y_true==0) & (y_pred==1)).sum()
false_negatives = ((y_true==1) & (y_pred==0)).sum()

score_stats = (df["score"].max(), df["score"].min(), df["score"].mean(), df["score"].std())
col = df[params["target_col"]]
col_stats = (col.max(), col.min(), col.mean(), col.std())

os.makedirs(os.path.dirname(params["metrics_path"]), exist_ok=True)

# GUARDAR TXT
with open(params["metrics_path"], "w") as f:
    f.write("===== Metricas de desempeño =====\n")
    f.write(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}\nAccuracy: {accuracy:.4f}\nMCC: {mcc:.4f}\n\n")
    f.write(f"Numero de anomalias reales: {n_real}\nNumero de anomalias detectadas: {n_detected}\nNumero correctamente detectadas: {n_correct}\nFalsos positivos: {false_positives}\nFalsos negativos: {false_negatives}\n\n")
    f.write("===== Estadisticas de scores =====\n")
    f.write(f"Max: {score_stats[0]:.4f}, Min: {score_stats[1]:.4f}, Media: {score_stats[2]:.4f}, Std: {score_stats[3]:.4f}\n\n")
    f.write(f"===== Estadisticas columna '{params['target_col']}' =====\n")
    f.write(f"Max: {col_stats[0]:.4f}, Min: {col_stats[1]:.4f}, Media: {col_stats[2]:.4f}, Std: {col_stats[3]:.4f}\n")

# GUARDAR CSV
metrics_csv = {
    "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "mcc": mcc,
    "n_real": n_real, "n_detected": n_detected, "n_correct": n_correct,
    "false_positives": false_positives, "false_negatives": false_negatives,
    "score_max": score_stats[0], "score_min": score_stats[1], "score_mean": score_stats[2], "score_std": score_stats[3],
    "col_max": col_stats[0], "col_min": col_stats[1], "col_mean": col_stats[2], "col_std": col_stats[3]
}
pd.DataFrame([metrics_csv]).to_csv(params["metrics_csv_path"], index=False)

# GUARDAR PARAMETROS
with open("../results/parameters.txt", "a") as f:
    f.write("\n===== evaluate_model.py =====\n")
    for k,v in params.items():
        f.write(f"{k}: {v}\n")

print("Paso 5 completado: Metricasa evaluadas y guardadas ✅")

import pandas as pd
import json
import os

def create_results(df_original, preds, scores, features_to_save=None, iso_forest=None):
    results_df = pd.DataFrame(df_original if features_to_save is None else df_original[features_to_save])
    results_df["Anomalia"] = (preds == -1).astype(int)
    results_df["Score"] = scores
    results_df["Score_norm"] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    decision_threshold = iso_forest.offset_ if iso_forest else None
    results_df["Umbral_IForest"] = decision_threshold
    results_df["Ranking"] = results_df["Score"].rank(ascending=True).astype(int)
    return results_df

def save_results(results_df, timestamp, params_dict, pred_path, params_path):
    os.makedirs(pred_path, exist_ok=True)
    os.makedirs(params_path, exist_ok=True)
    csv_path = f"{pred_path}/anomaly_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✅ Resultados guardados en {csv_path}")
    params_file = f"{params_path}/anomaly_{timestamp}.json"
    with open(params_file, "w") as f:
        json.dump(params_dict, f, indent=4)
    print(f"✅ Parámetros guardados en {params_file}")

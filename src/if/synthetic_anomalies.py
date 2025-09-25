import numpy as np

def introduce_synthetic_anomalies(df_original, columns, anomaly_fraction=0.01, factor=2.0, seed=None):
    df_anom = df_original.reset_index(drop=True).copy()
    n_samples = len(df_anom)
    n_anomalies = int(anomaly_fraction * n_samples)
    rng = np.random.default_rng(seed)
    anomaly_idx = rng.choice(n_samples, size=n_anomalies, replace=False)
    
    for col in columns:
        if col in df_anom.columns:
            df_anom.loc[anomaly_idx, col] = df_anom.loc[anomaly_idx, col] * factor
    print(f"✅ Se han introducido {n_anomalies} anomalías en columnas: {columns}")
    return df_anom, anomaly_idx

import pandas as pd
import numpy as np
import time
from scipy.stats import skew, kurtosis

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
    
    # Columnas categóricas a códigos
    for col in df.select_dtypes(include=['object']):
        df[col] = pd.Categorical(df[col]).codes
        print(f"🏷️ Columna categórica codificada: {col}")

    # Filtrar columnas numéricas
    df_num = df.select_dtypes(include=[np.number]).dropna()
    excluded = set(df.columns) - set(df_num.columns)
    if excluded:
        print(f"⚠️ Columnas excluidas: {excluded}")

    # Features derivadas
    features_dict = {}
    features_dict.update(df_num.to_dict('list'))
    df_diff = df_num.diff().fillna(0).add_suffix('_diff')
    df_lag = df_num.shift(1).fillna(0).add_suffix('_lag')
    df_roll_mean = df_num.rolling(window=rolling_window, min_periods=1).mean().add_suffix('_rollmean')
    df_roll_std = df_num.rolling(window=rolling_window, min_periods=1).std().fillna(0).add_suffix('_rollstd')
    features_dict.update(df_diff.to_dict('list'))
    features_dict.update(df_lag.to_dict('list'))
    features_dict.update(df_roll_mean.to_dict('list'))
    features_dict.update(df_roll_std.to_dict('list'))

    if not modo_rapido:
        df_skew = df_num.rolling(window=rolling_window, min_periods=1).apply(lambda x: skew(x), raw=True).add_suffix('_skew')
        df_kurt = df_num.rolling(window=rolling_window, min_periods=1).apply(lambda x: kurtosis(x), raw=True).add_suffix('_kurt')
        features_dict.update(df_skew.to_dict('list'))
        features_dict.update(df_kurt.to_dict('list'))
    
    df_features = pd.DataFrame(features_dict)
    end_time = time.time()
    print(f"⏱️ Features generadas en {end_time - start_time:.2f} s")
    return df_num, df_features

from sklearn.preprocessing import RobustScaler

def scale_features(df_features):
    print("🔧 Escalando features con RobustScaler...")
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df_features)
    return scaled_data

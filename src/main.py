import os

def main():
    # === PASO 1: CARGAR DATASET ===
    os.system("python load_data.py")
    
    # === PASO 2: INTRODUCIR ANOMALIAS ===
    os.system("python add_anomalies.py")
    
    # === PASO 3: ENTRENAR ISOLATION FOREST ===
    os.system("python isolation_forest_model.py")
    
    # === PASO 4: GENERAR GRAFICOS ===
    os.system("python generate_plots.py")
    
    # === PASO 5: EVALUAR MODELO Y GUARDAR METRICAS ===
    os.system("python evaluate_model.py")
    
    # === PASO 6: CLUSTERING DE ANOMALIAS ===
    os.system("python clustering_anomalies.py")

    # === FIN DEL FLUJO ===
    print("Flujo completo ejecutado ✅")

if __name__ == "__main__":
    main()

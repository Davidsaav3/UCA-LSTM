import load_data
import add_anomalies
import isolation_forest_model
import generate_plots
import evaluate_model
import clustering_anomalies

def main():
    # FLUJO COMPLETO
    print("Ejecutando pipeline completo...")
    load_data
    add_anomalies
    isolation_forest_model
    generate_plots
    evaluate_model
    clustering_anomalies
    print("Flujo completo ejecutado ✅")

if __name__=="__main__":
    main()

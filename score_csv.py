import pandas as pd
import numpy as np
import sys
import joblib
import json
import os
from datetime import datetime
from preprocess import load_preprocessor, transform


# -------------------------------------------------------------------
# Configuración global
# -------------------------------------------------------------------

# Carpeta base donde se almacenan los modelos entrenados
MODEL_DIR = "models"


# -------------------------------------------------------------------
# Scoring de un DataFrame contra un modelo entrenado
# -------------------------------------------------------------------

def score_csv(df, model_path):
    """
    Ejecuta el proceso de scoring sobre un DataFrame utilizando
    un conjunto de modelos previamente entrenados.

    Se cargan:
        - preprocesador
        - modelo KMeans
        - modelo Isolation Forest

    A partir de ellos se calculan métricas agregadas que permiten
    evaluar el comportamiento del tráfico de entrada frente al
    modelo entrenado.

    Métricas generadas:
        - media y mínimo del score de Isolation Forest
        - ratio de anomalías detectadas por Isolation Forest
        - media y máximo de la distancia al centroide KMeans

    Parameters
    ----------
    df : pd.DataFrame
        Datos de entrada a evaluar.
    model_path : str
        Ruta al directorio que contiene los modelos entrenados
        (preprocessor.joblib, kmeans.joblib, isoforest.joblib).

    Returns
    -------
    dict
        Diccionario con métricas agregadas y metadatos del modelo.
    """

    # Carga de modelos entrenados
    pre = load_preprocessor(os.path.join(model_path, "preprocessor.joblib"))
    kmeans = joblib.load(os.path.join(model_path, "kmeans.joblib"))
    iso = joblib.load(os.path.join(model_path, "isoforest.joblib"))

    # Transformación de los datos usando el mismo pipeline de entrenamiento
    X = transform(df, pre)

    # Scoring con Isolation Forest
    iso_score = iso.decision_function(X)
    iso_pred = iso.predict(X)

    # Asignación de clusters y cálculo de distancias a centroides
    labels = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    dists = np.linalg.norm(X - centers[labels], axis=1)

    # Métricas agregadas
    results = {
        "rows": len(df),
        "iso_score_mean": float(iso_score.mean()),
        "iso_score_min": float(iso_score.min()),
        "anomaly_ratio_iso": float((iso_pred == -1).mean()),
        "kmeans_distance_mean": float(dists.mean()),
        "kmeans_distance_max": float(dists.max())
    }

    # Lectura del modo de entrenamiento
    mode_file = os.path.join(model_path, "mode.txt")
    if os.path.exists(mode_file):
        with open(mode_file, "r") as f:
            results["mode"] = f.read().strip()
    else:
        results["mode"] = "unknown"

    # Lectura de metadatos del modelo
    info_file = os.path.join(model_path, "model_info.json")
    if os.path.exists(info_file):
        with open(info_file, "r") as f:
            info = json.load(f)

        results.update({
            "folder_name": info.get("folder_name", "unknown"),
            "trained_at": info.get("trained_at", "unknown"),
            "trained_rows": info.get("num_rows", None),
            "trained_features": info.get("n_features", None)
        })
    else:
        results.update({
            "folder_name": "unknown",
            "trained_at": "unknown",
            "trained_rows": None,
            "trained_features": None
        })

    return results


# -------------------------------------------------------------------
# Entrada por línea de comandos
# -------------------------------------------------------------------

if __name__ == "__main__":

    # CSV a evaluar
    csv_file = sys.argv[1]

    # Rango de fechas que identifica el modelo a utilizar
    start_date = sys.argv[2] if len(sys.argv) > 2 else None
    end_date = sys.argv[3] if len(sys.argv) > 3 else None

    # Construcción del nombre de carpeta siguiendo exactamente
    # el mismo convenio que el script de entrenamiento
    if start_date and end_date:
        folder_name = f"{start_date}_{end_date}"
    elif start_date:
        folder_name = f"{start_date}_to_latest"
    elif end_date:
        folder_name = f"from_earliest_{end_date}"
    else:
        folder_name = "global"

    model_path = os.path.join(MODEL_DIR, folder_name)

    # Validación de existencia del modelo
    if not os.path.exists(model_path):
        print(json.dumps({
            "error": f"No se encontró modelo en {model_path}"
        }))
        sys.exit(1)

    # Carga del CSV a evaluar
    df = pd.read_csv(csv_file)

    # Ejecución del scoring
    res = score_csv(df, model_path)

    # Salida estándar en JSON (pensado para ser consumido por backend)
    print(json.dumps(res))

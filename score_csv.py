
import pandas as pd
import numpy as np
import sys
import joblib
import json
import os
from datetime import datetime
from preprocess import load_preprocessor, transform

MODEL_DIR = "models" 

def score_csv(df, model_path):
    pre = load_preprocessor(os.path.join(model_path, "preprocessor.joblib"))
    kmeans = joblib.load(os.path.join(model_path, "kmeans.joblib"))
    iso = joblib.load(os.path.join(model_path, "isoforest.joblib"))

    X = transform(df, pre)

    iso_score = iso.decision_function(X)
    iso_pred = iso.predict(X)   

    labels = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    dists = np.linalg.norm(X - centers[labels], axis=1)

    results = {
        "rows": len(df),
        "iso_score_mean": float(iso_score.mean()),
        "iso_score_min": float(iso_score.min()),
        "anomaly_ratio_iso": float((iso_pred == -1).mean()),
        "kmeans_distance_mean": float(dists.mean()),
        "kmeans_distance_max": float(dists.max())
    }

    mode_file = os.path.join(model_path, "mode.txt")
    if os.path.exists(mode_file):
        with open(mode_file, "r") as f:
            results["mode"] = f.read().strip()
    else:
        results["mode"] = "unknown"

    info_file = os.path.join(model_path, "model_info.json")
    if os.path.exists(info_file):
        import json
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


if __name__ == "__main__":
    csv_file = sys.argv[1]
    start_date = sys.argv[2] if len(sys.argv) > 2 else None
    end_date = sys.argv[3] if len(sys.argv) > 3 else None

    if start_date and end_date:
        folder_name = f"{start_date}_{end_date}"
    elif start_date:
        folder_name = f"{start_date}_to_latest"
    elif end_date:
        folder_name = f"from_earliest_{end_date}"
    else:
        folder_name = "global"

    model_path = os.path.join(MODEL_DIR, folder_name)

    if not os.path.exists(model_path):
        print(json.dumps({"error": f"No se encontró modelo en {model_path}"}))
        sys.exit(1)

    df = pd.read_csv(csv_file)
    res = score_csv(df, model_path)
    print(json.dumps(res))

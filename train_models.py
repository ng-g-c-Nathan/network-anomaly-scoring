import pandas as pd
import glob
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from preprocess import fit_and_save_preprocessor, transform
import os
from datetime import datetime
import sys
import time 

DATA_FOLDER = "daily/*.csv"
MODEL_DIR = "models" 

MODES = {
    "low":      {"kmeans_n_init": 5,   "iso_n_estimators": 50,  "n_jobs": 1},
    "normal":   {"kmeans_n_init": 10,  "iso_n_estimators": 200, "n_jobs": -1},
    "hardcore": {"kmeans_n_init": 20,  "iso_n_estimators": 500, "n_jobs": -1},
}

def load_all_csvs(pattern, start_date=None, end_date=None):
    files = glob.glob(pattern)
    total_files = len(files)

    
    if start_date or end_date:
        filtered = []
        for f in files:
            base = os.path.basename(f)
            date_str = base.split("_")[1]
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            filtered.append(f)
        files = filtered

    print(f"Se encontraron {len(files)} de {total_files} CSVs en el rango solicitado.")

    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main(start_date=None, end_date=None, mode="normal"):
    start_time = time.perf_counter()  

    print(f"Modo de performance seleccionado: {mode}")
    
    
    params = MODES.get(mode, MODES["normal"])

   
    if start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    df_all = load_all_csvs(DATA_FOLDER, start_date, end_date)
    if df_all.empty:
        print("No se encontraron CSVs en ese rango.")
        return

    print("Entrenando preprocesador...")
    pre = fit_and_save_preprocessor(df_all)

    print("Transformando datos...")
    X = transform(df_all, pre)

    print("Entrenando KMeans...")
    kmeans = KMeans(
        n_clusters=8,
        random_state=42,
        n_init=params["kmeans_n_init"]
    )
    kmeans.fit(X)

    print("Entrenando Isolation Forest...")
    iso = IsolationForest(
        n_estimators=params["iso_n_estimators"],
        contamination="auto",
        random_state=42,
        n_jobs=params["n_jobs"]
    )
    iso.fit(X)

    
    if start_date and end_date:
        folder_name = f"{start_date}_{end_date}"
    elif start_date:
        folder_name = f"{start_date}_to_latest"
    elif end_date:
        folder_name = f"from_earliest_{end_date}"
    else:
        folder_name = "global"

    model_path = os.path.join(MODEL_DIR, folder_name)
    os.makedirs(model_path, exist_ok=True)

    
    joblib.dump(kmeans, os.path.join(model_path, "kmeans.joblib"))
    joblib.dump(iso, os.path.join(model_path, "isoforest.joblib"))
    joblib.dump(pre, os.path.join(model_path, "preprocessor.joblib"))

    mode_file = os.path.join(model_path, "mode.txt")
    with open(mode_file, "w") as f:
        f.write(mode)

    
    info_file = os.path.join(model_path, "model_info.json")
    model_info = {
        "mode": mode,
        "folder_name": folder_name,
        "trained_at": datetime.now().isoformat(),
        "num_rows": len(df_all),
        "n_features": df_all.shape[1]
        }
    with open(info_file, "w") as f:
        import json
        json.dump(model_info, f, indent=2)

    print(f"Modelos y metadata guardados en {model_path}")

    end_time = time.perf_counter()  
    elapsed = end_time - start_time
    print(f"Tiempo total de ejecución: {elapsed:.2f} segundos")  


if __name__ == "__main__":
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
    start_date = sys.argv[2] if len(sys.argv) > 2 else None
    end_date = sys.argv[3] if len(sys.argv) > 3 else None

    
    try:
        if start_date:
            datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    main(start_date, end_date, mode)

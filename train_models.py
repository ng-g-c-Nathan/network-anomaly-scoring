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
import json


# -------------------------------------------------------------------
# Configuración global
# -------------------------------------------------------------------

# Patrón de búsqueda de los CSV diarios
DATA_FOLDER = "daily/*.csv"

# Carpeta donde se almacenan los modelos entrenados
MODEL_DIR = "models"

# Modos de ejecución que controlan el coste computacional
# de los modelos de clustering y detección de anomalías.
MODES = {
    "low":      {"kmeans_n_init": 5,   "iso_n_estimators": 50,  "n_jobs": 1},
    "normal":   {"kmeans_n_init": 10,  "iso_n_estimators": 200, "n_jobs": -1},
    "hardcore": {"kmeans_n_init": 20,  "iso_n_estimators": 500, "n_jobs": -1},
}

# Archivo central de auditoría del proceso de entrenamiento
LOG_FILE = os.path.join(MODEL_DIR, "training_log.json")


# -------------------------------------------------------------------
# Utilidades de logging
# -------------------------------------------------------------------

def update_log(entry_id, update_data):
    """
    Actualiza (o crea) una entrada dentro del archivo training_log.json.

    - Cada entrenamiento se identifica mediante un entry_id.
    - La función no sobreescribe fases previas.
    - Si se incluyen fases nuevas, estas se fusionan con las ya existentes.

    Estructura esperada de update_data:
        {
            "status": "...",
            "finished_at": "...",
            "elapsed_sec": ...,
            "phases": {
                "nombre_fase": {
                    "status": "done | error: ...",
                    "time_sec": float
                }
            }
        }

    Parameters
    ----------
    entry_id : str
        Identificador único del entrenamiento (normalmente el rango de fechas).
    update_data : dict
        Datos parciales a fusionar en el log.
    """

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            log = json.load(f)
    else:
        log = {}

    entry = log.get(entry_id, {})

    # Fusión segura de fases para no perder información previa
    if "phases" in update_data:
        existing_phases = entry.get("phases", {})
        new_phases = update_data.pop("phases")
        entry["phases"] = {**existing_phases, **new_phases}

    entry.update(update_data)
    log[entry_id] = entry

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


# -------------------------------------------------------------------
# Carga de datos
# -------------------------------------------------------------------

def load_all_csvs(pattern, start_date=None, end_date=None):
    """
    Carga y concatena todos los CSV encontrados bajo un patrón de glob.

    Opcionalmente filtra los ficheros por rango de fechas extraído
    del nombre del archivo.

    Se asume que el nombre del fichero contiene la fecha en la
    segunda posición separada por '_' y con formato:

        algo_YYYY-MM-DD_algo.csv

    Parameters
    ----------
    pattern : str
        Patrón glob para localizar los CSV.
    start_date : datetime.date, optional
        Fecha mínima a incluir.
    end_date : datetime.date, optional
        Fecha máxima a incluir.

    Returns
    -------
    pd.DataFrame
        DataFrame con todos los registros concatenados.
        Si no se encuentran CSV válidos, se devuelve un DataFrame vacío.
    """

    files = glob.glob(pattern)

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

    dfs = [pd.read_csv(f) for f in files]

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# -------------------------------------------------------------------
# Pipeline principal de entrenamiento
# -------------------------------------------------------------------

def main(start_date=None, end_date=None, mode="normal"):
    """
    Ejecuta el pipeline completo de entrenamiento:

    1. Carga de CSVs.
    2. Entrenamiento del preprocesador.
    3. Transformación de los datos.
    4. Entrenamiento de KMeans.
    5. Entrenamiento de Isolation Forest.
    6. Persistencia de modelos y metadatos.
    7. Registro detallado de cada fase en training_log.json.

    Parameters
    ----------
    start_date : str or None
        Fecha inicial en formato YYYY-MM-DD.
    end_date : str or None
        Fecha final en formato YYYY-MM-DD.
    mode : str
        Modo de ejecución ("low", "normal", "hardcore").
    """

    start_time = time.perf_counter()
    params = MODES.get(mode, MODES["normal"])

    # Parseo de fechas
    if start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Carga de datos
    df_all = load_all_csvs(DATA_FOLDER, start_date, end_date)

    if df_all.empty:
        return

    # Construcción del identificador lógico del entrenamiento
    folder_name = (
        f"{start_date}_{end_date}" if start_date and end_date else
        f"{start_date}_to_latest" if start_date else
        f"from_earliest_{end_date}" if end_date else
        "global"
    )

    entry_id = folder_name

    # Registro inicial
    update_log(entry_id, {
        "folder_name": folder_name,
        "mode": mode,
        "status": "running",
        "num_rows": len(df_all),
        "n_features": df_all.shape[1],
        "started_at": datetime.now().isoformat(),
        "phases": {
            "inicio": {
                "status": "done",
                "time_sec": 0
            }
        },
        "finished_at": None,
        "elapsed_sec": None
    })

    # ---------------- Fase 1: preprocesador ----------------

    phase_start = time.perf_counter()

    try:
        pre = fit_and_save_preprocessor(df_all)

        phase_end = time.perf_counter()

        update_log(entry_id, {
            "phases": {
                "preprocessor": {
                    "status": "done",
                    "time_sec": phase_end - phase_start
                }
            }
        })

    except Exception as e:
        phase_end = time.perf_counter()

        update_log(entry_id, {
            "phases": {
                "preprocessor": {
                    "status": f"error: {str(e)}",
                    "time_sec": phase_end - phase_start
                }
            }
        })
        raise

    # ---------------- Fase 2: transformación ----------------

    phase_start = time.perf_counter()

    try:
        X = transform(df_all, pre)

        phase_end = time.perf_counter()

        update_log(entry_id, {
            "phases": {
                "transform": {
                    "status": "done",
                    "time_sec": phase_end - phase_start
                }
            }
        })

    except Exception as e:
        phase_end = time.perf_counter()

        update_log(entry_id, {
            "phases": {
                "transform": {
                    "status": f"error: {str(e)}",
                    "time_sec": phase_end - phase_start
                }
            }
        })
        raise

    # ---------------- Fase 3a: KMeans ----------------

    phase_start = time.perf_counter()

    try:
        kmeans = KMeans(
            n_clusters=8,
            random_state=42,
            n_init=params["kmeans_n_init"]
        )

        kmeans.fit(X)

        phase_end = time.perf_counter()

        update_log(entry_id, {
            "phases": {
                "kmeans": {
                    "status": "done",
                    "time_sec": phase_end - phase_start
                }
            }
        })

    except Exception as e:
        phase_end = time.perf_counter()

        update_log(entry_id, {
            "phases": {
                "kmeans": {
                    "status": f"error: {str(e)}",
                    "time_sec": phase_end - phase_start
                }
            }
        })
        raise

    # ---------------- Fase 3b: Isolation Forest ----------------

    phase_start = time.perf_counter()

    try:
        iso = IsolationForest(
            n_estimators=params["iso_n_estimators"],
            contamination="auto",
            random_state=42,
            n_jobs=params["n_jobs"]
        )

        iso.fit(X)

        phase_end = time.perf_counter()

        update_log(entry_id, {
            "phases": {
                "isoforest": {
                    "status": "done",
                    "time_sec": phase_end - phase_start
                }
            }
        })

    except Exception as e:
        phase_end = time.perf_counter()

        update_log(entry_id, {
            "phases": {
                "isoforest": {
                    "status": f"error: {str(e)}",
                    "time_sec": phase_end - phase_start
                }
            }
        })
        raise

    # ---------------- Persistencia de modelos ----------------

    model_path = os.path.join(MODEL_DIR, folder_name)
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(kmeans, os.path.join(model_path, "kmeans.joblib"))
    joblib.dump(iso, os.path.join(model_path, "isoforest.joblib"))
    joblib.dump(pre, os.path.join(model_path, "preprocessor.joblib"))

    with open(os.path.join(model_path, "mode.txt"), "w") as f:
        f.write(mode)

    with open(os.path.join(model_path, "model_info.json"), "w") as f:
        json.dump({
            "mode": mode,
            "folder_name": folder_name,
            "trained_at": datetime.now().isoformat(),
            "num_rows": len(df_all),
            "n_features": df_all.shape[1]
        }, f, indent=2)

    # ---------------- Cierre del log ----------------

    end_time = time.perf_counter()

    update_log(entry_id, {
        "status": "done",
        "finished_at": datetime.now().isoformat(),
        "elapsed_sec": end_time - start_time
    })


# -------------------------------------------------------------------
# Entrada por línea de comandos
# -------------------------------------------------------------------

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
        sys.exit(1)

    main(start_date, end_date, mode)

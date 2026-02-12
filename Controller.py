import json
import uuid
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys
import pandas as pd
from io import StringIO
import score_csv
import os


# -------------------------------------------------------------------
# Configuración global
# -------------------------------------------------------------------

# Archivo histórico de ejecuciones de análisis
HISTORY_FILE = "analysis_history.json"

# Carpeta que contiene los CSV diarios
DAILY_DIR = Path("daily")

# Carpeta base de modelos
MODEL_DIR = "models"


# -------------------------------------------------------------------
# Utilidades de tiempo y fechas
# -------------------------------------------------------------------

def now():
    """
    Devuelve la fecha y hora actual en formato ISO-8601 en UTC.

    Returns
    -------
    str
        Timestamp ISO-8601 con zona horaria.
    """
    return datetime.now(timezone.utc).isoformat()


# -------------------------------------------------------------------
# Gestión del histórico de ejecuciones
# -------------------------------------------------------------------

def load_history():
    """
    Carga el archivo de histórico de análisis.

    Si el archivo no existe o está corrupto, se devuelve una lista vacía.

    Returns
    -------
    list
        Lista de registros de ejecución.
    """
    p = Path(HISTORY_FILE)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history):
    """
    Guarda de forma atómica el histórico de análisis.

    Se escribe primero un archivo temporal y luego se reemplaza
    el archivo real para evitar corrupciones.

    Parameters
    ----------
    history : list
        Lista completa de registros a persistir.
    """
    tmp = Path(HISTORY_FILE + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    tmp.replace(HISTORY_FILE)


def create_record(csv_file):
    """
    Crea una nueva entrada de histórico para un análisis.

    Parameters
    ----------
    csv_file : str
        Nombre lógico del CSV o rango analizado.

    Returns
    -------
    dict
        Registro inicial del análisis.
    """
    return {
        "job_id": str(uuid.uuid4()),
        "csv_file": str(csv_file),
        "created_at": now(),
        "started_at": None,
        "finished_at": None,
        "progress": 0,
        "stage": "queued",
        "status": "PENDING",
        "duration_seconds": None,
        "duration_human": None,
        "duration_category": None,
        "result": None,
        "error": None
    }


def update_record(history, job_id, **fields):
    """
    Actualiza un registro del histórico identificado por job_id.

    Parameters
    ----------
    history : list
        Lista de registros.
    job_id : str
        Identificador del análisis.
    fields : dict
        Campos a actualizar.
    """
    for r in history:
        if r["job_id"] == job_id:
            r.update(fields)
            return
    raise RuntimeError("job_id no encontrado en history")


# -------------------------------------------------------------------
# Cálculo de duraciones
# -------------------------------------------------------------------

def duration_info(start_iso, end_iso):
    """
    Calcula duración en segundos, versión legible y categoría
    a partir de dos timestamps ISO.

    Parameters
    ----------
    start_iso : str
    end_iso : str

    Returns
    -------
    tuple(float, str, str)
        (segundos, texto legible, categoría)
    """

    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    seconds = (end - start).total_seconds()

    if seconds < 5 * 60:
        category = "SHORT"
    elif seconds < 60 * 60:
        category = "MEDIUM"
    elif seconds < 24 * 60 * 60:
        category = "LONG"
    else:
        category = "VERY_LONG"

    if seconds < 60:
        human = f"{seconds:.1f} seconds"
    elif seconds < 3600:
        human = f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        human = f"{seconds/3600:.2f} hours"
    else:
        human = f"{seconds/86400:.2f} days"

    return seconds, human, category


def extract_duration_from_csv_name(csv_name: str):
    """
    Extrae una duración estimada desde el nombre del CSV si
    contiene el patrón:

        (..._X_minutes)

    Parameters
    ----------
    csv_name : str

    Returns
    -------
    tuple or (None, str, None)
        (segundos, texto legible, categoría)
    """

    import re

    m = re.search(r"\(([\d\.]+)_minutes\)", csv_name)
    if m:
        minutes = float(m.group(1))
        seconds = minutes * 60
        human = f"{minutes:.1f} minutes"

        if seconds < 5 * 60:
            category = "SHORT"
        elif seconds < 60 * 60:
            category = "MEDIUM"
        elif seconds < 24 * 60 * 60:
            category = "LONG"
        else:
            category = "VERY_LONG"

        return seconds, human, category
    else:
        return None, "INDETERMINADO", None


def duration_from_dates(start_str, end_str):
    """
    Calcula duración a partir de dos fechas ISO.

    Nota:
    Esta función es equivalente a duration_info pero recibe strings
    de fechas y no timestamps completos.

    Parameters
    ----------
    start_str : str
    end_str : str

    Returns
    -------
    tuple(float, str, str)
    """

    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    seconds = (end - start).total_seconds()

    if seconds < 5 * 60:
        category = "SHORT"
    elif seconds < 60 * 60:
        category = "MEDIUM"
    elif seconds < 24 * 60 * 60:
        category = "LONG"
    else:
        category = "VERY_LONG"

    if seconds < 60:
        human = f"{seconds:.1f} seconds"
    elif seconds < 3600:
        human = f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        human = f"{seconds/3600:.2f} hours"
    else:
        human = f"{seconds/86400:.2f} days"

    return seconds, human, category


# -------------------------------------------------------------------
# Ejecución del scoring
# -------------------------------------------------------------------

def run_score(df, start_date=None, end_date=None):
    """
    Ejecuta el scoring sobre un DataFrame utilizando el modelo
    correspondiente al rango de fechas indicado.

    Parameters
    ----------
    df : pd.DataFrame
        Datos a evaluar.
    start_date : str or None
        Fecha inicial del modelo (YYYY-MM-DD).
    end_date : str or None
        Fecha final del modelo (YYYY-MM-DD).

    Returns
    -------
    dict
        Resultado agregado del scoring.
    """

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
        raise RuntimeError(f"No se encontró modelo en {model_path}")

    return score_csv.score_csv(df, model_path)


# -------------------------------------------------------------------
# Carga de CSVs diarios por rango
# -------------------------------------------------------------------

def load_csvs_in_range(start_str, end_str):
    """
    Carga y concatena todos los CSV diarios cuyo nombre contiene
    una fecha dentro del rango indicado.

    Se asume que el nombre del fichero sigue el patrón:

        algo_YYYY-MM-DD.csv

    Parameters
    ----------
    start_str : str
        Fecha inicial ISO.
    end_str : str
        Fecha final ISO.

    Returns
    -------
    pd.DataFrame
        DataFrame concatenado.
    """

    start_date = datetime.fromisoformat(start_str).date()
    end_date = datetime.fromisoformat(end_str).date()

    all_dfs = []

    for csv_path in DAILY_DIR.glob("*.csv"):
        try:
            date_part = csv_path.stem.split("_")[1]
            file_date = datetime.fromisoformat(date_part).date()

            if start_date <= file_date <= end_date:
                all_dfs.append(pd.read_csv(csv_path))

        except Exception:
            continue

    if not all_dfs:
        raise RuntimeError("No se encontraron CSVs en ese rango")

    return pd.concat(all_dfs, ignore_index=True)


# -------------------------------------------------------------------
# Orquestador principal
# -------------------------------------------------------------------

def main():
    """
    Orquesta una ejecución completa de análisis.

    Permite dos modos de entrada:

    1) Analizar un CSV concreto.
    2) Analizar un rango de fechas de los CSV diarios.

    En ambos casos:
        - registra el proceso en analysis_history.json
        - ejecuta el scoring
        - guarda resultados, duración y estado
        - devuelve un JSON por stdout
    """

    if len(sys.argv) < 2:
        print("Uso: python main_analysis.py archivo.csv | start_date end_date")
        sys.exit(1)

    history = load_history()

    start_date = end_date = None
    model_end = model_start = None
    csv_file = None

    # ---------------------------------------------------------------
    # Modo 1: se recibe un CSV explícito
    # ---------------------------------------------------------------

    if sys.argv[1].endswith(".csv"):

        csv_file = Path(sys.argv[1])

        if not csv_file.exists():
            print("No existe el CSV:", csv_file)
            sys.exit(1)

        df = pd.read_csv(csv_file)
        record_name = csv_file.name

        # Selección opcional del modelo a usar
        if len(sys.argv) == 3:
            model_start, model_end = sys.argv[2], sys.argv[2]
        elif len(sys.argv) == 4:
            model_start, model_end = sys.argv[2:4]

        # Estimación de duración desde el nombre del CSV
        secs, human, cat = extract_duration_from_csv_name(record_name)

    # ---------------------------------------------------------------
    # Modo 2: se recibe un rango de fechas de CSV diarios
    # ---------------------------------------------------------------

    else:

        if len(sys.argv) < 3:
            print("Datos incorrectos")
            sys.exit(1)

        data_start, data_end = sys.argv[1:3]
        df = load_csvs_in_range(data_start, data_end)
        record_name = f"{data_start}_to_{data_end}"

        if len(sys.argv) == 2:
            model_start, model_end = None, None
        elif len(sys.argv) == 3:
            model_start, model_end = sys.argv[1], sys.argv[2]
        elif len(sys.argv) == 4:
            model_start, model_end = sys.argv[3], sys.argv[3]
        else:
            model_start, model_end = sys.argv[3:5]

        start_dt = datetime.fromisoformat(data_start)
        end_dt = datetime.fromisoformat(data_end) + timedelta(days=1) - timedelta(seconds=1)

        secs, human, cat = duration_info(
            start_dt.isoformat(),
            end_dt.isoformat()
        )

    # ---------------------------------------------------------------
    # Registro inicial del análisis
    # ---------------------------------------------------------------

    record = create_record(record_name)
    history.append(record)
    save_history(history)
    job_id = record["job_id"]

    try:
        started_at = now()

        update_record(
            history,
            job_id,
            status="RUNNING",
            started_at=started_at,
            stage="running",
            progress=1
        )

        save_history(history)

        # -----------------------------------------------------------
        # Ejecución del scoring
        # -----------------------------------------------------------

        result = run_score(
            df,
            start_date=model_start,
            end_date=model_end
        )

        # -----------------------------------------------------------
        # Carga de metadatos del modelo utilizado
        # -----------------------------------------------------------

        folder_name = "global"
        if start_date and end_date:
            folder_name = f"{start_date}_{end_date}"
        elif start_date:
            folder_name = f"{start_date}_to_latest"
        elif end_date:
            folder_name = f"from_earliest_{end_date}"

        model_info_file = os.path.join(
            "models",
            folder_name,
            "model_info.json"
        )

        if os.path.exists(model_info_file):
            with open(model_info_file, "r") as f:
                model_info = json.load(f)

            result.update({
                "model_folder": model_info.get("folder_name", "unknown"),
                "trained_at": model_info.get("trained_at", "unknown"),
                "trained_rows": model_info.get("num_rows", None),
                "trained_features": model_info.get("n_features", None),
                "mode": model_info.get("mode", result.get("mode", "unknown"))
            })

        finished_at = now()

        # Recalcular duración en el caso de análisis por fechas
        if not csv_file and start_date and end_date:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date) + timedelta(days=1) - timedelta(seconds=1)
            secs, human, cat = duration_info(
                start_dt.isoformat(),
                end_dt.isoformat()
            )

        update_record(
            history,
            job_id,
            status="FINISHED",
            finished_at=finished_at,
            stage="done",
            progress=100,
            result=result,
            duration_seconds=secs,
            duration_human=human,
            duration_category=cat
        )

        save_history(history)

        print(json.dumps({
            "job_id": job_id,
            "status": "FINISHED",
            "result": result,
            "duration_seconds": secs,
            "duration_human": human,
            "duration_category": cat
        }))

    except Exception as e:

        finished_at = now()

        if record["started_at"]:
            secs, human, cat = duration_info(
                record["started_at"],
                finished_at
            )
        else:
            secs, human, cat = 0, "0 seconds", "SHORT"

        update_record(
            history,
            job_id,
            status="ERROR",
            finished_at=finished_at,
            stage="error",
            error=str(e),
            duration_seconds=secs,
            duration_human=human,
            duration_category=cat
        )

        save_history(history)

        print(json.dumps({
            "job_id": job_id,
            "status": "ERROR",
            "error": str(e),
            "duration_seconds": secs,
            "duration_human": human,
            "duration_category": cat
        }))

        sys.exit(2)


# -------------------------------------------------------------------
# Punto de entrada
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()

import pandas as pd
import glob
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from preprocess import fit_and_save_preprocessor, transform
import numpy as np
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

# Split train/val
VAL_SIZE   = 0.20
RANDOM_STATE = 42


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

# Máximo de filas a cargar en memoria. Con 12 GB de RAM y tráfico de red
# (~50-100 columnas numéricas) 500k filas ocupa ~400 MB — margen seguro
# para el preprocesador, PCA y los dos modelos conviviendo en memoria.
MAX_ROWS = 500_000

def load_all_csvs(pattern, start_date=None, end_date=None):
    """
    Carga y concatena los CSV encontrados bajo un patrón de glob,
    con muestreo proporcional por archivo para no superar MAX_ROWS.

    Estrategia anti-MemoryError
    ---------------------------
    1. Cuenta el total de filas de todos los archivos sin cargarlos
       (lee solo el header + wc de líneas via chunking liviano).
    2. Calcula un sample_rate = MAX_ROWS / total_rows.
    3. Lee cada CSV con ese sample_rate usando skiprows aleatorio,
       de forma que cada archivo aporta proporcionalmente.
    4. Concatena solo los samples — nunca el dataset completo.

    Esto garantiza que el DataFrame resultante nunca supera MAX_ROWS
    independientemente de cuántos CSVs haya.

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
        DataFrame muestreado con todos los registros concatenados.
        Si no se encuentran CSV válidos, se devuelve un DataFrame vacío.
    """

    files = glob.glob(pattern)

    if start_date or end_date:
        filtered = []
        for f in files:
            base = os.path.basename(f)
            try:
                date_str = base.split("_")[1]
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except (IndexError, ValueError):
                filtered.append(f)
                continue

            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

            filtered.append(f)

        files = filtered

    if not files:
        return pd.DataFrame()

    # --- Paso 1: contar filas totales sin cargar datos ---
    # Usamos chunking mínimo: solo leemos el índice para contar.
    row_counts = {}
    for f in files:
        count = 0
        for chunk in pd.read_csv(f, usecols=[0], chunksize=10_000):
            count += len(chunk)
        row_counts[f] = count

    total_rows  = sum(row_counts.values())
    sample_rate = min(1.0, MAX_ROWS / max(total_rows, 1))

    print(f"[load] {len(files)} archivos — {total_rows:,} filas totales — "
          f"sample_rate: {sample_rate:.2%} → ~{int(total_rows * sample_rate):,} filas cargadas")

    # --- Paso 2: leer cada CSV muestreando proporcionalmente ---
    samples = []
    for f in files:
        n_rows   = row_counts[f]
        n_sample = max(1, int(n_rows * sample_rate))

        if sample_rate >= 1.0:
            samples.append(pd.read_csv(f))
        else:
            # Generamos índices aleatorios a SALTAR (skiprows)
            # fila 0 es el header — trabajamos sobre filas 1..n_rows
            all_data_rows = np.arange(1, n_rows + 1)
            n_skip        = n_rows - n_sample
            skip_idx      = set(
                np.random.choice(all_data_rows, size=n_skip, replace=False).tolist()
            )
            samples.append(pd.read_csv(f, skiprows=lambda i: i in skip_idx))

    return pd.concat(samples, ignore_index=True)


# -------------------------------------------------------------------
# Diagnóstico de overfitting / underfitting
# -------------------------------------------------------------------

def _diagnose_kmeans(kmeans, X_train, X_val):
    """
    Evalúa KMeans en train y val usando inercia normalizada y
    silhouette score.  Devuelve métricas y un diagnóstico textual.

    Criterios de diagnóstico
    ------------------------
    * silhouette_train < 0.25
      → UNDERFITTING: clusters sin estructura clara ni en train.

    * inertia_gap_pct > 100 % AND silhouette_gap > 0.10
      → OVERFITTING: la inercia sola no es diagnóstica en datos dispersos
        (tráfico de red tiene features con rangos muy amplios que inflan
        la inercia absoluta). Solo se reporta overfitting si AMBAS condiciones
        se cumplen simultáneamente.

    * silhouette_gap > 0.10  (sin gap de inercia extremo)
      → ADVERTENCIA: los clusters son más nítidos en train que en val,
        puede indicar sobreajuste leve o simplemente varianza natural.

    Parameters
    ----------
    kmeans : KMeans entrenado
    X_train, X_val : np.ndarray  (ya reducidos con PCA)

    Returns
    -------
    dict con métricas y diagnóstico
    """

    n_train = X_train.shape[0]
    n_val   = X_val.shape[0]

    # Inercia normalizada por número de puntos
    inertia_train = kmeans.inertia_ / n_train

    labels_val    = kmeans.predict(X_val)
    dist_sq_val   = np.min(
        np.sum((X_val[:, None, :] - kmeans.cluster_centers_[None, :, :]) ** 2, axis=2),
        axis=1
    )
    inertia_val   = dist_sq_val.mean()

    inertia_gap_pct = 100 * (inertia_val - inertia_train) / max(inertia_train, 1e-9)

    # Silhouette score (muestrea para no explotar en memoria)
    sample_size = min(5000, n_train)
    idx_train   = np.random.choice(n_train, sample_size, replace=False)
    labels_train = kmeans.labels_

    sil_train = float(silhouette_score(X_train[idx_train], labels_train[idx_train]))

    sample_size_val = min(5000, n_val)
    idx_val = np.random.choice(n_val, sample_size_val, replace=False)
    sil_val = float(silhouette_score(X_val[idx_val], labels_val[idx_val]))

    sil_gap = sil_train - sil_val

    # --- Diagnóstico ---
    issues = []

    if sil_train < 0.25:
        issues.append("UNDERFITTING: silhouette_train bajo (< 0.25) — los clusters no tienen "
                      "estructura clara. Considera aumentar n_clusters o revisar features.")

    if inertia_gap_pct > 100 and sil_gap > 0.10:
        issues.append(f"OVERFITTING: inercia val {inertia_gap_pct:.1f}% mayor que train "
                      f"Y silhouette_gap = {sil_gap:.3f}. "
                      "El modelo no generaliza bien la geometría de los clusters.")
    elif sil_gap > 0.10:
        issues.append(f"ADVERTENCIA: silhouette_gap = {sil_gap:.3f} (> 0.10) — clusters más "
                      "nítidos en train que en val. Puede ser sobreajuste leve o varianza natural.")

    if not issues:
        verdict = "OK — KMeans generaliza bien al conjunto de validación."
    else:
        verdict = " | ".join(issues)

    return {
        "inertia_train_normalized": float(inertia_train),
        "inertia_val_normalized":   float(inertia_val),
        "inertia_gap_pct":          float(inertia_gap_pct),
        "silhouette_train":         sil_train,
        "silhouette_val":           sil_val,
        "silhouette_gap":           float(sil_gap),
        "diagnosis":                verdict,
    }


def _diagnose_isoforest(iso, X_train, X_val):
    """
    Evalúa Isolation Forest comparando la distribución de anomaly scores
    entre train y val.

    Criterios de diagnóstico
    ------------------------
    * mean_train > -0.10
      → UNDERFITTING: la frontera de normalidad no es clara ni en train.

    * mean_score_gap > 3 × std_train (z-score normalizado)
      → OVERFITTING real: val cae sistemáticamente fuera de la densidad
        de train más allá del ruido estadístico esperado.
        Se usa el z-score en lugar de un umbral absoluto porque con
        contamination dinámico bajo (≈2%) la escala de scores cambia.

    * anomaly_rate_val > anomaly_rate_train * 3 AND anomaly_rate_val > 0.10
      → DRIFT: val tiene muchas más anomalías que train Y la tasa absoluta
        es alta (>10%). Con contamination dinámico, anomaly_rate_train es
        casi exactamente contamination por construcción — necesitamos
        la condición absoluta para evitar falsos positivos.

    Parameters
    ----------
    iso : IsolationForest entrenado
    X_train, X_val : np.ndarray

    Returns
    -------
    dict con métricas y diagnóstico
    """

    scores_train = iso.score_samples(X_train)   # más negativo = más anómalo
    scores_val   = iso.score_samples(X_val)

    mean_train = float(scores_train.mean())
    mean_val   = float(scores_val.mean())
    std_train  = float(scores_train.std())
    std_val    = float(scores_val.std())

    mean_score_gap = mean_train - mean_val      # >0 → val más anómalo que train

    # Z-score del gap: cuántas desviaciones estándar se aleja val de train
    # Normalizar por std_train hace el umbral robusto al contamination usado
    gap_zscore = mean_score_gap / max(std_train, 1e-9)

    # Tasa de anomalías (predict devuelve -1 para anomalías, 1 para normal)
    anomaly_rate_train = float((iso.predict(X_train) == -1).mean())
    anomaly_rate_val   = float((iso.predict(X_val)   == -1).mean())

    # Distancia KL aproximada entre distribuciones de scores
    ks_proxy = abs(mean_score_gap) / max(std_train, 1e-9)

    # --- Diagnóstico ---
    issues = []

    if mean_train > -0.10:
        issues.append("UNDERFITTING: anomaly scores muy cercanos a 0 incluso en train. "
                      "El modelo no aprendió una frontera clara de normalidad. "
                      "Aumenta n_estimators o revisa la calidad del dataset.")

    if gap_zscore > 3.0:
        issues.append(f"OVERFITTING: val cae {gap_zscore:.1f}σ por debajo de train en anomaly score. "
                      "El modelo está demasiado ajustado a la densidad exacta de train.")

    if anomaly_rate_val > anomaly_rate_train * 3 and anomaly_rate_val > 0.10:
        issues.append(f"DRIFT: tasa de anomalías val ({anomaly_rate_val:.1%}) es más de 3× "
                      f"la de train ({anomaly_rate_train:.1%}) y supera el 10% absoluto. "
                      "El tráfico de val puede ser cualitativamente diferente al de train.")

    if not issues:
        verdict = "OK — Isolation Forest generaliza bien al conjunto de validación."
    else:
        verdict = " | ".join(issues)

    return {
        "mean_score_train":    mean_train,
        "mean_score_val":      mean_val,
        "std_score_train":     std_train,
        "std_score_val":       std_val,
        "mean_score_gap":      float(mean_score_gap),
        "gap_zscore":          float(gap_zscore),
        "ks_proxy":            float(ks_proxy),
        "anomaly_rate_train":  anomaly_rate_train,
        "anomaly_rate_val":    anomaly_rate_val,
        "diagnosis":           verdict,
    }


def _print_validation_report(kmeans_diag, iso_diag, n_train, n_val):
    """
    Imprime en consola un resumen legible del diagnóstico de validación.
    """

    SEP  = "=" * 70
    sep2 = "-" * 70

    print(f"\n{SEP}")
    print("  REPORTE DE VALIDACIÓN (80/20 SPLIT)")
    print(SEP)
    print(f"  Train: {n_train:,} muestras   |   Val: {n_val:,} muestras")

    print(f"\n{sep2}")
    print("  [KMEANS]")
    print(sep2)
    print(f"  Inercia normalizada  — train: {kmeans_diag['inertia_train_normalized']:.4f}  "
          f"val: {kmeans_diag['inertia_val_normalized']:.4f}  "
          f"gap: {kmeans_diag['inertia_gap_pct']:+.1f}%")
    print(f"  Silhouette           — train: {kmeans_diag['silhouette_train']:.4f}  "
          f"val: {kmeans_diag['silhouette_val']:.4f}  "
          f"gap: {kmeans_diag['silhouette_gap']:+.4f}")
    print(f"\n  → {kmeans_diag['diagnosis']}")

    print(f"\n{sep2}")
    print("  [ISOLATION FOREST]")
    print(sep2)
    print(f"  Anomaly score medio  — train: {iso_diag['mean_score_train']:.4f}  "
          f"val: {iso_diag['mean_score_val']:.4f}  "
          f"gap: {iso_diag['mean_score_gap']:+.4f}  "
          f"({iso_diag['gap_zscore']:.2f}σ)")
    print(f"  Tasa de anomalías    — train: {iso_diag['anomaly_rate_train']:.2%}  "
          f"val: {iso_diag['anomaly_rate_val']:.2%}")
    print(f"\n  → {iso_diag['diagnosis']}")

    print(f"\n{SEP}\n")


# -------------------------------------------------------------------
# Pipeline principal de entrenamiento
# -------------------------------------------------------------------

def main(start_date=None, end_date=None, mode="normal"):
    """
    Ejecuta el pipeline completo de entrenamiento:

    1. Carga de CSVs.
    2. Split 80/20 (train / val).
    3. Entrenamiento del preprocesador sobre train.
    4. Transformación de train y val.
    5. PCA (reducción dimensional al 95% de varianza) sobre train.
    6. Entrenamiento de KMeans.
    7. Entrenamiento de Isolation Forest.
    8. Diagnóstico overfitting / underfitting sobre val.
    9. Persistencia de modelos y metadatos.
    10. Registro detallado de cada fase en training_log.json.

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

    # ----------------------------------------------------------------
    # Split 80/20 — se hace sobre el DataFrame crudo antes de
    # cualquier preprocesamiento para evitar data leakage.
    # ----------------------------------------------------------------
    df_train, df_val = train_test_split(
        df_all,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

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
        "num_rows_total": len(df_all),
        "num_rows_train": len(df_train),
        "num_rows_val":   len(df_val),
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

    # ---------------- Fase 1: preprocesador (fit solo sobre train) ----------------

    phase_start = time.perf_counter()

    try:
        # El preprocesador se ajusta SOLO con datos de train para evitar
        # data leakage: los parámetros de escalado (mediana, IQR) no
        # deben ver los datos de validación.
        pre = fit_and_save_preprocessor(df_train)

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
        X_train = transform(df_train, pre)
        X_val   = transform(df_val, pre)

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

    # ---------------- Fase 2.5: PCA ----------------

    phase_start = time.perf_counter()

    try:
        # PCA se ajusta sobre train y se aplica a ambos splits.
        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca   = pca.transform(X_val)

        update_log(entry_id, {
            "phases": {
                "pca": {
                    "status": "done",
                    "time_sec": time.perf_counter() - phase_start,
                    "n_components_in": X_train.shape[1],
                    "n_components_out": X_train_pca.shape[1],
                    "variance_explained": float(pca.explained_variance_ratio_.sum())
                }
            }
        })

    except Exception as e:
        update_log(entry_id, {
            "phases": {
                "pca": {
                    "status": f"error: {str(e)}",
                    "time_sec": time.perf_counter() - phase_start
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

        kmeans.fit(X_train_pca)

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
    # contamination se calcula dinámicamente como el percentil 2 de los
    # anomaly scores en train — solo el 2% más extremo se clasifica como
    # anomalía. Mucho más realista que "auto" (= 0.1 interno de sklearn)
    # para tráfico de red sano donde las anomalías reales son raras.

    phase_start = time.perf_counter()

    try:
        # Primer fit con "auto" para obtener los scores base
        iso = IsolationForest(
            n_estimators=params["iso_n_estimators"],
            contamination="auto",
            random_state=42,
            n_jobs=params["n_jobs"]
        )
        iso.fit(X_train_pca)

        # Recalculamos contamination como percentil 2 de scores en train
        scores_train  = iso.score_samples(X_train_pca)
        contamination = float(np.clip(np.mean(scores_train < np.percentile(scores_train, 2)), 0.001, 0.5))

        # Reentrenamos con el contamination dinámico
        iso = IsolationForest(
            n_estimators=params["iso_n_estimators"],
            contamination=contamination,
            random_state=42,
            n_jobs=params["n_jobs"]
        )
        iso.fit(X_train_pca)

        phase_end = time.perf_counter()

        update_log(entry_id, {
            "phases": {
                "isoforest": {
                    "status": "done",
                    "time_sec": phase_end - phase_start,
                    "contamination_used": contamination,
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

    # ---------------- Fase 4: diagnóstico overfitting/underfitting ----------------
    # Inicializamos antes del try para que el scope sea visible en la
    # persistencia aunque el diagnóstico falle parcialmente.

    kmeans_diag = {}
    iso_diag    = {}

    phase_start = time.perf_counter()

    try:
        kmeans_diag = _diagnose_kmeans(kmeans, X_train_pca, X_val_pca)
        iso_diag    = _diagnose_isoforest(iso, X_train_pca, X_val_pca)

        _print_validation_report(
            kmeans_diag, iso_diag,
            n_train=X_train_pca.shape[0],
            n_val=X_val_pca.shape[0],
        )

        update_log(entry_id, {
            "phases": {
                "validation_diagnosis": {
                    "status": "done",
                    "time_sec": time.perf_counter() - phase_start,
                    "kmeans": kmeans_diag,
                    "isoforest": iso_diag,
                }
            }
        })

    except Exception as e:
        update_log(entry_id, {
            "phases": {
                "validation_diagnosis": {
                    "status": f"error: {str(e)}",
                    "time_sec": time.perf_counter() - phase_start,
                }
            }
        })
        # El diagnóstico no es bloqueante — si falla no paramos el pipeline
        print(f"[WARN] Diagnóstico de validación falló: {e}")

    # ---------------- Persistencia de modelos ----------------
    # Se guardan los objetos entrenados sobre train (80%).
    # El diagnóstico corresponde exactamente a estos mismos objetos:
    # lo que se mide es lo que se guarda — sin reentrenamiento.

    model_path = os.path.join(MODEL_DIR, folder_name)
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(kmeans, os.path.join(model_path, "kmeans.joblib"))
    joblib.dump(iso,    os.path.join(model_path, "isoforest.joblib"))
    joblib.dump(pca,    os.path.join(model_path, "pca.joblib"))
    joblib.dump(pre,    os.path.join(model_path, "preprocessor.joblib"))

    with open(os.path.join(model_path, "mode.txt"), "w") as f:
        f.write(mode)

    with open(os.path.join(model_path, "model_info.json"), "w") as f:
        json.dump({
            "mode": mode,
            "folder_name": folder_name,
            "trained_at": datetime.now().isoformat(),
            "num_rows": len(df_all),
            "num_rows_total": len(df_all),
            "num_rows_train": len(df_train),
            "num_rows_val":   len(df_val),
            "n_features": df_all.shape[1],
            "pca_components_in":    int(X_train.shape[1]),
            "pca_components_out":   int(X_train_pca.shape[1]),
            "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
            "validation": {
                "kmeans":    kmeans_diag,
                "isoforest": iso_diag,
            },
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
            datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as e:
        sys.exit(1)

    main(start_date, end_date, mode)
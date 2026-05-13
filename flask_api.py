"""
flask_api.py
------------
API HTTP que expone el pipeline de ML para ser consumido por Spring Boot.

Endpoints:
    POST /score          → ejecuta Controller.py (análisis de un CSV)
    POST /train          → ejecuta train_models.py (entrenamiento)
    GET  /history        → devuelve analysis_history.json
    GET  /models_info    → metadatos de todos los modelos entrenados
    GET  /training_log   → log del último entrenamiento
    GET  /health         → healthcheck
"""

from flask import Flask, request, jsonify
import subprocess
import threading
import json
import os
from pathlib import Path

app = Flask(__name__)

# -------------------------------------------------------------------
# Rutas base
# -------------------------------------------------------------------
BASE_DIR     = Path("/app")
DAILY_DIR    = BASE_DIR / "daily"
MODELS_DIR   = BASE_DIR / "models"
HISTORY_FILE = BASE_DIR / "analysis_history.json"
TRAIN_LOG    = MODELS_DIR / "training_log.json"


# -------------------------------------------------------------------
# Utilidad: ejecutar proceso en background (fire & forget)
# -------------------------------------------------------------------
def run_async(cmd, cwd):
    def _run():
        try:
            subprocess.run(cmd, cwd=str(cwd), capture_output=True)
        except Exception as e:
            print(f"[ERROR] proceso falló: {e}")
    threading.Thread(target=_run, daemon=True).start()


# -------------------------------------------------------------------
# POST /score
# Body: { "csv_file": "traffic_2026-05-13_....csv", "range": "global" }
# -------------------------------------------------------------------
@app.route("/score", methods=["POST"])
def score():
    data = request.get_json(force=True) or {}
    csv_file = data.get("csv_file")

    if not csv_file:
        return jsonify({"error": "csv_file requerido"}), 400

    if not csv_file.lower().endswith(".csv"):
        csv_file += ".csv"

    csv_path = DAILY_DIR / csv_file

    if not csv_path.exists():
        return jsonify({"error": f"CSV no encontrado: {csv_path}"}), 404

    # Construir comando igual que AnalysisService.java
    cmd = ["python", "Controller.py", str(csv_path)]

    range_val = data.get("range", "")
    if range_val and range_val.lower() != "global":
        parts = range_val.split("_", 1) if "_" in range_val else range_val.split()
        if len(parts) == 2:
            cmd += parts

    run_async(cmd, BASE_DIR)
    return jsonify({"status": "accepted"}), 202


# -------------------------------------------------------------------
# POST /train
# Body: { "mode": "normal", "fromDate": "2026-01-01", "toDate": "2026-05-13" }
# -------------------------------------------------------------------
@app.route("/train", methods=["POST"])
def train():
    data = request.get_json(force=True) or {}
    mode = data.get("mode", "").strip()

    if not mode:
        return jsonify({"error": "mode requerido"}), 400

    if mode not in ("low", "normal", "hardcore"):
        return jsonify({"error": "mode debe ser: low, normal, hardcore"}), 400

    cmd = ["python", "train_models.py", mode]

    from_date = data.get("fromDate", "").strip()
    to_date   = data.get("toDate", "").strip()

    if from_date and to_date:
        cmd += [from_date, to_date]

    run_async(cmd, BASE_DIR)
    return jsonify({"status": "accepted", "message": "Entrenamiento iniciado"}), 202


# -------------------------------------------------------------------
# GET /history
# -------------------------------------------------------------------
@app.route("/history", methods=["GET"])
def history():
    if not HISTORY_FILE.exists():
        return jsonify([]), 200
    with HISTORY_FILE.open("r", encoding="utf-8") as f:
        return jsonify(json.load(f)), 200


# -------------------------------------------------------------------
# GET /models_info
# -------------------------------------------------------------------
@app.route("/models_info", methods=["GET"])
def models_info():
    if not MODELS_DIR.exists():
        return jsonify([]), 200

    result = []
    for folder in sorted(MODELS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        info_file = folder / "model_info.json"
        if info_file.exists():
            with info_file.open("r") as f:
                info = json.load(f)
            info["folder"] = folder.name
            result.append(info)

    return jsonify(result), 200


# -------------------------------------------------------------------
# GET /training_log
# -------------------------------------------------------------------
@app.route("/training_log", methods=["GET"])
def training_log():
    if not TRAIN_LOG.exists():
        return jsonify({"error": "Log no disponible aún"}), 404
    with TRAIN_LOG.open("r", encoding="utf-8") as f:
        return jsonify(json.load(f)), 200


# -------------------------------------------------------------------
# GET /health
# -------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

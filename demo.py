"""
gui.py
------
Interfaz Tkinter ultra-sencilla para:
  1. Ver las fechas disponibles en daily/
  2. Lanzar entrenamientos (global, un día, rango, todos contra todos)
  3. Analizar un CSV contra todos los modelos disponibles

Sin estilos, sin colores, sin alineación fancy. Solo widgets.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import glob
import json
import subprocess
import sys
import threading
import random
import itertools
from datetime import datetime, date, timedelta
from pathlib import Path


# -------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------

DAILY_DIR  = Path("daily")
MODEL_DIR  = Path("models")
SCRIPT_DIR = Path(__file__).parent

TRAIN_SCRIPT      = SCRIPT_DIR / "train_models.py"
CONTROLLER_SCRIPT = SCRIPT_DIR / "controller.py"


# -------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------

def get_dates_in_daily():
    """Devuelve lista ordenada de fechas únicas encontradas en daily/*.csv"""
    dates = set()
    for f in DAILY_DIR.glob("*.csv"):
        try:
            # Patrón: algo_YYYY-MM-DD.csv
            date_str = f.stem.split("_")[1]
            dates.add(date_str)
        except IndexError:
            pass
    return sorted(dates)


def get_available_models():
    """Devuelve lista de carpetas de modelos disponibles en models/"""
    if not MODEL_DIR.exists():
        return []
    return sorted([
        d.name for d in MODEL_DIR.iterdir()
        if d.is_dir() and (d / "kmeans.joblib").exists()
    ])


def get_csv_row_count(path):
    """Cuenta filas de un CSV (sin contar encabezado)."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return sum(1 for _ in fh) - 1  # resta el header
    except Exception:
        return -1


def get_heaviest_and_lightest_per_day():
    """
    Para cada fecha única en daily/, devuelve el CSV con más filas y el de menos filas.
    Retorna lista de dicts: [{"date": ..., "heaviest": Path, "lightest": Path}, ...]
    Excluye días con un solo archivo (se incluye como ambos) o sin archivos.
    """
    from collections import defaultdict
    day_files = defaultdict(list)

    for f in DAILY_DIR.glob("*.csv"):
        try:
            # Nombre: traffic_YYYY-MM-DD_HH-MM-SS_...csv
            date_str = f.stem.split("_")[1]
            day_files[date_str].append(f)
        except IndexError:
            pass

    result = []
    for day in sorted(day_files.keys()):
        files = day_files[day]
        if not files:
            continue
        counted = [(get_csv_row_count(f), f) for f in files]
        counted.sort(key=lambda x: x[0])
        lightest = counted[0][1]
        heaviest = counted[-1][1]
        result.append({
            "date": day,
            "lightest": lightest,
            "heaviest": heaviest,
            "lightest_rows": counted[0][0],
            "heaviest_rows": counted[-1][0],
        })
    return result


# Cola global — un solo hilo worker consume jobs uno a uno
import queue as _queue
_job_queue = _queue.Queue()
_worker_running = False


def _worker_loop():
    """Hilo unico que ejecuta los jobs de la cola secuencialmente."""
    global _worker_running
    while True:
        cmd, log_widget, on_done = _job_queue.get()
        _exec_one(cmd, log_widget, on_done)
        _job_queue.task_done()


def _exec_one(cmd, log_widget, on_done):
    """Ejecuta un proceso y bloquea hasta que termina (dentro del worker)."""
    log_widget.after(0, lambda: (
        log_widget.insert(tk.END, "\n>>> " + " ".join(str(c) for c in cmd) + "\n"),
        log_widget.see(tk.END)
    ))
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        proc = subprocess.Popen(
            [sys.executable] + [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            cwd=str(SCRIPT_DIR),
            env=env
        )
        for line in proc.stdout:
            log_widget.after(0, lambda l=line: (
                log_widget.insert(tk.END, l),
                log_widget.see(tk.END)
            ))
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"ERROR (codigo {proc.returncode})"
        log_widget.after(0, lambda s=status: (
            log_widget.insert(tk.END, f"\n[{s}]\n" + "="*60 + "\n"),
            log_widget.see(tk.END)
        ))
    except Exception as e:
        log_widget.after(0, lambda err=e: log_widget.insert(tk.END, f"\n[EXCEPCION] {err}\n"))
    if on_done:
        log_widget.after(0, on_done)


def run_subprocess(cmd, log_widget, on_done=None):
    """Encola un comando. El worker lo ejecutara cuando terminen los anteriores."""
    global _worker_running
    _job_queue.put((cmd, log_widget, on_done))
    if not _worker_running:
        _worker_running = True
        threading.Thread(target=_worker_loop, daemon=True).start()


# ===================================================================
# APLICACIÓN PRINCIPAL
# ===================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Entrenamiento y Análisis de Modelos")
        self.resizable(True, True)
        self._build_ui()

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.tab_dates   = tk.Frame(nb)
        self.tab_train   = tk.Frame(nb)
        self.tab_score   = tk.Frame(nb)

        nb.add(self.tab_dates, text="Fechas en daily/")
        nb.add(self.tab_train, text="Entrenar modelos")
        nb.add(self.tab_score, text="Analizar CSV")

        self._build_tab_dates()
        self._build_tab_train()
        self._build_tab_score()

    # ---------------------------------------------------------------
    # TAB 1: Fechas disponibles
    # ---------------------------------------------------------------

    def _build_tab_dates(self):
        f = self.tab_dates

        tk.Label(f, text="Fechas únicas detectadas en daily/*.csv:").pack(anchor="w", padx=4, pady=4)

        frame_list = tk.Frame(f)
        frame_list.pack(fill=tk.BOTH, expand=True, padx=4)

        scrollbar = tk.Scrollbar(frame_list)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox_dates = tk.Listbox(frame_list, yscrollcommand=scrollbar.set, selectmode=tk.EXTENDED)
        self.listbox_dates.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox_dates.yview)

        tk.Button(f, text="Actualizar", command=self._refresh_dates).pack(pady=4)
        self.label_count = tk.Label(f, text="")
        self.label_count.pack()

        self._refresh_dates()

    def _refresh_dates(self):
        self.listbox_dates.delete(0, tk.END)
        dates = get_dates_in_daily()
        for d in dates:
            self.listbox_dates.insert(tk.END, d)
        self.label_count.config(text=f"Total: {len(dates)} fecha(s)")
        # Actualizar también los combos de la pestaña de entrenamiento
        self._update_date_combos(dates)

    def _update_date_combos(self, dates):
        if hasattr(self, "combo_start"):
            self.combo_start["values"] = dates
            self.combo_end["values"]   = dates
        if hasattr(self, "combo_d1"):
            self.combo_d1["values"] = dates
            self.combo_d2["values"] = dates

    # ---------------------------------------------------------------
    # TAB 2: Entrenar modelos
    # ---------------------------------------------------------------

    def _build_tab_train(self):
        f = self.tab_train
        dates = get_dates_in_daily()

        # --- Modo ---
        tk.Label(f, text="Modo de entrenamiento:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.var_mode = tk.StringVar(value="normal")
        for i, mode in enumerate(["low", "normal", "hardcore"]):
            tk.Radiobutton(f, text=mode, variable=self.var_mode, value=mode).grid(
                row=0, column=i+1, padx=2, pady=2)

        # --- Sección: Global ---
        tk.Label(f, text="--- Global (todos los dias) ---").grid(
            row=1, column=0, columnspan=5, sticky="w", padx=4, pady=(8,2))
        tk.Button(f, text="Entrenar GLOBAL", command=self._train_global).grid(
            row=2, column=0, padx=4, pady=2, sticky="w")

        # --- Sección: Rango ---
        tk.Label(f, text="--- Rango de fechas ---").grid(
            row=3, column=0, columnspan=5, sticky="w", padx=4, pady=(8,2))
        tk.Label(f, text="Desde:").grid(row=4, column=0, sticky="w", padx=4)
        self.combo_start = ttk.Combobox(f, values=dates, state="readonly", width=14)
        self.combo_start.grid(row=4, column=1, padx=2)
        tk.Label(f, text="Hasta:").grid(row=4, column=2, sticky="w")
        self.combo_end = ttk.Combobox(f, values=dates, state="readonly", width=14)
        self.combo_end.grid(row=4, column=3, padx=2)
        tk.Button(f, text="Entrenar rango", command=self._train_range).grid(
            row=4, column=4, padx=4)

        # --- Sección: Un día ---
        tk.Label(f, text="--- Un solo dia ---").grid(
            row=5, column=0, columnspan=5, sticky="w", padx=4, pady=(8,2))
        tk.Label(f, text="Fecha:").grid(row=6, column=0, sticky="w", padx=4)
        self.combo_single = ttk.Combobox(f, values=dates, state="readonly", width=14)
        self.combo_single.grid(row=6, column=1, padx=2)
        tk.Button(f, text="Entrenar dia", command=self._train_single).grid(
            row=6, column=4, padx=4)

        # --- Sección: Todos contra todos ---
        tk.Label(f, text="--- Todos los dias por separado (N modelos) ---").grid(
            row=7, column=0, columnspan=5, sticky="w", padx=4, pady=(8,2))
        tk.Button(f, text="Entrenar TODOS los dias", command=self._train_all_days).grid(
            row=8, column=0, padx=4, pady=2, sticky="w")

        # --- Sección: Día A vs Día B ---
        tk.Label(f, text="--- Dia A vs Dia B ---").grid(
            row=9, column=0, columnspan=5, sticky="w", padx=4, pady=(8,2))
        tk.Label(f, text="Dia A:").grid(row=10, column=0, sticky="w", padx=4)
        self.combo_d1 = ttk.Combobox(f, values=dates, state="readonly", width=14)
        self.combo_d1.grid(row=10, column=1, padx=2)
        tk.Label(f, text="Dia B:").grid(row=10, column=2, sticky="w")
        self.combo_d2 = ttk.Combobox(f, values=dates, state="readonly", width=14)
        self.combo_d2.grid(row=10, column=3, padx=2)
        tk.Button(f, text="Entrenar A vs B", command=self._train_ab).grid(
            row=10, column=4, padx=4)

        # --- Sección: Prueba masiva ---
        tk.Label(f, text="--- PRUEBA MASIVA (global + todos los rangos, modo aleatorio) ---").grid(
            row=11, column=0, columnspan=5, sticky="w", padx=4, pady=(8,2))
        tk.Button(f, text="Lanzar prueba masiva", command=self._train_prueba_masiva).grid(
            row=12, column=0, padx=4, pady=2, sticky="w")

        # --- Log de entrenamiento ---
        tk.Label(f, text="Log:").grid(row=13, column=0, sticky="w", padx=4, pady=(8,0))
        self.log_train = scrolledtext.ScrolledText(f, height=14, wrap=tk.WORD)
        self.log_train.grid(row=14, column=0, columnspan=5, padx=4, pady=4, sticky="nsew")

        f.rowconfigure(14, weight=1)
        f.columnconfigure(4, weight=1)

    def _get_mode(self):
        return self.var_mode.get()

    def _random_mode(self):
        return random.choice(["low", "normal", "hardcore"])

    def _train_global(self):
        self._launch_train(None, None)

    def _train_range(self):
        s = self.combo_start.get()
        e = self.combo_end.get()
        if not s or not e:
            messagebox.showwarning("Faltan datos", "Selecciona fecha de inicio y fin.")
            return
        if s > e:
            messagebox.showwarning("Fechas", "La fecha de inicio debe ser <= fecha fin.")
            return
        self._launch_train(s, e)

    def _train_single(self):
        d = self.combo_single.get()
        if not d:
            messagebox.showwarning("Faltan datos", "Selecciona una fecha.")
            return
        self._launch_train(d, d)

    def _train_all_days(self):
        dates = get_dates_in_daily()
        if not dates:
            messagebox.showwarning("Sin fechas", "No hay CSV en daily/")
            return
        self.log_train.insert(tk.END, f"\nLanzando {len(dates)} entrenamientos...\n")
        for d in dates:
            self._launch_train(d, d, sequential=False)

    def _train_ab(self):
        d1 = self.combo_d1.get()
        d2 = self.combo_d2.get()
        if not d1 or not d2:
            messagebox.showwarning("Faltan datos", "Selecciona Día A y Día B.")
            return
        # Entrena el par en orden
        for start, end in [(d1, d1), (d2, d2), (d1, d2)]:
            self._launch_train(start, end, sequential=False)

    def _launch_train(self, start_date, end_date, sequential=True, override_mode=None):
        mode = override_mode if override_mode else self._get_mode()
        cmd = [TRAIN_SCRIPT, mode]
        if start_date:
            cmd.append(start_date)
        if end_date:
            cmd.append(end_date)
        run_subprocess(cmd, self.log_train, on_done=self._refresh_model_list)

    def _train_prueba_masiva(self):
        """Global + todas las combinaciones de fechas, modo aleatorio en cada uno."""
        dates = get_dates_in_daily()
        if not dates:
            messagebox.showwarning("Sin fechas", "No hay CSV en daily/")
            return

        # Construir todas las combinaciones: global + cada par (i, j) con i <= j
        jobs = []
        jobs.append((None, None))                          # global
        for i, d in enumerate(dates):
            jobs.append((d, d))                            # dia solo
        for i in range(len(dates)):
            for j in range(i + 1, len(dates)):
                jobs.append((dates[i], dates[j]))          # rango i..j

        total = len(jobs)
        self.log_train.insert(tk.END,
            f"\n[PRUEBA MASIVA] {total} entrenamientos "
            f"({len(dates)} fechas -> global + dias + todos los rangos)\n"
            f"Modo: ALEATORIO por cada entrenamiento\n{'='*60}\n"
        )
        self.log_train.see(tk.END)

        for start, end in jobs:
            mode = self._random_mode()
            label = f"global [{mode}]" if not start else f"{start} -> {end} [{mode}]"
            self.log_train.insert(tk.END, f"  Encolando: {label}\n")
            self._launch_train(start, end, sequential=False, override_mode=mode)

        self.log_train.see(tk.END)

    # ---------------------------------------------------------------
    # TAB 3: Analizar CSV contra modelos
    # ---------------------------------------------------------------

    def _build_tab_score(self):
        f = self.tab_score

        # CSV a analizar
        tk.Label(f, text="CSV a analizar:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.var_csv = tk.StringVar()
        tk.Entry(f, textvariable=self.var_csv, width=50).grid(row=0, column=1, padx=2, sticky="ew")
        tk.Button(f, text="Browse", command=self._browse_csv).grid(row=0, column=2, padx=4)

        # Lista de modelos disponibles
        tk.Label(f, text="Modelos disponibles:").grid(row=1, column=0, sticky="nw", padx=4, pady=4)
        frame_models = tk.Frame(f)
        frame_models.grid(row=1, column=1, columnspan=2, sticky="nsew", padx=2, pady=2)
        sb_m = tk.Scrollbar(frame_models)
        sb_m.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_models = tk.Listbox(frame_models, yscrollcommand=sb_m.set,
                                          selectmode=tk.EXTENDED, height=8)
        self.listbox_models.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_m.config(command=self.listbox_models.yview)

        btn_frame = tk.Frame(f)
        btn_frame.grid(row=2, column=0, columnspan=3, padx=4, pady=4, sticky="w")
        tk.Button(btn_frame, text="Actualizar modelos",
                  command=self._refresh_model_list).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Analizar contra seleccionados",
                  command=self._score_selected).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Analizar contra TODOS",
                  command=self._score_all).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Uno contra todos  (pesado + ligero x día)",
                  command=self._score_uno_contra_todos).pack(side=tk.LEFT, padx=2)

        # Resultados
        tk.Label(f, text="Resultados:").grid(row=3, column=0, sticky="nw", padx=4, pady=(8,0))
        self.log_score = scrolledtext.ScrolledText(f, height=16, wrap=tk.WORD)
        self.log_score.grid(row=4, column=0, columnspan=3, padx=4, pady=4, sticky="nsew")

        f.rowconfigure(4, weight=1)
        f.columnconfigure(1, weight=1)

        self._refresh_model_list()

    def _browse_csv(self):
        path = filedialog.askopenfilename(
            title="Seleccionar CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.var_csv.set(path)

    def _refresh_model_list(self):
        self.listbox_models.delete(0, tk.END)
        for m in get_available_models():
            self.listbox_models.insert(tk.END, m)

    def _score_selected(self):
        selected = [self.listbox_models.get(i) for i in self.listbox_models.curselection()]
        if not selected:
            messagebox.showwarning("Sin selección", "Selecciona al menos un modelo de la lista.")
            return
        self._run_score_against(selected)

    def _score_all(self):
        models = get_available_models()
        if not models:
            messagebox.showwarning("Sin modelos", "No hay modelos entrenados en models/")
            return
        self._run_score_against(models)

    def _score_uno_contra_todos(self):
        """
        Para cada día en daily/, toma el CSV con más filas y el de menos filas.
        Analiza ambos contra todos los modelos disponibles.
        """
        models = get_available_models()
        if not models:
            messagebox.showwarning("Sin modelos", "No hay modelos entrenados en models/")
            return

        dias = get_heaviest_and_lightest_per_day()
        if not dias:
            messagebox.showwarning("Sin datos", "No se encontraron CSVs en daily/")
            return

        log = self.log_score
        total_csvs = len(dias) * 2
        total_jobs = total_csvs * len(models)

        log.insert(tk.END,
            f"\n{'='*60}\n"
            f"[UNO CONTRA TODOS]\n"
            f"  Días encontrados : {len(dias)}\n"
            f"  CSVs a analizar  : {total_csvs}  (pesado + ligero por día)\n"
            f"  Modelos          : {len(models)}\n"
            f"  Total de jobs    : {total_jobs}\n"
            f"{'='*60}\n"
        )
        log.see(tk.END)

        for info in dias:
            day        = info["date"]
            heaviest   = info["heaviest"]
            lightest   = info["lightest"]
            h_rows     = info["heaviest_rows"]
            l_rows     = info["lightest_rows"]

            log.insert(tk.END,
                f"\n--- Día {day} ---\n"
                f"  PESADO  ({h_rows} filas): {heaviest.name}\n"
                f"  LIGERO  ({l_rows} filas): {lightest.name}\n"
            )
            log.see(tk.END)

            # Pesado y ligero contra todos los modelos
            for csv_path, label in [(heaviest, "PESADO"), (lightest, "LIGERO")]:
                for model_name in models:
                    parts = self._parse_model_folder(model_name)
                    cmd = [CONTROLLER_SCRIPT, str(csv_path)]
                    if parts["start"]:
                        cmd.append(parts["start"])
                    if parts["end"]:
                        cmd.append(parts["end"])

                    # Pequeño encabezado para identificar el job en el log
                    header = f"\n[{label} {day}] vs modelo [{model_name}]\n"
                    log.insert(tk.END, header)
                    log.see(tk.END)

                    run_subprocess(cmd, log)

    def _run_score_against(self, model_names):
        csv_path = self.var_csv.get().strip()
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showwarning("CSV inválido", "Selecciona un CSV válido primero.")
            return

        self.log_score.insert(tk.END, f"\n{'='*60}\nAnalizando: {csv_path}\n")

        for model_name in model_names:
            parts = self._parse_model_folder(model_name)
            cmd = [CONTROLLER_SCRIPT, csv_path]
            if parts["start"]:
                cmd.append(parts["start"])
            if parts["end"]:
                cmd.append(parts["end"])
            self.log_score.insert(tk.END, f"\n--- Modelo: {model_name} ---\n")
            run_subprocess(cmd, self.log_score)

    def _parse_model_folder(self, folder_name):
        """Intenta extraer start y end del nombre de carpeta del modelo."""
        result = {"start": None, "end": None}
        if folder_name == "global":
            return result
        if "_to_latest" in folder_name:
            result["start"] = folder_name.replace("_to_latest", "")
        elif "from_earliest_" in folder_name:
            result["end"] = folder_name.replace("from_earliest_", "")
        else:
            # Asumimos formato YYYY-MM-DD_YYYY-MM-DD
            parts = folder_name.split("_")
            if len(parts) == 2:
                result["start"] = parts[0]
                result["end"]   = parts[1]
        return result


# ===================================================================
# Punto de entrada
# ===================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()
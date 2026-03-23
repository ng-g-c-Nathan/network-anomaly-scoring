# VPN Anomaly Detection — Python Pipeline

Machine learning pipeline for detecting anomalous behavior in IoT/VPN network traffic using Isolation Forest and K-Means.

---

## What does this project do?

Processes daily Wireshark CSV captures, trains unsupervised models, and produces an **anomaly score** per session consumed by the Spring Boot backend via JSON stdout.

## Pipeline Architecture

```
daily/*.csv  ──►  preprocess.py  ──►  train_models.py  ──►  models/
                                                                 │
                  score_csv.py  ◄────────────────────────────────┘
                       │
                       ▼
               Controller.py  ──►  analysis_history.json  ──►  Spring Boot
```

## File Structure

```
.
├── daily/                      # Wireshark CSV exports
│   └── traffic_2026-02-10_03-20-01_(10.0_minutes)_(0.06_input)_(0.05_output).csv
├── models/                     # Trained models organized by date range
│   └── 2024-01-01_2024-01-31/
│       ├── kmeans.joblib
│       ├── isoforest.joblib
│       ├── preprocessor.joblib
│       ├── mode.txt
│       └── model_info.json
├── preprocess.py               # Feature engineering and scikit-learn pipeline
├── train_models.py             # KMeans + Isolation Forest training
├── score_csv.py                # Scoring and inference on new data
├── Controller.py               # Main orchestrator, history management
└── analysis_history.json       # Persistent log of all executions
```

## CSV Filename Format

Capture files exported from Wireshark follow this naming convention:

```
traffic_YYYY-MM-DD_HH-MM-SS_(duration_minutes)_(input_ratio)_(output_ratio).csv
```

Example:
```
traffic_2026-02-10_03-20-01_(10.0_minutes)_(0.06_input)_(0.05_output).csv
```

The pipeline uses the date portion (`YYYY-MM-DD`) to filter files by date range when training or scoring.

## Installation

```bash
git clone https://github.com/nathanvargas/anomaly-scoring.git
cd anomaly-scoring

pip install pandas numpy scikit-learn joblib
```

## Usage

### 1. Train models

```bash
# Global training (all CSVs in daily/)
python train_models.py

# With date range
python train_models.py normal 2024-01-01 2024-01-31

# Available modes: low | normal | hardcore
python train_models.py hardcore 2024-01-01 2024-01-31
```

| Mode | KMeans n_init | IsoForest estimators | Speed |
|------|:---:|:---:|:---:|
| `low` | 5 | 50 | Fast |
| `normal` | 10 | 200 | Balanced |
| `hardcore` | 20 | 500 | Max precision |

### 2. Analyze a capture

```bash
# Analyze a CSV against the global model
python Controller.py "traffic_2026-02-10_03-20-01_(10.0_minutes)_(0.06_input)_(0.05_output).csv"

# Analyze using a specific trained model
python Controller.py traffic.csv 2024-01-01 2024-01-31

# Analyze a range of daily CSVs
python Controller.py 2024-02-01 2024-02-07
```

### 3. Output (stdout JSON)

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "FINISHED",
  "result": {
    "rows": 15234,
    "iso_score_mean": 0.082,
    "iso_score_min": -0.341,
    "anomaly_ratio_iso": 0.034,
    "kmeans_distance_mean": 1.27,
    "kmeans_distance_max": 8.93,
    "mode": "normal",
    "trained_at": "2024-01-31T22:00:00"
  },
  "duration_seconds": 4.2,
  "duration_human": "4.2 seconds",
  "duration_category": "SHORT"
}
```

## Scoring Metrics

| Metric | Description |
|--------|-------------|
| `iso_score_mean` | Mean Isolation Forest score (more negative = more anomalous) |
| `iso_score_min` | Worst individual score detected |
| `anomaly_ratio_iso` | Percentage of packets classified as anomalies |
| `kmeans_distance_mean` | Mean distance to the assigned cluster centroid |
| `kmeans_distance_max` | Maximum distance — potential extreme outliers |

## Model Features

The preprocessor extracts and normalizes the following columns from the Wireshark CSV export:

**Categorical** (OrdinalEncoder): `frame_protocols`, `ip_src`, `ip_dst`, `dns_qry_name`, `tls_handshake_extensions_server_name`, `http_request_method`, `http_request_uri`, `tcp_flags`

**Numerical** (StandardScaler): `frame_number`, `frame_len`, `tcp_srcport`, `tcp_dstport`, `udp_srcport`, `udp_dstport`, `http_response_code`, `iat` *(inter-arrival time, computed automatically from timestamps)*

## Execution History

Every analysis run is recorded in `analysis_history.json`:

```json
[
  {
    "job_id": "a1b2c3d4-...",
    "csv_file": "traffic_2026-02-10_03-20-01_(10.0_minutes)_(0.06_input)_(0.05_output).csv",
    "created_at": "2026-02-10T10:00:00+00:00",
    "started_at": "2026-02-10T10:00:01+00:00",
    "finished_at": "2026-02-10T10:00:05+00:00",
    "status": "FINISHED",
    "duration_human": "4.2 seconds",
    "duration_category": "SHORT",
    "result": {}
  }
]
```

## Related Projects

- [Angular Frontend](https://github.com/nathanvargas/angular-vpn-interface) — Real-time network traffic visualization
- [Spring Boot Backend](https://github.com/nathanvargas/springboot-vpn-backend) — VPN configuration management and scoring consumer

---

Built with Python, scikit-learn, pandas, numpy. Part of the VPN Anomaly Detection project.

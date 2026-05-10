# preprocess.py
"""
preprocess.py
-------------
Módulo de ingeniería de características y preprocesamiento para tráfico de red.

Define el pipeline completo de transformación de datos crudos de Wireshark/tshark
a una matriz numérica lista para ser consumida por los modelos de clustering y
detección de anomalías. Incluye:

- Expansión de TCP flags a bits individuales.
- Clasificación de puertos por rango semántico (well_known / registered / ephemeral).
- Encoding cíclico de hora y día de la semana (sin/cos).
- Features derivadas de IP, DNS y HTTP URI.
- Pipeline scikit-learn con imputación, RobustScaler y passthrough de binarios.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from sklearn.impute import SimpleImputer


# -------------------------------------------------------------------
# Definición de columnas del dataset
# -------------------------------------------------------------------

# Columnas categóricas de baja cardinalidad — seguras para OrdinalEncoder
CATEGORICAL_COLS = [
    "protocol_type",        # "tcp" | "udp" — solo 2 valores
    "port_cat_src",         # "well_known" | "registered" | "ephemeral"
    "port_cat_dst",
]

# Columnas numéricas que van al pipeline estándar (RobustScaler)
NUMERIC_COLS = [
    "frame_len",
    "http_response_code",
    "log_iat",              # log1p(iat) — distribución log-normal
    # Encoding cíclico de tiempo
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Flags TCP expandidos como bits individuales
    "flag_syn",
    "flag_ack",
    "flag_fin",
    "flag_rst",
    "flag_psh",
    "flag_urg",
    # Features derivadas de IP
    "is_loopback_src",
    "is_loopback_dst",
    "is_private_src",
    "is_private_dst",
    "is_same_host",
    # Features derivadas de DNS
    "dns_name_len",
    "dns_subdomain_count",
    # Features derivadas de HTTP URI
    "uri_len",
    "uri_depth",
    # Indicadores de presencia (binarios 0/1 — passthrough, sin escalar)
    "has_frame_len",
    "has_http_response_code",
    "has_frame_protocols",
    "has_dns_qry_name",
    "has_tls_sni",
    "has_http_request_method",
    "has_http_request_uri",
]

# Columnas binarias has_* que deben ir sin escalar (passthrough)
PASSTHROUGH_COLS = [
    "has_frame_len",
    "has_http_response_code",
    "has_frame_protocols",
    "has_dns_qry_name",
    "has_tls_sni",
    "has_http_request_method",
    "has_http_request_uri",
    "is_loopback_src",
    "is_loopback_dst",
    "is_private_src",
    "is_private_dst",
    "is_same_host",
    "flag_syn",
    "flag_ack",
    "flag_fin",
    "flag_rst",
    "flag_psh",
    "flag_urg",
]

# Columnas que sí se escalan con RobustScaler
ROBUST_COLS = [
    "frame_len",
    "http_response_code",
    "log_iat",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "dns_name_len",
    "dns_subdomain_count",
    "uri_len",
    "uri_depth",
]


# -------------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------------

def _classify_port(port):
    """
    Clasifica un número de puerto en su categoría de rango.

    Categorías:
        well_known  : 0    – 1023
        registered  : 1024 – 49151
        ephemeral   : 49152 – 65535
        NA          : null / NaN
    """
    if pd.isna(port):
        return "NA"
    port = int(port)
    if port <= 1023:
        return "well_known"
    if port <= 49151:
        return "registered"
    return "ephemeral"


def _parse_tcp_flags_to_bits(flag_hex):
    """
    Convierte un valor hexadecimal de TCP flags en sus bits individuales.

    Flags TCP estándar (RFC 793 / RFC 3168):
        bit 0 = FIN
        bit 1 = SYN
        bit 2 = RST
        bit 3 = PSH
        bit 4 = ACK
        bit 5 = URG

    Parameters
    ----------
    flag_hex : str | None
        Valor hexadecimal como "0x00000002" o None / "".

    Returns
    -------
    dict con claves: flag_syn, flag_ack, flag_fin, flag_rst, flag_psh, flag_urg
    """
    zero = {"flag_syn": 0, "flag_ack": 0, "flag_fin": 0,
            "flag_rst": 0, "flag_psh": 0, "flag_urg": 0}

    if pd.isna(flag_hex) or flag_hex == "":
        return zero

    try:
        val = int(flag_hex, 16) if isinstance(flag_hex, str) else int(flag_hex)
    except (ValueError, TypeError):
        return zero

    return {
        "flag_fin": int(bool(val & (1 << 0))),
        "flag_syn": int(bool(val & (1 << 1))),
        "flag_rst": int(bool(val & (1 << 2))),
        "flag_psh": int(bool(val & (1 << 3))),
        "flag_ack": int(bool(val & (1 << 4))),
        "flag_urg": int(bool(val & (1 << 5))),
    }


def _is_private_ip(ip_str):
    """
    Determina si una IP es de rango privado o de loopback.

    Rangos privados (RFC 1918):
        10.0.0.0/8
        172.16.0.0/12
        192.168.0.0/16
        127.0.0.0/8  (loopback)
        169.254.0.0/16 (link-local)

    Returns
    -------
    tuple (is_loopback: int, is_private: int)
    """
    if pd.isna(ip_str) or ip_str == "":
        return 0, 0

    try:
        parts = [int(p) for p in str(ip_str).split(".")]
        if len(parts) != 4:
            return 0, 0
        a, b = parts[0], parts[1]
    except (ValueError, AttributeError):
        return 0, 0

    is_loopback = int(a == 127)
    is_private = int(
        a == 10 or
        (a == 172 and 16 <= b <= 31) or
        (a == 192 and b == 168) or
        a == 127 or
        (a == 169 and b == 254)
    )
    return is_loopback, is_private


def _cyclic_encode(value, max_val):
    """
    Aplica encoding cíclico (sin/cos) para preservar la distancia circular.

    Ejemplo: hour=23 y hour=0 son "cercanas" — el encoding cíclico
    captura eso, StandardScaler no.

    Parameters
    ----------
    value : float | int
    max_val : int — número total de valores posibles (24 para horas, 7 para días)

    Returns
    -------
    tuple (sin_val: float, cos_val: float)
    """
    angle = 2 * np.pi * value / max_val
    return np.sin(angle), np.cos(angle)


def _dns_features(name):
    """
    Extrae features numéricas del nombre de consulta DNS.

    Features:
        dns_name_len        : longitud total del nombre
        dns_subdomain_count : número de componentes separados por "."

    Parameters
    ----------
    name : str | None

    Returns
    -------
    tuple (dns_name_len: int, dns_subdomain_count: int)
    """
    if pd.isna(name) or name == "":
        return 0, 0
    name = str(name).strip(".")
    return len(name), len(name.split("."))


def _uri_features(uri):
    """
    Extrae features numéricas de un HTTP URI.

    Features:
        uri_len   : longitud total del URI
        uri_depth : número de segmentos de path (slashes)

    Los parámetros de query (?key=value) se excluyen del cálculo
    de profundidad — solo se considera el path.

    Parameters
    ----------
    uri : str | None

    Returns
    -------
    tuple (uri_len: int, uri_depth: int)
    """
    if pd.isna(uri) or uri == "":
        return 0, 0
    uri = str(uri)
    path = uri.split("?")[0]
    depth = len([s for s in path.split("/") if s])
    return len(uri), depth


# -------------------------------------------------------------------
# Ingeniería de características
# -------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye y normaliza las características de entrada a partir
    del DataFrame original.

    Mejoras respecto a la versión anterior:

    1. TCP flags expandidos a bits individuales (SYN, ACK, FIN, RST, PSH, URG).
       OrdinalEncoder sobre flags hex era semánticamente incorrecto.

    2. IPs procesadas como features derivadas (loopback, privada, mismo host)
       en lugar de encodear el valor literal. Evita el colapso de OrdinalEncoder
       ante IPs nunca vistas en producción.

    3. log1p(iat) antes del scaler. El inter-arrival time tiene distribución
       log-normal — StandardScaler directo aplasta la variación útil.

    4. Encoding cíclico para hour y dow (sin/cos).
       La hora 23 y la hora 0 son contiguas; el encoding ordinal las separa al máximo.

    5. Clasificación de puertos por rango semántico en lugar de valor numérico raw.

    6. Features derivadas de DNS y URI (longitud, profundidad, subdominios).

    7. Columnas has_* marcadas como passthrough — no escalar binarios 0/1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original leído desde los CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame con todas las columnas necesarias para el preprocesador.
    """

    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Timestamp y ordenación
    # ------------------------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Inter-arrival time + transformación log
    # ------------------------------------------------------------------
    df["iat"] = df["timestamp"].diff().dt.total_seconds().fillna(0).clip(lower=0)
    df["log_iat"] = np.log1p(df["iat"])

    # ------------------------------------------------------------------
    # 3. Features temporales con encoding cíclico
    # ------------------------------------------------------------------
    df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["dow"]  = df["timestamp"].dt.dayofweek.fillna(0).astype(int)

    df["hour_sin"], df["hour_cos"] = zip(*df["hour"].apply(lambda h: _cyclic_encode(h, 24)))
    df["dow_sin"],  df["dow_cos"]  = zip(*df["dow"].apply(lambda d: _cyclic_encode(d, 7)))

    # ------------------------------------------------------------------
    # 4. Protocol type inferido desde puertos
    # ------------------------------------------------------------------
    df["protocol_type"] = "udp"
    df.loc[df["tcp_srcport"].notna(), "protocol_type"] = "tcp"

    # ------------------------------------------------------------------
    # 5. Clasificación de puertos por rango semántico
    # ------------------------------------------------------------------
    src_port = df["tcp_srcport"].fillna(df["udp_srcport"])
    dst_port = df["tcp_dstport"].fillna(df["udp_dstport"])

    df["port_cat_src"] = src_port.apply(_classify_port)
    df["port_cat_dst"] = dst_port.apply(_classify_port)

    # ------------------------------------------------------------------
    # 6. TCP flags → bits individuales
    #    OrdinalEncoder sobre 0x00000002 era semánticamente incorrecto.
    # ------------------------------------------------------------------
    flag_bits = df["tcp_flags"].apply(_parse_tcp_flags_to_bits).apply(pd.Series)
    df = pd.concat([df, flag_bits], axis=1)

    # ------------------------------------------------------------------
    # 7. Features derivadas de IP
    #    No se encodea el valor literal — demasiada cardinalidad + valores
    #    nunca vistos en producción quiebran OrdinalEncoder.
    # ------------------------------------------------------------------
    ip_src_feats = df["ip_src"].apply(_is_private_ip).apply(pd.Series)
    ip_src_feats.columns = ["is_loopback_src", "is_private_src"]

    ip_dst_feats = df["ip_dst"].apply(_is_private_ip).apply(pd.Series)
    ip_dst_feats.columns = ["is_loopback_dst", "is_private_dst"]

    df = pd.concat([df, ip_src_feats, ip_dst_feats], axis=1)
    df["is_same_host"] = (df["ip_src"] == df["ip_dst"]).astype(int)

    # ------------------------------------------------------------------
    # 8. Features derivadas de DNS
    # ------------------------------------------------------------------
    df["dns_name_len"], df["dns_subdomain_count"] = zip(
        *df.get("dns_qry_name", pd.Series([""] * len(df))).apply(_dns_features)
    )

    # ------------------------------------------------------------------
    # 9. Features derivadas de HTTP URI
    # ------------------------------------------------------------------
    df["uri_len"], df["uri_depth"] = zip(
        *df.get("http_request_uri", pd.Series([""] * len(df))).apply(_uri_features)
    )

    # ------------------------------------------------------------------
    # 10. Indicadores de presencia (has_*)
    #     Incluye has_tls_sni en lugar del nombre largo anterior.
    # ------------------------------------------------------------------
    presence_map = {
        "has_frame_len":            "frame_len",
        "has_http_response_code":   "http_response_code",
        "has_frame_protocols":      "frame_protocols",
        "has_dns_qry_name":         "dns_qry_name",
        "has_tls_sni":              "tls_handshake_extensions_server_name",
        "has_http_request_method":  "http_request_method",
        "has_http_request_uri":     "http_request_uri",
    }
    for feat, col in presence_map.items():
        df[feat] = df[col].notna().astype(int) if col in df.columns else 0

    # ------------------------------------------------------------------
    # 11. Garantizar existencia de todas las columnas del pipeline
    # ------------------------------------------------------------------
    all_pipeline_cols = CATEGORICAL_COLS + ROBUST_COLS + PASSTHROUGH_COLS
    for c in all_pipeline_cols:
        if c not in df.columns:
            df[c] = 0

    # ------------------------------------------------------------------
    # 12. Conversión explícita de columnas numéricas
    # ------------------------------------------------------------------
    for c in ROBUST_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ------------------------------------------------------------------
    # Advertencia de columnas con varianza cero
    # ------------------------------------------------------------------
    zero_var = [c for c in ROBUST_COLS if df[c].std() == 0]
    if zero_var:
        print(f"[WARN] Columnas numéricas con varianza cero: {zero_var}")

    return df


# -------------------------------------------------------------------
# Construcción del preprocesador de scikit-learn
# -------------------------------------------------------------------

def build_preprocessor():
    """
    Construye el pipeline de preprocesamiento completo.

    Pipelines:

    1. Categórico (CATEGORICAL_COLS):
       - protocol_type, port_cat_src, port_cat_dst
       - Imputación con "__NA__" + OrdinalEncoder con unknown=-1
       - Solo columnas de baja cardinalidad y valores acotados

    2. Numérico robusto (ROBUST_COLS):
       - Imputación con mediana + RobustScaler
       - RobustScaler usa IQR en lugar de std — resistente a outliers
       - Incluye log_iat, hour_sin/cos, dow_sin/cos, frame_len, etc.

    3. Passthrough (PASSTHROUGH_COLS):
       - Columnas binarias 0/1 (has_*, flags, is_*)
       - Sin transformación — ya están en escala correcta
       - StandardScaler / RobustScaler sobre 0/1 no aportan nada útil

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocesador listo para ser entrenado (fit).
    """

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__NA__")),
        ("enc", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    robust_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("cat",         categorical_pipeline, CATEGORICAL_COLS),
            ("num_robust",  robust_pipeline,       ROBUST_COLS),
            ("passthrough", "passthrough",         PASSTHROUGH_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return pre


# -------------------------------------------------------------------
# Entrenamiento y persistencia del preprocesador
# -------------------------------------------------------------------

def fit_and_save_preprocessor(df_all, out_file="preprocessor.joblib"):
    """
    Ajusta (fit) el preprocesador sobre el conjunto de datos completo
    y lo guarda en disco.

    Parameters
    ----------
    df_all : pd.DataFrame
        Datos de entrenamiento completos.
    out_file : str
        Ruta de salida del archivo joblib.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocesador entrenado.
    """
    df_all = build_features(df_all)

    pre = build_preprocessor()
    pre.fit(df_all)

    joblib.dump(pre, out_file)
    print(f"[INFO] Preprocesador guardado en: {out_file}")

    return pre


# -------------------------------------------------------------------
# Carga de preprocesador
# -------------------------------------------------------------------

def load_preprocessor(path="preprocessor.joblib"):
    """
    Carga un preprocesador previamente entrenado desde disco.

    Parameters
    ----------
    path : str

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    return joblib.load(path)


# -------------------------------------------------------------------
# Transformación de datos
# -------------------------------------------------------------------

def transform(df: pd.DataFrame, preprocessor) -> np.ndarray:
    """
    Aplica el mismo pipeline de ingeniería y transformación que en
    entrenamiento.

    Parameters
    ----------
    df : pd.DataFrame
    preprocessor : sklearn.compose.ColumnTransformer
        Preprocesador ya entrenado.

    Returns
    -------
    numpy.ndarray
        Matriz de características lista para los modelos.
    """
    df = build_features(df)
    return preprocessor.transform(df)


# -------------------------------------------------------------------
# Utilidad: obtener nombres de features del preprocesador entrenado
# -------------------------------------------------------------------

def get_feature_names(preprocessor) -> list:
    """
    Retorna los nombres de features en el orden que produce el
    ColumnTransformer, útil para interpretar importancias de modelos.

    Parameters
    ----------
    preprocessor : sklearn.compose.ColumnTransformer (ya fiteado)

    Returns
    -------
    list of str
    """
    return list(preprocessor.get_feature_names_out())
# preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.impute import SimpleImputer


# -------------------------------------------------------------------
# Definición de columnas del dataset
# -------------------------------------------------------------------

# Columnas categóricas (alto cardinal y/o valores tipo string)
CATEGORICAL_COLS = [
    "frame_protocols",
    "ip_src",
    "ip_dst",
    "dns_qry_name",
    "tls_handshake_extensions_server_name",
    "http_request_method",
    "http_request_uri",
    "tcp_flags",
]

# Columnas numéricas utilizadas por el modelo
NUMERIC_COLS = [
    "frame_number",
    "frame_len",
    "tcp_srcport",
    "tcp_dstport",
    "udp_srcport",
    "udp_dstport",
    "http_response_code",
    "iat",
]


# -------------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------------

def _parse_tcp_flags(x):
    """
    Normaliza el campo tcp_flags.

    - Convierte valores nulos o vacíos en el literal "NA".
    - Asegura que el valor final sea siempre string.

    Esta función existe para estabilizar la codificación categórica
    y evitar errores en el encoder cuando aparecen valores NaN.
    """
    if pd.isna(x) or x == "":
        return "NA"
    return str(x)


# -------------------------------------------------------------------
# Ingeniería de características
# -------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye y normaliza las características de entrada a partir
    del DataFrame original.

    Operaciones que se aplican:

    1. Conversión y ordenación por timestamp.
    2. Cálculo del inter-arrival time (iat).
    3. Normalización de tcp_flags.
    4. Creación de columnas numéricas inexistentes con valor por defecto.
    5. Conversión explícita de columnas numéricas a tipo numérico.

    Este método debe usarse de forma idéntica tanto en entrenamiento
    como en inferencia para garantizar consistencia.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original leído desde los CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas necesarias y características derivadas.
    """

    df = df.copy()

    # Conversión robusta del timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Orden temporal para poder calcular correctamente el IAT
    df = df.sort_values("timestamp")

    # Inter-arrival time entre paquetes
    df["iat"] = df["timestamp"].diff().dt.total_seconds()
    df["iat"] = df["iat"].fillna(0)

    # Normalización de flags TCP
    df["tcp_flags"] = df["tcp_flags"].apply(_parse_tcp_flags)

    # Garantizar que todas las columnas numéricas existen
    for c in NUMERIC_COLS:
        if c not in df:
            df[c] = 0

    # Asegurar tipo string y valores no nulos para categorías
    df["tcp_flags"] = df["tcp_flags"].fillna("NA").astype(str)

    # Conversión explícita de columnas numéricas
    for c in [
        "frame_number",
        "frame_len",
        "tcp_srcport",
        "tcp_dstport",
        "udp_srcport",
        "udp_dstport",
        "http_response_code",
        "iat",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -------------------------------------------------------------------
# Construcción del preprocesador de scikit-learn
# -------------------------------------------------------------------

def build_preprocessor():
    """
    Construye el pipeline de preprocesamiento completo basado en
    ColumnTransformer.

    Pipeline categórico:
        - Imputación de valores faltantes con "__NA__".
        - Codificación ordinal (OrdinalEncoder).
        - Valores desconocidos se codifican como -1.

    Pipeline numérico:
        - Imputación con 0.
        - Estandarización (StandardScaler).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocesador listo para ser entrenado (fit).
    """

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__NA__")),
            (
                "enc",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
            ),
        ]
    )

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", categorical, CATEGORICAL_COLS),
            ("num", numeric, NUMERIC_COLS),
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
    Ajusta (fit) el preprocesador sobre un conjunto completo de datos
    y lo guarda en disco.

    Este método:
        1. Aplica build_features().
        2. Construye el pipeline.
        3. Ajusta el preprocesador.
        4. Guarda el objeto resultante en un archivo joblib.

    Parameters
    ----------
    df_all : pd.DataFrame
        Conjunto completo de datos de entrenamiento.
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
        Ruta del archivo joblib.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocesador cargado.
    """
    return joblib.load(path)


# -------------------------------------------------------------------
# Transformación de datos
# -------------------------------------------------------------------

def transform(df, preprocessor):
    """
    Aplica exactamente el mismo pipeline de ingeniería de características
    y transformación que se utilizó en el entrenamiento.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada (entrenamiento o inferencia).
    preprocessor : sklearn.compose.ColumnTransformer
        Preprocesador ya entrenado.

    Returns
    -------
    numpy.ndarray
        Matriz de características lista para los modelos.
    """

    df = build_features(df)
    X = preprocessor.transform(df)

    return X

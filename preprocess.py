import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.impute import SimpleImputer


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


def _parse_tcp_flags(x):
    if pd.isna(x) or x == "":
        return "NA"
    return str(x)


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.sort_values("timestamp")

    df["iat"] = df["timestamp"].diff().dt.total_seconds()
    df["iat"] = df["iat"].fillna(0)

    df["tcp_flags"] = df["tcp_flags"].apply(_parse_tcp_flags)

    for c in NUMERIC_COLS:
        if c not in df:
            df[c] = 0

    df["tcp_flags"] = df["tcp_flags"].fillna("NA").astype(str)

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


def build_preprocessor():

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__NA__")),
            (
                "enc",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
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


def fit_and_save_preprocessor(df_all, out_file="preprocessor.joblib"):

    df_all = build_features(df_all)

    pre = build_preprocessor()
    pre.fit(df_all)

    joblib.dump(pre, out_file)

    return pre


def load_preprocessor(path="preprocessor.joblib"):
    return joblib.load(path)


def transform(df, preprocessor):

    df = build_features(df)
    X = preprocessor.transform(df)

    return X

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .base import Dataset

ID_LIKE_UNIQUE_RATIO = 0.95


def _looks_like_sequential_id(series: pd.Series) -> bool:
    """True only for numeric columns whose values are literally a set of
    distinct integers (e.g. patient_id 1..n) - never for continuous
    measurements, which are legitimately near-unique per row but must not
    be dropped as if they were identifiers."""
    values = series.dropna().to_numpy()
    if len(values) == 0:
        return False
    if not np.all(values == np.floor(values)):
        return False
    return len(np.unique(values)) == len(values)


def detect_id_like_columns(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    """Flag columns that are almost certainly identifiers, so they don't get
    one-hot encoded into thousands of useless features or leak row identity
    into the model. Continuous numeric features (naturally high-cardinality)
    are never flagged - only near-unique text/categorical columns (names,
    free text, string IDs) and literal integer-sequence ID columns are."""
    id_like = []
    for col in df.columns:
        if col in exclude:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            if _looks_like_sequential_id(series):
                id_like.append(col)
        elif series.nunique(dropna=True) >= ID_LIKE_UNIQUE_RATIO * len(df):
            id_like.append(col)
    return id_like


def build_dataset_from_dataframe(
    df: pd.DataFrame,
    target_col: str,
    positive_value,
    drop_cols: list[str] | None = None,
    dataset_name: str = "custom",
) -> tuple[Dataset, dict]:
    """Turn an arbitrary uploaded CSV into a Dataset: impute missing values,
    one-hot encode categoricals, drop identifier-like columns, and binarize
    the target around the user-chosen positive value.

    Returns (dataset, report). `report` documents everything that was
    auto-dropped, imputed, or encoded - shown to the user in the dashboard
    so a non-ML person can see what happened to their data, not just trust it.
    """
    df = df.copy()
    drop_cols = list(drop_cols or [])

    auto_id_like = detect_id_like_columns(df, exclude=[target_col] + drop_cols)
    all_drop = set(drop_cols) | set(auto_id_like) | {target_col}
    feature_df = df.drop(columns=[c for c in all_drop if c in df.columns])

    if feature_df.shape[1] == 0:
        raise ValueError("No usable feature columns remain after dropping the target and ID-like columns.")

    y_raw = df[target_col]
    y = (y_raw.astype(str) == str(positive_value)).astype(int).to_numpy()

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))

    ct = ColumnTransformer(transformers)
    X = ct.fit_transform(feature_df)

    feature_names = list(numeric_cols)
    if categorical_cols:
        ohe = ct.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(ohe.get_feature_names_out(categorical_cols).tolist())

    report = {
        "dropped_id_like": auto_id_like,
        "dropped_requested": drop_cols,
        "n_missing_values_imputed": int(df[numeric_cols + categorical_cols].isna().sum().sum()),
        "n_features_after_encoding": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "class_counts": {"positive": int(y.sum()), "negative": int(len(y) - y.sum())},
    }

    dataset = Dataset(
        name=dataset_name,
        X=np.asarray(X, dtype=float),
        y=y,
        feature_names=feature_names,
        positive_label=str(positive_value),
        negative_label="other",
    )
    return dataset, report

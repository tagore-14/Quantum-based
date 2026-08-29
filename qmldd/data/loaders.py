import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer

from .base import Dataset, DataLoader


def _binarize_exact(labels, positive_value: str) -> np.ndarray:
    """Map a categorical target to 0/1 by exact match against positive_value."""
    return (np.asarray(labels).astype(str) == positive_value).astype(int)


class BreastCancerLoader(DataLoader):
    """UCI Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn."""

    name = "breast_cancer"

    def load(self) -> Dataset:
        data = load_breast_cancer()
        # sklearn convention is 0=malignant, 1=benign; flip so 1=disease-positive.
        y = 1 - data.target
        return Dataset(
            name=self.name,
            X=data.data.astype(float),
            y=y,
            feature_names=list(data.feature_names),
            positive_label="malignant",
            negative_label="benign",
        )


class HeartDiseaseLoader(DataLoader):
    """Statlog Heart dataset (Cleveland-derived), fetched from OpenML."""

    name = "heart_disease"

    def load(self) -> Dataset:
        frame = fetch_openml(name="heart-statlog", version=1, as_frame=True)
        df = frame.frame.copy()
        target_col = frame.target_names[0]
        y = _binarize_exact(df[target_col], positive_value="present")
        X = df.drop(columns=[target_col]).astype(float)
        return Dataset(
            name=self.name,
            X=X.to_numpy(),
            y=y,
            feature_names=list(X.columns),
            positive_label="heart_disease_present",
            negative_label="heart_disease_absent",
        )


class DiabetesLoader(DataLoader):
    """Pima Indians Diabetes dataset, fetched from OpenML."""

    name = "diabetes"

    def load(self) -> Dataset:
        frame = fetch_openml(name="diabetes", version=1, as_frame=True)
        df = frame.frame.copy()
        target_col = "class"
        y = _binarize_exact(df[target_col], positive_value="tested_positive")
        X = df.drop(columns=[target_col]).astype(float)
        return Dataset(
            name=self.name,
            X=X.to_numpy(),
            y=y,
            feature_names=list(X.columns),
            positive_label="diabetic",
            negative_label="non_diabetic",
        )

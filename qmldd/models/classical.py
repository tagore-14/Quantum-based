import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .base import BaseModel


class SklearnModel(BaseModel):
    is_quantum = False

    def __init__(self, name: str, estimator):
        self.name = name
        self.estimator = estimator

    def fit(self, X_train, y_train):
        self.estimator.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)[:, 1]
        return super().predict_proba(X)

    def metadata(self) -> dict:
        n_params = None
        if hasattr(self.estimator, "coef_"):
            n_params = int(np.size(self.estimator.coef_)) + int(
                np.size(getattr(self.estimator, "intercept_", 0))
            )
        elif hasattr(self.estimator, "coefs_"):  # MLPClassifier
            n_params = sum(w.size for w in self.estimator.coefs_) + sum(
                b.size for b in self.estimator.intercepts_
            )
        return {"n_params": n_params}


def logistic_regression(**kwargs) -> SklearnModel:
    return SklearnModel("classical_logreg", LogisticRegression(max_iter=1000, **kwargs))


def svm_rbf(**kwargs) -> SklearnModel:
    base = SVC(kernel="rbf", **kwargs)
    return SklearnModel("classical_svm", CalibratedClassifierCV(base, ensemble=False))


def random_forest(**kwargs) -> SklearnModel:
    return SklearnModel(
        "classical_rf", RandomForestClassifier(n_estimators=200, random_state=42, **kwargs)
    )


def small_nn(**kwargs) -> SklearnModel:
    return SklearnModel(
        "classical_nn",
        MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42, **kwargs),
    )

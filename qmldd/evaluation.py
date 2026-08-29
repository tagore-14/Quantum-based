import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalResult:
    model_name: str
    is_quantum: bool
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    roc_auc: float | None
    training_time_seconds: float
    inference_time_seconds: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "is_quantum": self.is_quantum,
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "training_time_seconds": self.training_time_seconds,
            "inference_time_seconds": self.inference_time_seconds,
            "metadata": self.metadata,
        }


def evaluate_model(model, X_train, y_train, X_test, y_test) -> EvalResult:
    """Fit a model, run it on the held-out test set, and compute the metrics
    the objectives call for: accuracy, sensitivity, specificity, plus
    supporting metrics and timing for the efficiency comparison.

    Convention: y == 1 is the disease-positive class, so sensitivity (recall
    on class 1) is the clinically meaningful "catch the sick patients" rate
    and specificity is the "don't false-alarm the healthy patients" rate.
    """
    start = time.time()
    model.fit(X_train, y_train)
    wall_training_time = time.time() - start
    training_time = model.metadata().get("training_time_seconds") or wall_training_time

    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start

    try:
        y_proba = model.predict_proba(X_test)
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc_auc = None

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return EvalResult(
        model_name=model.name,
        is_quantum=getattr(model, "is_quantum", False),
        accuracy=float(accuracy_score(y_test, y_pred)),
        sensitivity=float(recall_score(y_test, y_pred, zero_division=0)),
        specificity=float(specificity),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        training_time_seconds=float(training_time),
        inference_time_seconds=float(inference_time),
        metadata=model.metadata(),
    )

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Interface every model plugin (classical or quantum) must implement.

    Adding a new model to the platform means implementing this class and
    registering a factory in qmldd.models.MODEL_REGISTRY - the benchmark
    harness and dashboard work with any model that satisfies this interface.
    """

    name: str
    is_quantum: bool = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseModel": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1). Default falls back to hard 0/1 predictions."""
        return self.predict(X).astype(float)

    def metadata(self) -> dict:
        """Extra info surfaced in the benchmark report (param count, circuit depth, etc.)."""
        return {}

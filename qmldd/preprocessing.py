import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class QuantumReadyPreprocessor:
    """Shared preprocessing for every model in a benchmark run.

    Standardizes then reduces to n_components via PCA. Every model (classical
    and quantum) in a benchmark sees the exact same reduced feature set, so
    the comparison isolates the model rather than the feature engineering.
    n_components is also the qubit budget for quantum models, since NISQ-scale
    simulators/hardware can't handle full-dimensional biomedical feature
    vectors directly.

    fit()/fit_transform() accept (and ignore) y_train and feature_names so
    this shares a call signature with QAOAFeatureSelector, letting the
    pipeline treat both interchangeably.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit(self, X_train: np.ndarray, y_train=None, feature_names=None) -> "QuantumReadyPreprocessor":
        scaled = self.scaler.fit_transform(X_train)
        self.pca.fit(scaled)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))

    def fit_transform(self, X_train: np.ndarray, y_train=None, feature_names=None) -> np.ndarray:
        self.fit(X_train, y_train, feature_names)
        return self.transform(X_train)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    def summary(self) -> dict:
        return {
            "method": "pca",
            "n_components": self.n_components,
            "explained_variance_ratio": float(self.pca.explained_variance_ratio_.sum()),
        }

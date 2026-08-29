import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from .base import BaseModel


def _best_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Pick the decision threshold over `scores` that maximizes balanced
    accuracy ((sensitivity + specificity) / 2) on labeled data."""
    candidates = np.unique(scores)
    if len(candidates) == 0:
        return 0.0
    midpoints = (candidates[:-1] + candidates[1:]) / 2 if len(candidates) > 1 else candidates
    best_thresh, best_score = 0.0, -1.0
    for t in midpoints:
        pred = (scores > t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced = (sens + spec) / 2
        if balanced > best_score:
            best_score, best_thresh = balanced, float(t)
    return best_thresh


class VariationalQuantumClassifier(BaseModel):
    """Hybrid VQC: classical angle-encoding + a trainable entangling ansatz.

    Runs on any PennyLane device (default.qubit simulator here; swapping to
    a real backend is a one-line device change, not a model rewrite).
    """

    is_quantum = True

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        epochs: int = 30,
        lr: float = 0.1,
        batch_size: int = 16,
        random_state: int = 42,
    ):
        self.name = "quantum_vqc"
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.device = qml.device("default.qubit", wires=n_qubits)
        self.weights = None
        self.bias = None
        self.loss_history: list[float] = []
        self.training_time_seconds = None
        self._build_circuit()

    def _build_circuit(self):
        n_qubits = self.n_qubits

        @qml.qnode(self.device)
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            # individual single-qubit Z per wire, averaged outside the qnode -
            # a wider readout than a single wire gives every qubit's
            # information a gradient path, which matters for a shallow ansatz.
            return tuple(qml.expval(qml.PauliZ(w)) for w in range(n_qubits))

        self.circuit = circuit

    def _raw_score(self, x, weights, bias):
        return pnp.sum(pnp.stack(self.circuit(x, weights))) / self.n_qubits + bias

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "VariationalQuantumClassifier":
        rng = np.random.default_rng(self.random_state)
        X_scaled = self.angle_scaler.fit_transform(X_train)
        y_train = np.asarray(y_train)
        y_signed = pnp.array(np.where(y_train == 1, 1.0, -1.0))

        # class-balanced sample weights: plain MSE on +-1 labels lets the
        # optimizer shave error by drifting toward the majority class under
        # imbalance, which silently wrecks sensitivity. Weighting by inverse
        # class frequency keeps both classes equally "expensive" to get wrong.
        class_counts = np.bincount(y_train, minlength=2)
        class_weight = len(y_train) / (2.0 * np.maximum(class_counts, 1))
        sample_weight = pnp.array(class_weight[y_train])

        weight_shape = qml.BasicEntanglerLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        self.weights = pnp.array(rng.uniform(0, 2 * np.pi, size=weight_shape), requires_grad=True)
        self.bias = pnp.array(0.0, requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=self.lr)
        n_samples = len(X_scaled)

        def cost(weights, bias, X_batch, y_batch, w_batch):
            preds = pnp.stack([self._raw_score(x, weights, bias) for x in X_batch])
            return pnp.mean(w_batch * (preds - y_batch) ** 2)

        start = time.time()
        for _ in range(self.epochs):
            perm = rng.permutation(n_samples)
            epoch_loss = None
            for start_idx in range(0, n_samples, self.batch_size):
                idx = perm[start_idx : start_idx + self.batch_size]
                X_batch, y_batch, w_batch = X_scaled[idx], y_signed[idx], sample_weight[idx]
                (self.weights, self.bias), epoch_loss = opt.step_and_cost(
                    lambda w, b: cost(w, b, X_batch, y_batch, w_batch), self.weights, self.bias
                )
            self.loss_history.append(float(epoch_loss))
        self.training_time_seconds = time.time() - start

        # calibrate the decision threshold on training scores instead of
        # assuming the raw score is symmetric around 0 - it usually isn't.
        train_scores = self._decision_values(X_train)
        self.threshold = _best_threshold(train_scores, y_train)
        return self

    def _decision_values(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.angle_scaler.transform(X)
        return np.array([float(self._raw_score(x, self.weights, self.bias)) for x in X_scaled])

    def predict(self, X: np.ndarray) -> np.ndarray:
        threshold = getattr(self, "threshold", 0.0)
        return (self._decision_values(X) > threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._decision_values(X)
        return 1 / (1 + np.exp(-raw))

    def metadata(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_params": int(np.size(self.weights)) + 1 if self.weights is not None else None,
            "circuit_depth": self.n_layers + 1,
            "final_train_loss": self.loss_history[-1] if self.loss_history else None,
            "training_time_seconds": self.training_time_seconds,
            "decision_threshold": getattr(self, "threshold", 0.0),
        }


class QuantumKernelSVM(BaseModel):
    """Quantum kernel SVM: a fidelity kernel from a repeated angle-embedding
    feature map, fed into a classical SVM with a precomputed kernel matrix.

    Kernel computation is O(n^2) circuit evaluations, which does not scale
    to large training sets on a simulator. max_train_samples subsamples the
    training set to keep runtime tractable - a standard NISQ-era workaround,
    not a hidden shortcut; it is reported in metadata().
    """

    is_quantum = True

    def __init__(
        self,
        n_qubits: int,
        reps: int = 2,
        C: float = 1.0,
        max_train_samples: int = 120,
        random_state: int = 42,
    ):
        self.name = "quantum_qsvm"
        self.n_qubits = n_qubits
        self.reps = reps
        self.C = C
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.device = qml.device("default.qubit", wires=n_qubits)
        self.svc = SVC(kernel="precomputed", C=C, probability=True, random_state=random_state)
        self.X_train_scaled = None
        self.n_train_used = None
        self.training_time_seconds = None
        self._build_kernel_circuit()

    def _build_kernel_circuit(self):
        n_qubits = self.n_qubits
        reps = self.reps

        def feature_map(x):
            for _ in range(reps):
                qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

        @qml.qnode(self.device)
        def kernel_circuit(x1, x2):
            feature_map(x1)
            qml.adjoint(feature_map)(x2)
            return qml.probs(wires=range(n_qubits))

        self.kernel_circuit = kernel_circuit

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return float(self.kernel_circuit(x1, x2)[0])  # P(all-zeros) = |<phi(x1)|phi(x2)>|^2

    def _kernel_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.array([[self._kernel(a, b) for b in B] for a in A])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "QuantumKernelSVM":
        rng = np.random.default_rng(self.random_state)
        X_scaled_full = self.angle_scaler.fit_transform(X_train)

        if len(X_scaled_full) > self.max_train_samples:
            idx = rng.choice(len(X_scaled_full), size=self.max_train_samples, replace=False)
            X_scaled_full = X_scaled_full[idx]
            y_train = np.asarray(y_train)[idx]

        self.X_train_scaled = X_scaled_full
        self.n_train_used = len(X_scaled_full)

        start = time.time()
        K_train = self._kernel_matrix(self.X_train_scaled, self.X_train_scaled)
        self.svc.fit(K_train, y_train)
        self.training_time_seconds = time.time() - start
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.angle_scaler.transform(X)
        K_test = self._kernel_matrix(X_scaled, self.X_train_scaled)
        return self.svc.predict(K_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.angle_scaler.transform(X)
        K_test = self._kernel_matrix(X_scaled, self.X_train_scaled)
        return self.svc.predict_proba(K_test)[:, 1]

    def metadata(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "circuit_depth": self.reps * 2,
            "n_train_used": self.n_train_used,
            "n_support_vectors": int(np.sum(self.svc.n_support_))
            if hasattr(self.svc, "n_support_")
            else None,
            "training_time_seconds": self.training_time_seconds,
        }

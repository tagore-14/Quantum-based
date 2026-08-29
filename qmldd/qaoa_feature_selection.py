import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler


class QAOAFeatureSelector:
    """Selects k of the n raw features via QAOA over a relevance-vs-redundancy
    QUBO, as an alternative to PCA's dimensionality reduction.

    Why: PCA components are linear combinations of every raw feature - opaque
    to a clinician ("0.3*age - 0.1*bmi + ..."). Feature selection keeps
    whichever original, clinically interpretable columns get chosen (e.g.
    "age", "resting_heart_rate"), at the cost of solving a genuinely
    combinatorial (harder) problem instead of PCA's closed-form eigenvector
    solve. That combinatorial problem - maximize relevance to the diagnosis,
    minimize redundancy between chosen features, subject to picking exactly
    k of them - is a real QUBO, which is what QAOA is actually built for
    (unlike using QAOA as a classifier, which it is not built for).

    QAOA needs one qubit per candidate feature. A raw feature count above
    max_candidate_features (e.g. breast cancer's 30) is not tractable to
    simulate with one qubit each, so candidates are pre-filtered down to
    max_candidate_features by a cheap classical univariate score (mutual
    information) first - documented here and in summary(), not hidden.
    """

    def __init__(
        self,
        n_components: int,
        max_candidate_features: int = 12,
        p_layers: int = 2,
        redundancy_weight: float = 1.0,
        cardinality_penalty: float = 2.0,
        epochs: int = 30,
        lr: float = 0.2,
        random_state: int = 42,
    ):
        self.k = n_components
        self.max_candidate_features = max_candidate_features
        self.p_layers = p_layers
        self.redundancy_weight = redundancy_weight
        self.cardinality_penalty = cardinality_penalty
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

        self.candidate_idx = None
        self.n_candidates = None
        self.selected_idx = None
        self.selected_feature_names = None
        self.scaler = None
        self.final_cost = None
        self.training_time_seconds = None

    def _build_qubo(self, X_cand: np.ndarray, y_train: np.ndarray):
        mi = mutual_info_classif(X_cand, y_train, random_state=self.random_state)
        relevance = mi / mi.max() if mi.max() > 0 else np.zeros_like(mi)
        corr = np.corrcoef(X_cand, rowvar=False)
        redundancy = np.abs(np.nan_to_num(corr))

        k = self.k
        L = -relevance + self.cardinality_penalty * (1 - 2 * k)
        Q = self.redundancy_weight * redundancy + 2 * self.cardinality_penalty
        np.fill_diagonal(Q, 0.0)
        return L, Q

    @staticmethod
    def _qubo_to_ising(L: np.ndarray, Q: np.ndarray):
        h = -L / 2 - Q.sum(axis=1) / 4
        J = Q / 4
        return h, J

    def _build_qaoa_circuit(self, h: np.ndarray, J: np.ndarray, n_qubits: int):
        dev = qml.device("default.qubit", wires=n_qubits)
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
        p_layers = self.p_layers

        def cost_layer(gamma):
            for i in range(n_qubits):
                qml.RZ(2 * gamma * h[i], wires=i)
            for (i, j) in pairs:
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma * J[i, j], wires=j)
                qml.CNOT(wires=[i, j])

        def mixer_layer(beta):
            for i in range(n_qubits):
                qml.RX(2 * beta, wires=i)

        @qml.qnode(dev)
        def circuit(gammas, betas):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for l in range(p_layers):
                cost_layer(gammas[l])
                mixer_layer(betas[l])
            obs = [qml.PauliZ(i) for i in range(n_qubits)]
            obs += [qml.PauliZ(i) @ qml.PauliZ(j) for (i, j) in pairs]
            return [qml.expval(o) for o in obs]

        return circuit, pairs

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str] | None = None) -> "QAOAFeatureSelector":
        rng = np.random.default_rng(self.random_state)
        y_train = np.asarray(y_train)
        n_raw = X_train.shape[1]

        if n_raw > self.max_candidate_features:
            mi_full = mutual_info_classif(X_train, y_train, random_state=self.random_state)
            self.candidate_idx = np.sort(np.argsort(mi_full)[::-1][: self.max_candidate_features])
        else:
            self.candidate_idx = np.arange(n_raw)
        self.n_candidates = len(self.candidate_idx)

        X_cand = X_train[:, self.candidate_idx]
        L, Q = self._build_qubo(X_cand, y_train)
        h, J = self._qubo_to_ising(L, Q)
        circuit, pairs = self._build_qaoa_circuit(h, J, self.n_candidates)

        gammas = pnp.array(rng.uniform(0, np.pi, size=self.p_layers), requires_grad=True)
        betas = pnp.array(rng.uniform(0, np.pi, size=self.p_layers), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=self.lr)

        def cost_fn(gammas, betas):
            outs = circuit(gammas, betas)
            n = self.n_candidates
            z = outs[:n]
            zz = outs[n:]
            total = sum(h[i] * z[i] for i in range(n))
            total = total + sum(J[pairs[idx][0], pairs[idx][1]] * zz[idx] for idx in range(len(pairs)))
            return total

        start = time.time()
        cost_val = None
        for _ in range(self.epochs):
            (gammas, betas), cost_val = opt.step_and_cost(cost_fn, gammas, betas)
        self.training_time_seconds = time.time() - start
        self.final_cost = float(cost_val)

        final_outs = circuit(gammas, betas)
        z_marginals = np.array([float(v) for v in final_outs[: self.n_candidates]])
        selection_prob = (1 - z_marginals) / 2  # P(qubit measures |1>) under this convention
        top_k_local = np.sort(np.argsort(selection_prob)[::-1][: self.k])
        self.selected_idx = self.candidate_idx[top_k_local]

        if feature_names is not None:
            self.selected_feature_names = [feature_names[i] for i in self.selected_idx]
        else:
            self.selected_feature_names = [f"feature_{i}" for i in self.selected_idx]

        self.scaler = StandardScaler().fit(X_train[:, self.selected_idx])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X[:, self.selected_idx])

    def fit_transform(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        self.fit(X_train, y_train, feature_names=feature_names)
        return self.transform(X_train)

    def summary(self) -> dict:
        return {
            "method": "qaoa_feature_select",
            "k": self.k,
            "candidate_pool_size": self.n_candidates,
            "p_layers": self.p_layers,
            "selected_feature_names": self.selected_feature_names,
            "final_qaoa_cost": self.final_cost,
            "training_time_seconds": self.training_time_seconds,
        }

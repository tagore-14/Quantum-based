from .base import BaseModel
from .classical import logistic_regression, random_forest, small_nn, svm_rbf
from .quantum import QuantumKernelSVM, VariationalQuantumClassifier

MODEL_REGISTRY = {
    "classical_logreg": lambda n_qubits=None, **kw: logistic_regression(**kw),
    "classical_svm": lambda n_qubits=None, **kw: svm_rbf(**kw),
    "classical_rf": lambda n_qubits=None, **kw: random_forest(**kw),
    "classical_nn": lambda n_qubits=None, **kw: small_nn(**kw),
    "quantum_vqc": lambda n_qubits, **kw: VariationalQuantumClassifier(n_qubits=n_qubits, **kw),
    "quantum_qsvm": lambda n_qubits, **kw: QuantumKernelSVM(n_qubits=n_qubits, **kw),
}

__all__ = ["BaseModel", "MODEL_REGISTRY"]

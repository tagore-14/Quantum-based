import numpy as np
import pandas as pd
import pytest

from qmldd.data import DATA_LOADERS
from qmldd.data.custom import build_dataset_from_dataframe, detect_id_like_columns
from qmldd.evaluation import evaluate_model
from qmldd.models import MODEL_REGISTRY
from qmldd.preprocessing import QuantumReadyPreprocessor
from qmldd.qaoa_feature_selection import QAOAFeatureSelector
from qmldd.validation import check_class_balance, clip_n_components, subsample_dataset


@pytest.mark.parametrize("dataset_name", list(DATA_LOADERS))
def test_data_loader_shapes(dataset_name):
    dataset = DATA_LOADERS[dataset_name]().load()
    assert dataset.X.ndim == 2
    assert dataset.X.shape[0] == dataset.y.shape[0]
    assert set(np.unique(dataset.y)).issubset({0, 1})
    assert dataset.X.shape[0] > 50


def test_preprocessor_reduces_dimensionality():
    dataset = DATA_LOADERS["breast_cancer"]().load()
    prep = QuantumReadyPreprocessor(n_components=4)
    X_reduced = prep.fit_transform(dataset.X)
    assert X_reduced.shape == (dataset.X.shape[0], 4)
    assert 0 < prep.explained_variance_ratio.sum() <= 1.0


@pytest.mark.parametrize("model_type", ["classical_logreg", "classical_rf"])
def test_classical_models_end_to_end(model_type):
    dataset = DATA_LOADERS["breast_cancer"]().load()
    prep = QuantumReadyPreprocessor(n_components=4)
    split = int(len(dataset.X) * 0.75)
    X_train_raw, X_test_raw = dataset.X[:split], dataset.X[split:]
    y_train, y_test = dataset.y[:split], dataset.y[split:]
    X_train = prep.fit_transform(X_train_raw)
    X_test = prep.transform(X_test_raw)

    model = MODEL_REGISTRY[model_type](n_qubits=4)
    result = evaluate_model(model, X_train, y_train, X_test, y_test)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.sensitivity <= 1.0
    assert 0.0 <= result.specificity <= 1.0


def test_vqc_runs_and_returns_valid_metrics():
    """Not asserting accuracy thresholds - a couple of epochs on a tiny slice
    is only meant to prove the circuit/optimizer wiring works end to end."""
    dataset = DATA_LOADERS["breast_cancer"]().load()
    prep = QuantumReadyPreprocessor(n_components=4)
    X_train_raw, y_train = dataset.X[:40], dataset.y[:40]
    X_test_raw, y_test = dataset.X[40:60], dataset.y[40:60]
    X_train = prep.fit_transform(X_train_raw)
    X_test = prep.transform(X_test_raw)

    model = MODEL_REGISTRY["quantum_vqc"](n_qubits=4, n_layers=1, epochs=2, batch_size=8)
    result = evaluate_model(model, X_train, y_train, X_test, y_test)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.metadata["n_params"] > 0


def _make_mixed_dataframe(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "patient_id": [f"P{i:05d}" for i in range(n)],
        "age": rng.normal(55, 12, n),
        "bmi": rng.normal(27, 5, n),
        "smoker": rng.choice(["yes", "no", None], size=n, p=[0.3, 0.65, 0.05]),
        "gender": rng.choice(["M", "F"], size=n),
    })
    df.loc[rng.choice(n, 10, replace=False), "age"] = np.nan
    score = 0.05 * df["bmi"] + 0.03 * df["age"].fillna(df["age"].mean()) + rng.normal(0, 1, n)
    df["diagnosis"] = np.where(score > score.median(), "disease", "healthy")
    return df


def test_id_detection_never_flags_continuous_numeric_columns():
    """Regression test: continuous float columns are naturally near-unique
    per row and must never be mistaken for identifier columns and dropped."""
    df = _make_mixed_dataframe()
    id_like = detect_id_like_columns(df, exclude=["diagnosis"])
    assert id_like == ["patient_id"]
    assert "age" not in id_like
    assert "bmi" not in id_like


def test_build_dataset_from_dataframe_keeps_continuous_features_and_encodes_categoricals():
    df = _make_mixed_dataframe()
    dataset, report = build_dataset_from_dataframe(
        df, target_col="diagnosis", positive_value="disease", dataset_name="custom_test"
    )
    assert "age" in dataset.feature_names
    assert "bmi" in dataset.feature_names
    assert any(f.startswith("smoker_") for f in dataset.feature_names)
    assert report["dropped_id_like"] == ["patient_id"]
    assert not np.isnan(dataset.X).any()  # missing values were imputed


def test_validation_helpers():
    df = _make_mixed_dataframe(n=300)
    dataset, _ = build_dataset_from_dataframe(df, target_col="diagnosis", positive_value="disease")

    assert check_class_balance(dataset) == []  # roughly balanced by construction

    clipped, warnings = clip_n_components(dataset, requested=100)
    assert clipped == dataset.X.shape[1]
    assert len(warnings) == 1

    subsampled, was_subsampled = subsample_dataset(dataset, max_rows=50)
    assert was_subsampled
    assert len(subsampled.y) == 50
    _, was_subsampled_noop = subsample_dataset(dataset, max_rows=10_000)
    assert not was_subsampled_noop


def test_qaoa_feature_selector_selects_k_original_columns():
    """QAOA feature selection should return real (interpretable) columns
    from the input, not PCA-style linear combinations - and respect the
    requested k even when the pool is pre-filtered."""
    dataset = DATA_LOADERS["heart_disease"]().load()
    split = int(len(dataset.X) * 0.75)
    X_train, y_train = dataset.X[:split], dataset.y[:split]
    X_test = dataset.X[split:]

    selector = QAOAFeatureSelector(n_components=4, max_candidate_features=8, p_layers=1, epochs=8)
    X_train_proc = selector.fit_transform(X_train, y_train, feature_names=dataset.feature_names)
    X_test_proc = selector.transform(X_test)

    assert X_train_proc.shape == (len(X_train), 4)
    assert X_test_proc.shape == (len(X_test), 4)
    assert len(selector.selected_idx) == 4
    assert len(set(selector.selected_idx)) == 4  # no duplicate feature picked twice
    assert all(name in dataset.feature_names for name in selector.selected_feature_names)

    summary = selector.summary()
    assert summary["method"] == "qaoa_feature_select"
    assert summary["candidate_pool_size"] == 8


def test_qaoa_feature_selector_prefilters_large_feature_counts():
    """breast_cancer has 30 raw features - QAOA needs one qubit per candidate,
    so the candidate pool must be capped rather than using all 30 qubits."""
    dataset = DATA_LOADERS["breast_cancer"]().load()
    X_train, y_train = dataset.X[:100], dataset.y[:100]

    selector = QAOAFeatureSelector(n_components=3, max_candidate_features=10, p_layers=1, epochs=5)
    selector.fit(X_train, y_train, feature_names=dataset.feature_names)

    assert selector.n_candidates == 10
    assert len(selector.selected_idx) == 3


def test_preprocessor_registry_shares_interface():
    """Both preprocessing methods must be interchangeable from the
    pipeline's point of view: same fit_transform/transform/summary shape."""
    from qmldd.pipeline import PREPROCESSOR_REGISTRY

    dataset = DATA_LOADERS["heart_disease"]().load()
    X_train, y_train = dataset.X[:100], dataset.y[:100]
    X_test = dataset.X[100:120]

    for method, factory in [
        ("pca", {}),
        ("qaoa_feature_select", {"max_candidate_features": 6, "p_layers": 1, "epochs": 5}),
    ]:
        prep = PREPROCESSOR_REGISTRY[method](n_components=3, **factory)
        X_train_proc = prep.fit_transform(X_train, y_train, feature_names=dataset.feature_names)
        X_test_proc = prep.transform(X_test)
        assert X_train_proc.shape == (100, 3)
        assert X_test_proc.shape == (20, 3)
        summary = prep.summary()
        assert summary["method"] == method

import numpy as np
import shap


def classical_shap_summary(
    model, X_background: np.ndarray, X_explain: np.ndarray, feature_names: list[str],
    max_background: int = 50, nsamples: int = 100,
) -> dict:
    """SHAP KernelExplainer attribution for classical (or any cheap-to-query)
    models. Not used for quantum models - see perturbation_attribution."""
    background = X_background[:max_background]
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_explain, nsamples=nsamples, silent=True)
    mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
    return dict(zip(feature_names, mean_abs.tolist()))


def perturbation_attribution(
    model, X_explain: np.ndarray, feature_names: list[str],
    epsilon: float = 0.3, n_repeats: int = 5, random_state: int = 42,
) -> dict:
    """Model-agnostic sensitivity attribution: average change in predicted
    probability when each feature is perturbed with Gaussian noise.

    Used for quantum models because SHAP's KernelExplainer would require
    thousands of extra quantum-circuit evaluations per explained sample.
    This is a classical surrogate technique applied to the model's
    black-box predict_proba - genuine quantum-native explainability is
    still an open research problem, not something this approximates away.
    """
    rng = np.random.default_rng(random_state)
    base_proba = model.predict_proba(X_explain)
    n_features = X_explain.shape[1]
    importances = np.zeros(n_features)
    for f in range(n_features):
        deltas = []
        for _ in range(n_repeats):
            X_perturbed = X_explain.copy()
            X_perturbed[:, f] += rng.normal(0, epsilon, size=len(X_explain))
            perturbed_proba = model.predict_proba(X_perturbed)
            deltas.append(float(np.mean(np.abs(perturbed_proba - base_proba))))
        importances[f] = np.mean(deltas)
    return dict(zip(feature_names, importances.tolist()))


def explain_model(model, X_background, X_explain, feature_names: list[str]) -> dict:
    """Dispatch to the appropriate explainer based on the model type."""
    if getattr(model, "is_quantum", False):
        return {"method": "perturbation_attribution", "importances": perturbation_attribution(model, X_explain, feature_names)}
    return {"method": "shap_kernel_explainer", "importances": classical_shap_summary(model, X_background, X_explain, feature_names)}

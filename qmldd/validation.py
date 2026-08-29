import numpy as np

from .data.base import Dataset


def check_class_balance(dataset: Dataset, min_per_class: int = 10) -> list[str]:
    """Non-fatal warnings surfaced to the user before running, so a small or
    lopsided dataset doesn't produce results that look confident but aren't."""
    warnings = []
    n_pos = int(dataset.y.sum())
    n_neg = int(len(dataset.y) - n_pos)
    if n_pos < min_per_class or n_neg < min_per_class:
        warnings.append(
            f"Very few samples in one class (positive={n_pos}, negative={n_neg}). "
            f"Results with fewer than {min_per_class} samples per class are unreliable."
        )
    if min(n_pos, n_neg) > 0:
        ratio = max(n_pos, n_neg) / min(n_pos, n_neg)
        if ratio > 10:
            warnings.append(
                f"Severe class imbalance detected ({n_pos} positive vs {n_neg} negative). "
                "Sensitivity/specificity may be volatile on the held-out test split."
            )
    return warnings


def clip_n_components(dataset: Dataset, requested: int) -> tuple[int, list[str]]:
    """PCA can't produce more components than input features - clip and warn
    instead of letting sklearn raise a confusing error."""
    warnings = []
    max_allowed = dataset.X.shape[1]
    if requested > max_allowed:
        warnings.append(
            f"Requested {requested} components/qubits but only {max_allowed} features are "
            f"available after encoding; using {max_allowed} instead."
        )
        return max_allowed, warnings
    return requested, warnings


def subsample_dataset(dataset: Dataset, max_rows: int, random_state: int = 42) -> tuple[Dataset, bool]:
    """Cap dataset size uniformly (applied to ALL models, not just quantum
    ones) so every model in the run still sees the identical train/test
    split - the fair-comparison guarantee holds even when we have to
    subsample for a quantum model's sake."""
    if len(dataset.y) <= max_rows:
        return dataset, False
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(dataset.y), size=max_rows, replace=False)
    subsampled = Dataset(
        name=dataset.name,
        X=dataset.X[idx],
        y=dataset.y[idx],
        feature_names=dataset.feature_names,
        positive_label=dataset.positive_label,
        negative_label=dataset.negative_label,
    )
    return subsampled, True

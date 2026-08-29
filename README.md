# Hybrid Quantum-Classical ML Platform for Early Disease Detection

A config-driven, plugin-architected platform for benchmarking hybrid
quantum-classical models against classical baselines on biomedical
classification tasks. Built for the SIH problem statement of the same name.

## Why hybrid, and why honestly

Current quantum hardware and simulators only handle a handful of qubits
before noise and simulation cost make training impractical. That means:
classical preprocessing reduces high-dimensional biomedical data down to a
small qubit budget (default: 4 features/qubits via PCA) before any quantum
model sees it, and the quantum models here are not expected to beat strong
classical baselines on small tabular datasets today. The platform's value is
in the architecture: a fair, reproducible benchmarking harness that stays
useful as quantum hardware matures, with a plugin design that scales to new
diseases and new model types without touching the core pipeline.

## No-code path for non-ML users

The point of a *platform* over a one-off model is that a biomedical
researcher with no ML background should be able to use it directly - upload
data, pick models, get results, the way you'd retrain YOLOv8 on a new set of
images without touching training code. `streamlit run dashboard/app.py` ->
"Run your own dataset":

- Upload any CSV, pick which column is the diagnosis/outcome and which value
  means "positive" (the picker defaults to a plausibly-named outcome column
  and a plausible positive value, e.g. "yes"/"positive"/"present", instead of
  guessing the first column, which is often an ID).
- The platform auto-detects and drops identifier-like columns (patient IDs,
  free text), imputes missing values, and one-hot encodes categoricals -
  continuous numeric measurements are explicitly protected from ever being
  mistaken for identifiers (a real bug caught during development: naturally
  near-unique float columns like BMI were briefly being auto-dropped).
- Non-binary targets are supported as one-vs-rest (pick the positive value,
  everything else becomes negative), with the tradeoff stated on screen.
- Guardrails surface as on-screen warnings, not silent failures or crashes:
  severe class imbalance, too few samples per class, a requested qubit count
  larger than the available features, or a dataset too large to run the
  quantum models on interactively (auto-subsampled uniformly across ALL
  selected models, preserving the fair-comparison guarantee).
- Choose PCA or QAOA feature selection for the dimensionality/qubit-budget
  reduction step (see below) directly from the UI.
- Pick models via checkboxes, tune advanced settings if you want to, click
  "Run Benchmark", watch live per-model progress, get the same metrics
  table/charts/explainability as the curated datasets - auto-saved to
  `results/` so it shows up under "View saved results" too.

This reuses the exact same `DataLoader` -> `Preprocessor` -> `Model` ->
`Evaluator` pipeline as the curated datasets; the only new code is the
ingestion layer (`qmldd/data/custom.py`) and validation guardrails
(`qmldd/validation.py`), not a parallel implementation.

## Architecture

```
qmldd/
  data/          # DataLoader plugins (breast_cancer, heart_disease, diabetes)
  preprocessing.py           # StandardScaler -> PCA (default preprocessing method)
  qaoa_feature_selection.py  # QAOA-based feature selection (alternative method)
  models/        # BaseModel plugins: classical baselines + quantum VQC/QSVM
  evaluation.py  # accuracy / sensitivity / specificity / F1 / AUC / timing
  explain.py     # SHAP (classical) + perturbation attribution (quantum)
  pipeline.py    # Orchestrator: config -> run -> results JSON
  cli.py         # `python -m qmldd.cli run --config ...`
configs/         # One YAML per experiment (dataset + preprocessing + models)
dashboard/app.py # Streamlit UI over results/*.json
results/         # Benchmark output JSON, read by the dashboard
tests/           # pytest smoke tests for every plugin
```

**Extending it:** a new disease/dataset is a new `DataLoader` subclass
registered in `qmldd/data/__init__.py`. A new model (classical or quantum) is
a new `BaseModel` subclass registered in `qmldd/models/__init__.py`. Neither
requires touching the pipeline, evaluation, or dashboard code.

## Datasets

| Dataset | Source | Positive (disease) class |
|---|---|---|
| `breast_cancer` | scikit-learn (UCI Breast Cancer Wisconsin) | malignant |
| `heart_disease` | OpenML `heart-statlog` (Cleveland-derived) | heart disease present |
| `diabetes` | OpenML `diabetes` (Pima Indians) | tested positive |

All three are small, well-understood tabular benchmarks chosen because they
fit within a NISQ-scale qubit budget after PCA - not imaging or genomics,
which would need lossy compression far more aggressive than PCA to reach 4-8
features.

## Models

- **Classical baselines:** Logistic Regression, RBF-kernel SVM, Random
  Forest, small MLP - all scikit-learn, trained on the identical
  PCA-reduced features the quantum models see.
- **Quantum VQC:** angle-embedding feature map + `BasicEntanglerLayers`
  ansatz, trained with PennyLane's Adam optimizer on a class-weighted MSE
  loss, with a calibrated decision threshold (not a hardcoded 0) since
  imbalanced classes otherwise skew a raw regression-style loss.
- **Quantum kernel SVM (QSVM):** a fidelity kernel from a repeated
  angle-embedding feature map, computed via PennyLane and fed into a
  classical SVM with a precomputed kernel matrix. Kernel computation is
  O(n^2) circuit evaluations, so training-set size is capped
  (`max_train_samples`) - a documented, standard NISQ-era workaround, not a
  hidden shortcut.

All quantum models run on PennyLane's `default.qubit` simulator; swapping to
a real backend (IBM Quantum, AWS Braket, etc.) is a device change, not a
model rewrite.

## Preprocessing: PCA vs. QAOA feature selection

`preprocessing.method` in a config picks between two ways to get down to the
qubit budget, selectable per-experiment (`qmldd.pipeline.PREPROCESSOR_REGISTRY`):

- **`pca`** (default): PCA components are linear combinations of every raw
  feature - typically the more accurate reduction, but opaque
  ("0.3*age - 0.1*bmi + ...").
- **`qaoa_feature_select`**: solves a real combinatorial optimization problem
  with QAOA - maximize relevance to the diagnosis (mutual information),
  minimize redundancy between chosen features (pairwise correlation), select
  exactly k of them - formulated as a QUBO, converted to an Ising
  Hamiltonian, and optimized via a PennyLane QAOA circuit. The payoff: the
  chosen features stay in their original, clinician-readable units (e.g.
  "resting_blood_pressure", "oldpeak") rather than PCA's abstract axes,
  which matters directly for the explainability objective - SHAP/perturbation
  output then names real measurements instead of `pc_0`, `pc_1`.
  QAOA needs one qubit per candidate feature, so raw feature counts above
  `max_candidate_features` (default 12) are pre-filtered down by a cheap
  classical relevance score first - this is why breast cancer's 30 features
  are usable at all, not a hidden shortcut (see `summary()`'s
  `candidate_pool_size`). This is *not* "QAOA as a classifier" - QAOA solves
  the combinatorial feature-selection problem it's actually built for;
  classification is still done by VQC/QSVM/the classical baselines on
  whichever features it selects.

## Explainability

- Classical models: SHAP `KernelExplainer` on `predict_proba`.
- Quantum models: perturbation-based sensitivity attribution (how much the
  predicted probability moves when each feature is perturbed). This is a
  classical surrogate applied to the quantum model's black-box output, not
  SHAP, and not a claim that quantum-native explainability is solved - it
  isn't, and the platform doesn't pretend otherwise.

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Running an experiment

```bash
python -m qmldd.cli run --config configs/breast_cancer.yaml
```

This trains every model listed in the config on the same preprocessed
features, evaluates each on a held-out test set, and writes a results JSON
to the path in `output:`. See `configs/` for the three ready-made
per-dataset benchmark configs, and `configs/*_smoke.yaml` for fast
(few-epoch) sanity-check variants.

## Viewing results

```bash
streamlit run dashboard/app.py
```

Picks up every `results/*.json` file and lets you compare hybrid vs.
classical models per dataset: metrics table, accuracy/sensitivity/specificity
bar charts, training time, per-model metadata (circuit depth, param count,
etc.), and feature-importance charts.

## Tests

```bash
python -m pytest tests/ -q
```

## Honest limitations

- No claim of quantum advantage: on these dataset sizes, classical
  baselines are expected to match or beat the quantum models on raw
  accuracy. The pitch is architectural readiness and fair benchmarking, not
  a performance win today.
- QSVM training-set subsampling and VQC's shallow 2-layer ansatz are
  simulator/time-budget-driven choices, documented in `metadata()` on every
  model's output rather than hidden.
- Quantum explainability here is a classical approximation, not a
  quantum-native method - none exists yet in the literature at this level
  of maturity.

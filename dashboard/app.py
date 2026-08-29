import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmldd.data.custom import build_dataset_from_dataframe
from qmldd.models import MODEL_REGISTRY
from qmldd.pipeline import run_on_dataset
from qmldd.validation import check_class_balance, clip_n_components, subsample_dataset

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_LABELS = {
    "classical_logreg": "Logistic Regression (classical)",
    "classical_rf": "Random Forest (classical)",
    "classical_svm": "SVM - RBF kernel (classical)",
    "classical_nn": "Small Neural Net (classical)",
    "quantum_vqc": "Variational Quantum Classifier (quantum)",
    "quantum_qsvm": "Quantum Kernel SVM (quantum)",
}
DEFAULT_CHECKED = {"classical_logreg", "classical_rf", "quantum_vqc"}
QUANTUM_MODELS = {"quantum_vqc", "quantum_qsvm"}

TARGET_NAME_HINTS = ["diagnos", "outcome", "target", "label", "class", "result", "status"]
POSITIVE_VALUE_HINTS = ["yes", "positive", "true", "present", "malignant", "1", "sick", "disease"]


def guess_target_column_index(columns: list[str]) -> int:
    """Non-ML users often don't know to skip past an ID column - default the
    picker to a plausibly-named outcome column, or the last column (the
    common convention), rather than the first column (often an ID)."""
    lowered = [str(c).lower() for c in columns]
    for hint in TARGET_NAME_HINTS:
        for i, name in enumerate(lowered):
            if hint in name:
                return i
    return len(columns) - 1


def guess_positive_value_index(values: list) -> int:
    lowered = [str(v).lower() for v in values]
    for hint in POSITIVE_VALUE_HINTS:
        for i, v in enumerate(lowered):
            if hint == v:
                return i
    return 0

st.set_page_config(page_title="Hybrid QML Disease Detection", layout="wide")
st.title("Hybrid Quantum-Classical ML Platform for Early Disease Detection")
st.caption(
    "Benchmarks variational quantum classifiers and quantum kernel SVMs against "
    "classical baselines on the same preprocessed features, for a fair comparison."
)

mode = st.sidebar.radio("Mode", ["View saved results", "Run your own dataset"])


# ---------------------------------------------------------------------------
# Shared rendering
# ---------------------------------------------------------------------------

def render_results(data: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train samples", data["n_train"])
    col2.metric("Test samples", data["n_test"])
    col3.metric("Raw features", data["raw_feature_count"])
    col4.metric("Qubits / components", data["n_components"])

    prep = data.get("preprocessing_summary", {})
    if prep.get("method") == "qaoa_feature_select":
        names = ", ".join(prep.get("selected_feature_names") or [])
        st.caption(
            f"Preprocessing: QAOA feature selection (candidate pool: {prep.get('candidate_pool_size')}, "
            f"p={prep.get('p_layers')}). Selected features: {names}"
        )
    elif prep.get("method") == "pca":
        st.caption(f"Preprocessing: PCA, explained variance retained: {prep.get('explained_variance_ratio', 0):.1%}")

    df = pd.DataFrame(data["results"])
    if df.empty:
        st.warning("No model results to show.")
        return
    df["kind"] = df["is_quantum"].map({True: "Quantum", False: "Classical"})

    st.subheader("Benchmark: hybrid quantum models vs. classical baselines")
    metric_cols = ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
    display_df = df[["model_name", "kind"] + metric_cols + ["training_time_seconds", "inference_time_seconds"]].copy()
    display_df.columns = [
        "Model", "Kind", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC AUC",
        "Train time (s)", "Inference time (s)",
    ]
    st.dataframe(
        display_df.style.format(
            {c: "{:.3f}" for c in ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC AUC"]}
        ).format({"Train time (s)": "{:.2f}", "Inference time (s)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Accuracy / Sensitivity / Specificity")
        st.bar_chart(df.set_index("model_name")[["accuracy", "sensitivity", "specificity"]])
    with right:
        st.subheader("Training time (seconds)")
        st.bar_chart(df.set_index("model_name")[["training_time_seconds"]])

    st.subheader("Model details")
    for _, row in df.iterrows():
        with st.expander(f"{row['model_name']} ({row['kind']})"):
            st.json(row["metadata"])

    explanations = data.get("explanations", {})
    if explanations:
        st.subheader("Explainability: feature importance")
        st.caption(
            "Classical models use SHAP. Quantum models use perturbation-based "
            "sensitivity attribution, a classical surrogate technique applied to "
            "the quantum model's predictions - not SHAP, and not a claim of "
            "quantum-native explainability (an open research problem)."
        )
        exp_model = st.selectbox("Model", list(explanations.keys()), key="explain_model_select")
        exp_data = explanations[exp_model]
        if "error" in exp_data:
            st.error(exp_data["error"])
        else:
            st.caption(f"Method: {exp_data['method']}")
            imp_df = pd.DataFrame(
                list(exp_data["importances"].items()), columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=False)
            st.bar_chart(imp_df.set_index("Feature"))


# ---------------------------------------------------------------------------
# Mode: view saved results
# ---------------------------------------------------------------------------

def load_saved_results() -> dict[str, dict]:
    results = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            if "results" in data:
                results[f"{data['dataset']} ({path.name})"] = data
        except (json.JSONDecodeError, KeyError):
            continue
    return results


if mode == "View saved results":
    all_results = load_saved_results()
    if not all_results:
        st.warning(
            "No result files found in results/. Run an experiment first via the CLI "
            "(`python -m qmldd.cli run --config configs/breast_cancer.yaml`) or use "
            "'Run your own dataset' in the sidebar."
        )
        st.stop()
    key = st.sidebar.selectbox("Saved run", list(all_results.keys()))
    render_results(all_results[key])


# ---------------------------------------------------------------------------
# Mode: run your own dataset (no-code path for non-ML users)
# ---------------------------------------------------------------------------

else:
    st.markdown(
        "Upload a CSV, tell the platform which column is the diagnosis and which "
        "models to try, and it handles preprocessing, training, and evaluation - "
        "no code required."
    )

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.info("Waiting for a CSV upload. Any tabular dataset with a diagnosis/outcome column works.")
        st.stop()

    df = pd.read_csv(uploaded)
    st.write(f"**{df.shape[0]} rows, {df.shape[1]} columns**")
    st.dataframe(df.head(10), use_container_width=True)

    default_target_idx = guess_target_column_index(list(df.columns))
    target_col = st.selectbox(
        "Which column is the diagnosis/outcome you want to predict?", df.columns, index=default_target_idx,
    )
    unique_vals = df[target_col].dropna().unique().tolist()

    if len(unique_vals) < 2:
        st.error(f"'{target_col}' only has one distinct value - it can't be used as a prediction target.")
        st.stop()

    positive_value = st.selectbox(
        "Which value means 'disease present / positive'?", unique_vals,
        index=guess_positive_value_index(unique_vals),
        help="Every other value will be treated as the negative class.",
    )
    if len(unique_vals) > 2:
        st.info(
            f"'{target_col}' has {len(unique_vals)} distinct values. This platform benchmarks "
            f"binary classifiers, so '{positive_value}' will be treated as positive and the "
            f"other {len(unique_vals) - 1} value(s) grouped together as negative (one-vs-rest)."
        )

    drop_cols = st.multiselect(
        "Columns to exclude (e.g. patient ID, notes) - optional",
        [c for c in df.columns if c != target_col],
    )

    complexity = st.select_slider(
        "Model complexity / qubit budget",
        options=["Low (4 qubits)", "Medium (6 qubits)", "High (8 qubits)"],
        value="Low (4 qubits)",
        help="Higher uses more PCA components / qubits - more expressive but slower, "
             "especially for the quantum models.",
    )
    n_components = {"Low (4 qubits)": 4, "Medium (6 qubits)": 6, "High (8 qubits)": 8}[complexity]

    prep_method_label = st.radio(
        "Feature reduction method",
        ["PCA (Recommended)", "QAOA feature selection"],
        horizontal=True,
        help=(
            "PCA compresses all features into abstract combinations - usually more accurate. "
            "QAOA feature selection instead picks whichever original columns to keep (e.g. "
            "'age', 'bmi') by solving a relevance-vs-redundancy optimization on a quantum "
            "circuit - less accurate typically, but the result stays in clinician-readable units."
        ),
    )
    prep_method = "pca" if prep_method_label.startswith("PCA") else "qaoa_feature_select"

    st.write("**Select models to benchmark:**")
    cols = st.columns(3)
    selected_models = []
    for i, (mtype, label) in enumerate(MODEL_LABELS.items()):
        checked = cols[i % 3].checkbox(label, value=mtype in DEFAULT_CHECKED, key=f"chk_{mtype}")
        if checked:
            selected_models.append(mtype)

    with st.expander("Advanced settings"):
        test_size = st.slider("Test set fraction", 0.1, 0.4, 0.25, 0.05)
        vqc_epochs = st.slider("VQC training epochs", 5, 60, 15, help="More epochs = better fit, slower.")
        vqc_layers = st.slider("VQC ansatz layers", 1, 4, 2)
        qsvm_max_train = st.slider("QSVM max training samples", 30, 200, 80,
                                    help="Quantum kernel cost grows quadratically with this.")
        max_rows_for_quantum = st.slider("Row cap before subsampling for quantum models", 200, 2000, 600)
        explain_enabled = st.checkbox("Compute feature-importance explanations", value=True)
        if prep_method == "qaoa_feature_select":
            st.caption("QAOA feature selection settings")
            qaoa_max_candidates = st.slider(
                "Candidate feature pool size", 4, 16, 12,
                help="QAOA uses one qubit per candidate - raw features beyond this are "
                     "pre-filtered out by a quick classical relevance score first.",
            )
            qaoa_p_layers = st.slider("QAOA layers (p)", 1, 4, 2)
            qaoa_epochs = st.slider("QAOA optimization epochs", 10, 60, 30)

    if st.button("Run Benchmark", type="primary"):
        if not selected_models:
            st.error("Select at least one model to run.")
            st.stop()

        try:
            dataset, report = build_dataset_from_dataframe(
                df, target_col=target_col, positive_value=positive_value,
                drop_cols=drop_cols, dataset_name=f"custom_{Path(uploaded.name).stem}",
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        st.write("**Data preparation report:**")
        st.json(report)

        for w in check_class_balance(dataset):
            st.warning(w)

        n_components, clip_warnings = clip_n_components(dataset, n_components)
        for w in clip_warnings:
            st.warning(w)

        if any(m in QUANTUM_MODELS for m in selected_models):
            dataset, was_subsampled = subsample_dataset(dataset, max_rows=max_rows_for_quantum)
            if was_subsampled:
                st.info(
                    f"Dataset subsampled to {max_rows_for_quantum} rows so the quantum models stay "
                    "interactive. All selected models (including classical ones) train on this same "
                    "subsample, preserving a fair comparison."
                )

        model_specs = []
        for mtype in selected_models:
            spec = {"type": mtype}
            if mtype == "quantum_vqc":
                spec.update({"epochs": vqc_epochs, "n_layers": vqc_layers})
            elif mtype == "quantum_qsvm":
                spec.update({"max_train_samples": qsvm_max_train})
            model_specs.append(spec)

        preprocessing_cfg = {"n_components": n_components, "method": prep_method}
        if prep_method == "qaoa_feature_select":
            preprocessing_cfg.update({
                "max_candidate_features": qaoa_max_candidates,
                "p_layers": qaoa_p_layers,
                "epochs": qaoa_epochs,
            })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"{dataset.name}_{timestamp}.json"
        config = {
            "test_size": test_size,
            "random_state": 42,
            "preprocessing": preprocessing_cfg,
            "models": model_specs,
            "explain": explain_enabled,
            "explain_samples": 15,
            "output": str(output_path),
        }

        status_box = st.status("Running benchmark...", expanded=True)

        def on_progress(event, payload):
            if event == "dataset_ready":
                prep = payload["preprocessing_summary"]
                if prep.get("method") == "qaoa_feature_select":
                    detail = f"selected: {', '.join(prep.get('selected_feature_names') or [])}"
                else:
                    detail = f"{prep.get('explained_variance_ratio', 0):.1%} variance retained"
                status_box.write(
                    f"Preprocessed: {payload['n_train']} train / {payload['n_test']} test samples, {detail}."
                )
            elif event == "training_start":
                status_box.write(f"Training **{payload['model_type']}**...")
            elif event == "training_done":
                r = payload["result"]
                status_box.write(
                    f"Done: {payload['model_type']} - accuracy={r['accuracy']:.3f}, "
                    f"sensitivity={r['sensitivity']:.3f}, specificity={r['specificity']:.3f} "
                    f"({payload['wall_seconds']:.1f}s)"
                )
            elif event == "explain_done":
                status_box.write(f"Computed explainability for {payload['model_type']}.")

        t_start = time.time()
        result_data = run_on_dataset(dataset, config, verbose=False, progress_cb=on_progress)
        status_box.update(label=f"Benchmark complete in {time.time() - t_start:.1f}s", state="complete")

        st.success(f"Results saved to `{output_path.relative_to(RESULTS_DIR.parent)}` - also available under 'View saved results'.")
        render_results(result_data)

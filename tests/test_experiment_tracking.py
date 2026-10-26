import json

import pytest

from src.experiment_tracking import candidate_metrics, flat_metrics, log_comparison, model_params, summary_params
from src.training import MAIN_CANDIDATES, candidate_models


def _metrics(pr_auc: float) -> dict:
    return {
        "threshold": 0.7,
        "n": 100,
        "positive_rate": 0.05,
        "confusion_matrix": {"tn": 80, "fp": 15, "fn": 2, "tp": 3},
        "precision": 0.17,
        "recall": 0.6,
        "f1": 0.26,
        "roc_auc": 0.7,
        "pr_auc": pr_auc,
    }


@pytest.fixture
def report():
    names = list(candidate_models())
    return {
        "anchor": "aprobacion",
        "alert_volume": 0.05,
        "selection": {"metric": "pr_auc", "tolerance": 0.01, "cv_folds": 2, "candidates": list(MAIN_CANDIDATES)},
        "cohorts": {
            cohort: {"period": ["2017-01-01", "2017-06-30"], "n": 100, "late_rate": 0.05}
            for cohort in ("train", "validation", "test")
        },
        "cross_validation": {
            name: {"roc_auc_folds": [0.7, 0.72], "pr_auc_folds": [0.1, 0.12], "roc_auc": 0.71, "pr_auc": 0.11}
            for name in names
        },
        "validation": {name: _metrics(0.11) for name in names},
        "test": {name: _metrics(0.07) for name in names},
        "selected_model": "logistica_seleccion",
        "selected_threshold": 0.73,
    }


def test_flat_metrics_flattens_confusion_matrix_and_keeps_numbers():
    flat = flat_metrics(_metrics(0.07), "test_")
    assert flat["test_pr_auc"] == 0.07
    assert flat["test_confusion_matrix_tp"] == 3.0
    assert "test_confusion_matrix" not in flat
    assert all(isinstance(v, float) for v in flat.values())


def test_candidate_metrics_use_cv_means_validation_and_test(report):
    metrics = candidate_metrics(report, "arboles_seleccion")
    assert metrics["cv_pr_auc"] == 0.11
    assert metrics["val_pr_auc"] == 0.11
    assert metrics["test_pr_auc"] == 0.07


def test_model_params_read_the_classifier_of_each_candidate():
    for name, (pipeline, columns) in candidate_models().items():
        params = model_params(pipeline, columns)
        assert params["n_features"] == len(columns)
        assert params["random_state"] == 42
    trees, columns = candidate_models()["arboles_seleccion"]
    assert model_params(trees, columns)["estimator"] == "HistGradientBoostingClassifier"
    assert model_params(trees, columns)["max_depth"] == 3


def test_summary_params_name_the_choice_and_cohorts(report):
    params = summary_params(report)
    assert params["selected_model"] == "logistica_seleccion"
    assert params["test_period"] == "2017-01-01 a 2017-06-30"
    assert params["train_n"] == 100


@pytest.fixture
def mlflow_local(tmp_path, monkeypatch):
    """MLflow apuntando a una base temporal; al terminar vuelve a la configuración por defecto."""
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)  # los artefactos van a ./mlruns: que no caigan en el repo
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield mlflow
    mlflow.set_tracking_uri(None)


def test_log_comparison_writes_parent_child_runs_and_artifact(report, mlflow_local, tmp_path):
    artifact = tmp_path / "comparacion_modelos.json"
    artifact.write_text(json.dumps(report), encoding="utf-8")

    run_id = log_comparison(report, candidate_models(), [artifact], experiment="prueba")

    client = mlflow_local.tracking.MlflowClient()
    parent = client.get_run(run_id)
    assert parent.data.params["selected_model"] == "logistica_seleccion"
    assert parent.data.metrics["selected_threshold"] == pytest.approx(0.73)
    assert [a.path for a in client.list_artifacts(run_id)] == ["comparacion_modelos.json"]
    children = client.search_runs(
        [parent.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{run_id}'"
    )
    assert {c.info.run_name for c in children} == set(candidate_models())
    selected = [c.info.run_name for c in children if c.data.tags["seleccionado"] == "true"]
    assert selected == ["logistica_seleccion"]
    history = client.get_metric_history(children[0].info.run_id, "cv_pr_auc_tramo")
    assert [m.step for m in history] == [0, 1]

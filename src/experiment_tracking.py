"""Registro de la comparación de modelos en MLflow.

Lo usa `python -m src.training --mlflow`. No cambia ningún modelo ni cifra: toma
el reporte que ya arma compare_models y lo deja como un experimento de MLflow.

  - Una corrida "padre" con lo elegido (modelo, umbral, cohortes) y los artefactos:
    reports/comparacion_modelos.json y el .joblib que guarda training.
  - Una corrida hija por candidato, con sus hiperparámetros y sus métricas de
    validación cruzada (cada tramo como un paso), validación y test.

Los resultados quedan en la carpeta local que use MLflow (mlruns/ o mlflow.db,
ignoradas por Git) y se ven con `python -m mlflow ui`.
"""

from pathlib import Path

from sklearn.pipeline import Pipeline

DEFAULT_EXPERIMENT = "datacraft-retraso-pedidos"
CV_FOLD_METRICS = ("roc_auc", "pr_auc")


def model_params(pipeline: Pipeline, columns: list[str]) -> dict:
    """Hiperparámetros del clasificador (el último paso del pipeline) y tamaño del set de features."""
    estimator = pipeline.steps[-1][1]
    params = {
        name: value
        for name, value in estimator.get_params().items()
        if isinstance(value, (bool, int, float, str)) or value is None
    }
    return {"estimator": type(estimator).__name__, "n_features": len(columns), **params}


def flat_metrics(metrics: dict, prefix: str) -> dict[str, float]:
    """Las métricas numéricas de classification_metrics, con la matriz de confusión aplanada."""
    flat = {}
    for name, value in metrics.items():
        if isinstance(value, dict):
            flat.update({f"{prefix}{name}_{key}": float(v) for key, v in value.items()})
        elif isinstance(value, (int, float)):
            flat[f"{prefix}{name}"] = float(value)
    return flat


def candidate_metrics(report: dict, name: str) -> dict[str, float]:
    """Métricas de un candidato: medias de la validación cruzada, validación y test."""
    cv = report["cross_validation"][name]
    metrics = {f"cv_{metric}": float(cv[metric]) for metric in CV_FOLD_METRICS}
    metrics.update(flat_metrics(report["validation"][name], "val_"))
    metrics.update(flat_metrics(report["test"][name], "test_"))
    return metrics


def summary_params(report: dict) -> dict:
    """Lo elegido y cómo se eligió, para la corrida padre."""
    params = {
        "anchor": report["anchor"],
        "alert_volume": report["alert_volume"],
        "selected_model": report["selected_model"],
        "selection_metric": report["selection"]["metric"],
        "selection_tolerance": report["selection"]["tolerance"],
        "cv_folds": report["selection"]["cv_folds"],
    }
    for cohort, info in report["cohorts"].items():
        params[f"{cohort}_period"] = " a ".join(info["period"])
        params[f"{cohort}_n"] = info["n"]
    return params


def log_comparison(
    report: dict,
    candidates: dict[str, tuple[Pipeline, list[str]]],
    artifacts: list[Path],
    experiment: str = DEFAULT_EXPERIMENT,
) -> str:
    """Registra la comparación en MLflow y devuelve el id de la corrida padre."""
    import mlflow  # import diferido: MLflow solo hace falta con --mlflow

    mlflow.set_experiment(experiment)
    best = report["selected_model"]
    with mlflow.start_run(run_name=f"comparacion_{report['anchor']}") as parent:
        mlflow.log_params(summary_params(report))
        mlflow.log_metric("selected_threshold", report["selected_threshold"])
        mlflow.log_metrics({f"selected_{k}": v for k, v in candidate_metrics(report, best).items()})
        for name, (pipeline, columns) in candidates.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tag("seleccionado", str(name == best).lower())
                mlflow.log_params(model_params(pipeline, columns))
                mlflow.log_metrics(candidate_metrics(report, name))
                for metric in CV_FOLD_METRICS:
                    for step, value in enumerate(report["cross_validation"][name][f"{metric}_folds"]):
                        mlflow.log_metric(f"cv_{metric}_tramo", value, step=step)
        for path in artifacts:
            mlflow.log_artifact(str(path))
        return parent.info.run_id

"""Entrenamiento y comparación de modelos (parte 2).

Uso (desde la raíz del repo, con los 9 CSV de Kaggle en data/raw):
    python -m src.training
    python -m src.training --anchor compra

Compara, con las tres cohortes por fecha de src/validation.py::temporal_three_way_split
(las mismas que src/umbral.py):
  - logistica_baseline: el baseline tal cual (features de src/features.py), referencia
  - logistica_seleccion: regresión logística con la selección de features
  - arboles_seleccion: HistGradientBoosting (árboles con boosting) con la selección
  - arboles_completa: HistGradientBoosting con todas las features de feature_engineering

Selección de features (ver docs/decisiones.md, decisión 10): las del baseline sin
purchase_month, más la distancia cliente-vendedor. El mes empeora la validación
porque la tasa de retraso cambia mucho entre meses (deriva temporal) y el modelo
aprende meses que no se repiten.

El modelo se elige por PR-AUC medio en una validación cruzada temporal (4 tramos)
sobre train + validación; ante un empate (menos de 0,01) gana el más simple. El
umbral de alerta se fija en validación (5 % de pedidos con más riesgo). Recién
con el modelo y los umbrales elegidos se calculan las métricas de test: se informan
para los 4 candidatos, como diagnóstico, pero no cambian la elección.
Escribe reports/comparacion_modelos.json y guarda el modelo en models/.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.baseline import make_model as make_baseline_model
from src.data_loading import RAW_DIR, load_raw_tables
from src.feature_engineering import build_model_dataset, load_geolocation, model_feature_lists
from src.features import (
    ANCHORS,
    CATEGORICAL_FEATURES,
    DEFAULT_ANCHOR,
    PREDICTION_TIME_COL,
    feature_columns,
    numeric_features,
)
from src.target import TARGET_COL
from src.validation import classification_metrics, temporal_three_way_split

RANDOM_STATE = 42
MAX_ITER_LOGISTIC = 2000
ALERT_VOLUME = 0.05  # se compara el 5 % de pedidos con más riesgo (nivel "alto" de la decisión 9)
SELECTION_METRIC = "pr_auc"
SELECTION_TOLERANCE = 0.01
CV_FOLDS = 4
# Candidatos a modelo principal, del más simple al más complejo. El baseline queda como referencia.
MAIN_CANDIDATES = ("logistica_seleccion", "arboles_seleccion", "arboles_completa")
DROPPED_FEATURES = ["purchase_month"]
SELECTED_EXTRA_FEATURES = ["max_distance_km", "n_national_holidays_next_7d"]
MODELS_DIR = Path("models")


def make_logistic(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="unknown"),
                    OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                ),
                categorical,
            ),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=MAX_ITER_LOGISTIC, random_state=RANDOM_STATE
    )
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def make_hist_gradient_boosting(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Árboles con boosting. Los nulos se manejan nativamente; las categóricas van codificadas."""
    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="unknown"),
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
                categorical,
            ),
        ]
    )
    is_categorical = [False] * len(numeric) + [True] * len(categorical)
    # Árboles poco profundos y muy regularizados: con la deriva temporal del dataset,
    # los árboles más complejos se sobreajustan a train y rinden peor en validación.
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_depth=3,
        min_samples_leaf=500,
        l2_regularization=10.0,
        class_weight="balanced",
        categorical_features=is_categorical,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def selected_feature_lists(anchor: str = DEFAULT_ANCHOR) -> tuple[list[str], list[str]]:
    """(numéricas, categóricas) de la selección: baseline sin el mes, más distancia y feriados próximos."""
    numeric_all, _ = model_feature_lists(anchor)
    numeric = [c for c in numeric_features(anchor) if c not in DROPPED_FEATURES]
    numeric += [c for c in SELECTED_EXTRA_FEATURES if c in numeric_all]
    return numeric, list(CATEGORICAL_FEATURES)


def candidate_models(anchor: str = DEFAULT_ANCHOR) -> dict[str, tuple[Pipeline, list[str]]]:
    """Nombre -> (pipeline sin entrenar, columnas que usa)."""
    sel_num, sel_cat = selected_feature_lists(anchor)
    all_num, all_cat = model_feature_lists(anchor)
    all_num = [c for c in all_num if c not in DROPPED_FEATURES]
    return {
        "logistica_baseline": (make_baseline_model(anchor), feature_columns(anchor)),
        "logistica_seleccion": (make_logistic(sel_num, sel_cat), sel_num + sel_cat),
        "arboles_seleccion": (make_hist_gradient_boosting(sel_num, sel_cat), sel_num + sel_cat),
        "arboles_completa": (make_hist_gradient_boosting(all_num, all_cat), all_num + all_cat),
    }


def alert_threshold(scores, volume: float = ALERT_VOLUME) -> float:
    """Puntaje a partir del cual queda el `volume` de pedidos con más riesgo."""
    return float(pd.Series(scores).quantile(1 - volume))


def _period(part: pd.DataFrame) -> list[str]:
    return [str(part[PREDICTION_TIME_COL].min().date()), str(part[PREDICTION_TIME_COL].max().date())]


def temporal_cv_folds(table: pd.DataFrame, n_folds: int = CV_FOLDS) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Validación cruzada temporal con ventana creciente.

    La segunda mitad del período se parte en n_folds tramos consecutivos; cada
    tramo se evalúa con un modelo entrenado solo con lo anterior a ese tramo.
    """
    times = table[PREDICTION_TIME_COL]
    quantiles = [0.5 + 0.5 * i / n_folds for i in range(n_folds + 1)]
    edges = [times.quantile(q) for q in quantiles[:-1]] + [times.max() + pd.Timedelta(days=1)]
    return [(table[times < start], table[(times >= start) & (times < stop)]) for start, stop in zip(edges, edges[1:])]


def cross_validate(pipeline: Pipeline, columns: list[str], folds) -> dict:
    roc, pr = [], []
    for fit_part, eval_part in folds:
        model = clone(pipeline).fit(fit_part[columns], fit_part[TARGET_COL])
        scores = model.predict_proba(eval_part[columns])[:, 1]
        roc.append(roc_auc_score(eval_part[TARGET_COL], scores))
        pr.append(average_precision_score(eval_part[TARGET_COL], scores))
    return {
        "roc_auc_folds": [round(v, 4) for v in roc],
        "pr_auc_folds": [round(v, 4) for v in pr],
        "roc_auc": float(np.mean(roc)),
        "pr_auc": float(np.mean(pr)),
    }


def select_model(cv_results: dict, tolerance: float = SELECTION_TOLERANCE) -> str:
    """El mejor PR-AUC medio de la validación cruzada entre los candidatos a modelo principal.

    Si otro candidato más simple (anterior en MAIN_CANDIDATES) queda a menos de
    `tolerance`, se elige el más simple: la diferencia no es concluyente y el
    modelo simple tolera mejor la deriva temporal.
    """
    best_score = max(cv_results[name][SELECTION_METRIC] for name in MAIN_CANDIDATES)
    return next(name for name in MAIN_CANDIDATES if cv_results[name][SELECTION_METRIC] >= best_score - tolerance)


def compare_models(dataset: pd.DataFrame, anchor: str = DEFAULT_ANCHOR) -> tuple[dict, dict]:
    """Compara los candidatos y evalúa en test.

    1. Validación cruzada temporal sobre train + validación (elige el modelo).
    2. Cada modelo se entrena en train y se mide en validación (fija el umbral de alerta).
    3. Test: con todo ya elegido se miden los 4 candidatos, solo como diagnóstico.

    Devuelve (reporte, modelos entrenados en train).
    """
    train, validation, test = temporal_three_way_split(dataset)
    folds = temporal_cv_folds(pd.concat([train, validation]))
    report = {
        "anchor": anchor,
        "alert_volume": ALERT_VOLUME,
        "selection": {
            "metric": SELECTION_METRIC,
            "tolerance": SELECTION_TOLERANCE,
            "cv_folds": len(folds),
            "candidates": list(MAIN_CANDIDATES),
        },
        "cohorts": {
            name: {"period": _period(part), "n": int(len(part)), "late_rate": float(part[TARGET_COL].mean())}
            for name, part in [("train", train), ("validation", validation), ("test", test)]
        },
        "cross_validation": {},
        "validation": {},
        "test": {},
    }
    fitted, thresholds = {}, {}
    for name, (pipeline, columns) in candidate_models(anchor).items():
        report["cross_validation"][name] = cross_validate(pipeline, columns, folds)
        model = clone(pipeline).fit(train[columns], train[TARGET_COL])
        scores = model.predict_proba(validation[columns])[:, 1]
        thresholds[name] = alert_threshold(scores)
        report["validation"][name] = classification_metrics(validation[TARGET_COL], scores, thresholds[name])
        fitted[name] = (model, columns)

    best = select_model(report["cross_validation"])
    report["selected_model"] = best
    report["selected_threshold"] = thresholds[best]
    for name, (model, columns) in fitted.items():
        scores = model.predict_proba(test[columns])[:, 1]
        report["test"][name] = classification_metrics(test[TARGET_COL], scores, thresholds[name])
    return report, fitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--anchor", choices=list(ANCHORS), default=DEFAULT_ANCHOR)
    parser.add_argument("--output", type=Path, default=Path("reports/comparacion_modelos.json"))
    args = parser.parse_args()

    tables = {**load_raw_tables(args.raw_dir), "geolocation": load_geolocation(args.raw_dir)}
    dataset = build_model_dataset(tables, args.anchor)
    report, fitted = compare_models(dataset, args.anchor)

    best = report["selected_model"]
    model, columns = fitted[best]
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "features": columns, "anchor": args.anchor, "threshold": report["selected_threshold"]},
        MODELS_DIR / "modelo_principal.joblib",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

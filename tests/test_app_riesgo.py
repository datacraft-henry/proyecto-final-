import numpy as np
import pandas as pd
import pytest

from app.riesgo import (
    FEATURE_LABELS,
    SEMAFORO,
    format_feature_value,
    load_artifact,
    orders_with_risk,
    risk_factors,
    summary_by_level,
)
from src.calibration import PlattCalibrator
from src.features import PREDICTION_TIME_COL
from src.target import TARGET_COL
from src.training import make_logistic, selected_feature_lists


class _ScoreIsColumn:
    """Modelo de mentira: el puntaje es la columna 'score'."""

    def predict_proba(self, X):
        score = X["score"].to_numpy()
        return np.column_stack([1 - score, score])


@pytest.fixture
def artifact():
    scores = np.linspace(0.05, 0.95, 200)
    y = (scores > 0.7).astype(int)
    return {
        "model": _ScoreIsColumn(),
        "features": ["score"],
        "calibrator": PlattCalibrator().fit(scores, y),
        "cuts": {"alto": 0.6, "medio": 0.3},
        "anchor": "aprobacion",
    }


@pytest.fixture
def orders():
    return pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d"],
            PREDICTION_TIME_COL: pd.to_datetime(["2018-06-01"] * 4),
            "customer_state": ["SP", "RJ", "MG", "SP"],
            "score": [0.1, 0.9, 0.5, 0.75],
            TARGET_COL: [0, 1, 0, 0],
        }
    )


def test_orders_are_sorted_by_risk_with_traffic_light(artifact, orders):
    scored = orders_with_risk(artifact, orders)
    assert scored["prob_retraso"].is_monotonic_decreasing
    assert scored["order_id"].iloc[0] == "b"
    assert (scored["semaforo"] == scored["nivel_riesgo"].map(SEMAFORO)).all()
    assert "score" not in scored.columns  # solo columnas para mostrar
    assert TARGET_COL in scored.columns


def test_summary_has_every_level_even_if_empty(artifact, orders):
    scored = orders_with_risk(artifact, orders.iloc[[0]])
    summary = summary_by_level(scored)
    assert list(summary.index) == ["alto", "medio", "bajo"]
    assert summary["pedidos"].sum() == 1
    assert summary["pct_pedidos"].sum() == pytest.approx(1.0)


def test_summary_reports_real_late_rate_when_target_is_present(artifact, orders):
    summary = summary_by_level(orders_with_risk(artifact, orders))
    assert summary.loc["alto", "tasa_retraso_real"] == pytest.approx(0.5)


def test_missing_artifact_explains_how_to_generate_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.umbral"):
        load_artifact(tmp_path / "no_existe.joblib")


def test_summary_reports_share_of_late_orders_by_level(artifact, orders):
    summary = summary_by_level(orders_with_risk(artifact, orders))
    assert summary["pct_retrasos"].sum() == pytest.approx(1.0)
    assert summary.loc["alto", "pct_retrasos"] == pytest.approx(1.0)


def test_feature_values_are_written_for_operations():
    assert format_feature_value("max_distance_km", 2133.7) == "2.134 km"
    assert format_feature_value("total_price", 1050) == "R$ 1.050,00"
    assert format_feature_value("purchase_dayofweek", 0) == "lunes"
    assert format_feature_value("seller_other_state", True) == "sí"
    assert format_feature_value("promised_days", float("nan")) == "sin dato"
    assert format_feature_value("n_national_holidays_next_7d", 0.0) == "ninguno"
    assert format_feature_value("n_national_holidays_next_7d", 1.0) == "1 feriado"
    assert format_feature_value("n_national_holidays_next_7d", 2) == "2 feriados"


def test_every_model_feature_has_an_operations_label():
    numeric, categorical = selected_feature_lists()
    assert set(numeric + categorical) <= set(FEATURE_LABELS)


def test_risk_factors_point_to_what_moves_the_order_away_from_average():
    rng = np.random.default_rng(0)
    n = 400
    data = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(n)],
            "distancia": rng.uniform(0, 3000, n),
            "ruido": rng.normal(0, 1, n),
            "estado": rng.choice(["SP", "RJ"], n),
        }
    )
    data[TARGET_COL] = (data["distancia"] + rng.normal(0, 300, n) > 2000).astype(int)
    model = make_logistic(["distancia", "ruido"], ["estado"]).fit(data[["distancia", "ruido", "estado"]], data[TARGET_COL])
    artifact = {"model": model, "features": ["distancia", "ruido", "estado"]}
    far = data.loc[data["distancia"].idxmax(), "order_id"]

    factors = risk_factors(artifact, data, far)

    assert set(factors["variable"]) == {"distancia", "ruido", "estado"}  # categorías sumadas por variable
    assert factors["variable"].iloc[0] == "distancia"
    assert factors["aporte"].iloc[0] > 0

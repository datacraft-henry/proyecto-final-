from typing import get_args

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import EstadoBrasil, Pedido, TipoPago, app
from src.calibration import PlattCalibrator, risk_level_cuts
from src.features import LEAKY_COLUMNS
from src.training import make_logistic, selected_feature_lists
from src.umbral import predict_risk

NUMERIC, CATEGORICAL = selected_feature_lists("aprobacion")


def _synthetic_orders(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = _pedido()
    data = {c: base[c] * rng.uniform(0.5, 1.5, n) for c in NUMERIC}  # alrededor de un pedido típico
    data["promised_days"] = rng.uniform(2, 40, n)
    data["customer_state"] = rng.choice(["SP", "RJ", "MA"], n)
    data["main_payment_type"] = rng.choice(["credit_card", "boleto"], n)
    data["seller_other_state"] = rng.choice(["True", "False"], n)
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def artifact():
    """Artefacto con la misma forma que el de src/umbral.py, entrenado con datos sintéticos."""
    orders = _synthetic_orders(400)
    noise = np.random.default_rng(1).normal(0, 8, len(orders))
    y = (orders["promised_days"] + noise < 12).astype(int)  # plazo corto -> más retraso, con ruido
    model = make_logistic(NUMERIC, CATEGORICAL).fit(orders[NUMERIC + CATEGORICAL], y)
    scores = model.predict_proba(orders[NUMERIC + CATEGORICAL])[:, 1]
    calibrator = PlattCalibrator().fit(scores, y)
    return {
        "model": model,
        "features": NUMERIC + CATEGORICAL,
        "calibrator": calibrator,
        "cuts": risk_level_cuts(calibrator.transform(scores)),
        "anchor": "aprobacion",
        "entrenado_con": "sintetico",
    }


@pytest.fixture
def client(artifact):
    with TestClient(app) as c:
        app.state.artifact = artifact  # después del arranque, para no depender de models/
        yield c


def _pedido(**overrides) -> dict:
    base = {
        "order_id": "o1",
        "purchase_dayofweek": 2,
        "purchase_hour": 14,
        "promised_days": 20.0,
        "approval_hours": 0.3,
        "n_items": 1,
        "n_sellers": 1,
        "total_price": 120.0,
        "total_freight": 18.0,
        "freight_ratio": 0.15,
        "total_weight_g": 800.0,
        "total_volume_cm3": 4000.0,
        "max_installments": 3,
        "max_distance_km": 430.0,
        "n_national_holidays_next_7d": 0,
        "customer_state": "SP",
        "main_payment_type": "credit_card",
        "seller_other_state": "True",
    }
    return {**base, **overrides}


def test_schema_covers_every_model_feature():
    """Si el modelo suma una variable, este test avisa que la API también tiene que recibirla."""
    assert set(NUMERIC + CATEGORICAL) <= set(Pedido.model_fields)


def test_schema_rejects_leaky_columns(client):
    for column in LEAKY_COLUMNS:
        response = client.post("/predict", json={"pedidos": [_pedido(**{column: "2018-01-01"})]})
        assert response.status_code == 422, column


def test_predict_matches_predict_risk(client, artifact):
    pedidos = [_pedido(order_id="corto", promised_days=3.0), _pedido(order_id="largo", promised_days=40.0)]
    response = client.post("/predict", json={"pedidos": pedidos})
    assert response.status_code == 200
    preds = response.json()["predicciones"]
    assert [p["order_id"] for p in preds] == ["corto", "largo"]

    expected = predict_risk(artifact, pd.DataFrame(pedidos))
    assert [p["prob_retraso"] for p in preds] == pytest.approx(expected["prob_retraso"].tolist(), abs=1e-4)
    assert [p["nivel_riesgo"] for p in preds] == expected["nivel_riesgo"].tolist()
    assert preds[0]["prob_retraso"] > preds[1]["prob_retraso"]


def test_missing_values_are_imputed(client):
    pedido = {"order_id": "vacio"}  # todo lo demás falta: el pipeline imputa
    response = client.post("/predict", json={"pedidos": [pedido]})
    assert response.status_code == 200
    assert response.json()["predicciones"][0]["nivel_riesgo"] in {"alto", "medio", "bajo"}


def test_invalid_values_are_rejected(client):
    assert client.post("/predict", json={"pedidos": [_pedido(purchase_hour=25)]}).status_code == 422
    assert client.post("/predict", json={"pedidos": []}).status_code == 422


def test_unknown_categories_are_rejected(client):
    """Un estado o un tipo de pago que no existen en Olist no llegan al modelo como "desconocidos"."""
    for campo, valor in [("customer_state", "st"), ("customer_state", "XX"), ("main_payment_type", "string")]:
        response = client.post("/predict", json={"pedidos": [_pedido(**{campo: valor})]})
        assert response.status_code == 422, (campo, valor)
        assert response.json()["detail"][0]["loc"][-1] == campo


def test_category_lists_cover_olist():
    """Los 27 estados de Brasil y los tipos de pago de Olist tienen que estar permitidos."""
    assert len(get_args(EstadoBrasil)) == 27
    assert {"credit_card", "boleto", "voucher", "debit_card"} <= set(get_args(TipoPago))


def test_without_model_predict_returns_503(client):
    app.state.artifact = None
    assert client.get("/health").json()["status"] == "sin_modelo"
    assert client.post("/predict", json={"pedidos": [_pedido()]}).status_code == 503


def test_health_reports_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["features"] == NUMERIC + CATEGORICAL
    assert set(body["cortes_riesgo"]) == {"alto", "medio"}

def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"

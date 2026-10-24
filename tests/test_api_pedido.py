import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.umbral import predict_risk
from tests.test_api import _synthetic_orders, artifact  # noqa: F401  (artifact es un fixture)


@pytest.fixture
def dataset():
    """Pedidos ya armados, como los deja build_model_dataset, indexados por order_id."""
    orders = _synthetic_orders(5, seed=3)
    orders.index = [f"pedido-{i}" for i in range(5)]
    orders.index.name = "order_id"
    return orders


@pytest.fixture
def client(artifact, dataset):  # noqa: F811
    with TestClient(app) as c:
        app.state.artifact = artifact
        app.state.dataset = dataset  # después del arranque, para no leer data/raw
        yield c


def test_predict_por_order_id_da_lo_mismo_que_predict_risk(client, artifact, dataset):  # noqa: F811
    response = client.post("/predict/pedido", json={"order_ids": ["pedido-2", "pedido-0"]})
    assert response.status_code == 200
    body = response.json()
    assert [p["order_id"] for p in body["predicciones"]] == ["pedido-2", "pedido-0"]
    esperado = predict_risk(artifact, dataset.loc[["pedido-2", "pedido-0"]])
    assert [p["prob_retraso"] for p in body["predicciones"]] == pytest.approx(esperado["prob_retraso"].tolist(), abs=1e-4)
    assert [p["nivel_riesgo"] for p in body["predicciones"]] == esperado["nivel_riesgo"].tolist()
    assert body["no_encontrados"] == []


def test_order_id_inexistente_va_a_no_encontrados(client):
    body = client.post("/predict/pedido", json={"order_ids": ["pedido-1", "no-existe", "pedido-1"]}).json()
    assert [p["order_id"] for p in body["predicciones"]] == ["pedido-1"]  # sin repetidos
    assert body["no_encontrados"] == ["no-existe"]


def test_pedido_por_order_id_valida_la_entrada(client):
    assert client.post("/predict/pedido", json={"order_ids": []}).status_code == 422
    assert client.post("/predict/pedido", json={"order_id": "pedido-1"}).status_code == 422


def test_sin_datos_responde_503(client, monkeypatch, tmp_path):
    app.state.dataset = None
    monkeypatch.setenv("DATOS_PATH", str(tmp_path))  # carpeta vacía: no hay CSV
    response = client.post("/predict/pedido", json={"order_ids": ["pedido-1"]})
    assert response.status_code == 503
    assert "No hay datos" in response.json()["detail"]


def test_sin_modelo_responde_503(client):
    app.state.artifact = None
    assert client.post("/predict/pedido", json={"order_ids": ["pedido-1"]}).status_code == 503


def test_el_dataset_se_arma_una_sola_vez(client, dataset):
    client.post("/predict/pedido", json={"order_ids": ["pedido-1"]})
    assert app.state.dataset is dataset
    assert isinstance(app.state.dataset, pd.DataFrame)

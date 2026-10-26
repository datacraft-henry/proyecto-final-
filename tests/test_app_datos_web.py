import json

import numpy as np
import pandas as pd
import pytest

from app import datos_web
from app.datos_web import (
    build_summary,
    contributions_of,
    load_recommender,
    load_scored_orders,
    ranked_sellers,
    scored_orders_snapshot,
    sellers_ranking,
    web_mode,
)
from app.riesgo import factors_table, orders_with_risk, risk_factors
from src.calibration import PlattCalibrator
from src.features import PREDICTION_TIME_COL
from src.recomendador import CRITERIOS, puntaje_vendedores, recomendar
from src.target import TARGET_COL
from src.training import make_logistic
from tests.test_recomendador import CORTE_TEST, _pedidos_con_riesgo

FEATURES = ["distancia", "ruido", "estado"]


@pytest.fixture
def orders():
    rng = np.random.default_rng(0)
    n = 300
    data = pd.DataFrame(
        {
            "order_id": [f"p{i}" for i in range(n)],
            PREDICTION_TIME_COL: pd.date_range("2018-06-01", periods=n, freq="h"),
            "customer_state": rng.choice(["SP", "RJ"], n),
            "promised_days": rng.integers(5, 30, n),
            "max_distance_km": rng.uniform(10, 3000, n),
            "total_price": rng.uniform(20, 500, n),
            "total_freight": rng.uniform(5, 60, n),
            "n_items": rng.integers(1, 4, n),
            "distancia": rng.uniform(10, 3000, n),
            "ruido": rng.normal(0, 1, n),
            "estado": rng.choice(["SP", "RJ", "BA"], n),
        }
    )
    data[TARGET_COL] = (data["distancia"] + rng.normal(0, 300, n) > 2000).astype(int)
    return data


@pytest.fixture
def artifact(orders):
    model = make_logistic(["distancia", "ruido"], ["estado"]).fit(orders[FEATURES], orders[TARGET_COL])
    scores = model.predict_proba(orders[FEATURES])[:, 1]
    return {
        "model": model,
        "features": FEATURES,
        "calibrator": PlattCalibrator().fit(scores, orders[TARGET_COL]),
        "cuts": {"alto": 0.6, "medio": 0.3},
        "anchor": "aprobacion",
        "entrenado_con": "train + validacion",
    }


def test_web_mode_only_without_model_and_with_files(tmp_path, monkeypatch):
    monkeypatch.delenv(datos_web.ENV_FLAG, raising=False)
    model, summary = tmp_path / "modelo.joblib", tmp_path / "resumen.json"
    assert not web_mode(model, summary)  # sin nada: el error de siempre, que pide generar el modelo
    summary.write_text("{}", encoding="utf-8")
    assert web_mode(model, summary)  # como en la nube: archivos sí, modelo no
    model.write_bytes(b"x")
    assert not web_mode(model, summary)  # en la PC con el modelo manda el modelo
    monkeypatch.setenv(datos_web.ENV_FLAG, "1")
    assert web_mode(model, summary)  # salvo que se pida probar el modo web


def test_snapshot_keeps_the_app_order_and_risk(artifact, orders, tmp_path):
    path = tmp_path / "pedidos.parquet"
    scored_orders_snapshot(artifact, orders).to_parquet(path, index=False)

    snapshot = load_scored_orders(path)
    expected = orders_with_risk(artifact, orders)

    pd.testing.assert_frame_equal(snapshot[expected.columns], expected, check_dtype=False)


def test_snapshot_factors_match_the_model(artifact, orders, tmp_path):
    path = tmp_path / "pedidos.parquet"
    scored_orders_snapshot(artifact, orders).to_parquet(path, index=False)
    snapshot = load_scored_orders(path)

    for order_id in ["p0", "p17", orders.loc[orders["distancia"].idxmax(), "order_id"]]:
        from_model = risk_factors(artifact, orders, order_id)
        from_file = factors_table(*contributions_of(snapshot, order_id))
        pd.testing.assert_frame_equal(from_file, from_model, check_exact=False, atol=1e-9)


def _recommender():
    orders = pd.concat(
        [
            _pedidos_con_riesgo([("a", 0.30, 0), ("b", 0.10, 0), ("c", 0.20, 0)]),
            _pedidos_con_riesgo([("d", 0.01, 0), ("b", 0.10, 0)], estado="SP"),
        ],
        ignore_index=True,
    )
    orders["order_id"] = [f"o{i}" for i in range(len(orders))]
    sellers = puntaje_vendedores(orders, CORTE_TEST)
    combos = pd.DataFrame(
        {"product_category_name": ["cama_mesa_banho"] * 2, "customer_state": ["RJ", "SP"], "vendedores": [3, 2]}
    )
    metrics = pd.DataFrame(
        {"precision": [0.7, 0.69, 0.69, 0.66], "recall": [0.28] * 4, "retraso": [0.026, 0.028, 0.0275, 0.031]},
        index=CRITERIOS,
    )
    return {
        "orders": orders,
        "sellers": sellers,
        "cut": CORTE_TEST,
        "test_end": pd.Timestamp("2018-08-29"),
        "combos": combos,
        "metrics": metrics,
        "n_combos": 2,
        "vs_random": (-0.005, -0.008, -0.003),
        "vs_history": (-0.002, -0.006, 0.001),
        "vs_popularity": (-0.002, -0.005, 0.002),
        "dispatch": {"pct_retrasos_con_despacho_tarde": 0.277},
        "seller_state": pd.Series({"a": "SP", "b": "RJ", "c": "MG", "d": "SP"}),
        "names": {"cama_mesa_banho": "bed_bath_table", "otra": "other"},
        "cuts": {"alto": 0.6, "medio": 0.3},
    }


def test_ranking_from_files_matches_recomendar(artifact, orders, tmp_path):
    rec = _recommender()
    ranking_path, summary_path = tmp_path / "ranking.parquet", tmp_path / "resumen.json"
    sellers_ranking(rec["orders"], rec["sellers"], rec["cut"], rec["combos"], rec["seller_state"]).to_parquet(
        ranking_path, index=False
    )
    summary_path.write_text(json.dumps(build_summary(artifact, rec, orders)), encoding="utf-8")

    loaded = load_recommender(ranking_path, summary_path)

    for state in ["RJ", "SP"]:
        expected = recomendar(rec["orders"], rec["sellers"], rec["cut"], "cama_mesa_banho", state, k=len(rec["sellers"]))
        got = ranked_sellers(loaded["ranking"], "cama_mesa_banho", state)
        pd.testing.assert_frame_equal(got, expected, check_dtype=False, check_names=False)
    assert loaded["cut"] == rec["cut"]
    assert loaded["vs_random"] == rec["vs_random"]
    assert loaded["seller_state"]["b"] == "RJ"
    assert loaded["names"] == {"cama_mesa_banho": "bed_bath_table"}  # solo categorías que se ofrecen
    pd.testing.assert_frame_equal(loaded["metrics"], rec["metrics"])
    pd.testing.assert_frame_equal(loaded["combos"], rec["combos"])


def test_missing_files_explain_how_to_generate_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m app.datos_web"):
        load_scored_orders(tmp_path / "no.parquet")
    with pytest.raises(FileNotFoundError, match="python -m app.datos_web"):
        load_recommender(tmp_path / "no.parquet", tmp_path / "no.json")

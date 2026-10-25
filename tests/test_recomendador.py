import pandas as pd
import pytest

from src.recomendador import (
    CORTE,
    pedidos_un_vendedor,
    evaluar_recomendacion,
    puntaje_vendedores,
    recomendar,
    resumen_evaluacion,
    recompra_tras_primer_pedido,
    preparar,
    retraso_que_empieza_en_el_vendedor,
    riesgo_vendedores,
)
from src.target import TARGET_COL


def _tablas(spec):
    """spec: lista de (vendedor, fecha de compra, llegó tarde, despachó tarde). Un ítem por pedido."""
    orders, items = [], []
    for i, (vendedor, fecha, tarde, despacho_tarde) in enumerate(spec):
        compra = pd.Timestamp(fecha)
        estimada = compra + pd.Timedelta(days=10)
        entrega = estimada + pd.Timedelta(days=3 if tarde else -2)
        limite = compra + pd.Timedelta(days=2)
        orders.append({
            "order_id": f"o{i}", "customer_id": f"c{i}", "order_status": "delivered",
            "order_purchase_timestamp": compra, "order_approved_at": compra + pd.Timedelta(hours=1),
            "order_delivered_carrier_date": limite + pd.Timedelta(days=1 if despacho_tarde else -1),
            "order_delivered_customer_date": entrega, "order_estimated_delivery_date": estimada,
        })
        items.append({"order_id": f"o{i}", "order_item_id": 1, "product_id": "p1", "seller_id": vendedor,
                      "shipping_limit_date": limite, "price": 50.0, "freight_value": 5.0})
    vendedores = sorted({s[0] for s in spec})
    return {
        "orders": pd.DataFrame(orders),
        "items": pd.DataFrame(items),
        "customers": pd.DataFrame({"customer_id": [f"c{i}" for i in range(len(spec))], "customer_state": "SP"}),
        "products": pd.DataFrame({"product_id": ["p1"], "product_category_name": ["cama_mesa_banho"]}),
        "sellers": pd.DataFrame({"seller_id": vendedores, "seller_state": "SP"}),
    }


def _spec(antes_bueno=0, antes_malo=6, despues_bueno=1, despues_malo=5, n=12):
    """Vendedor 'bueno' y 'malo' con n pedidos antes y después del corte, y uno nuevo con 2 pedidos."""
    spec = []
    for vendedor, tardes_antes, tardes_despues in [("bueno", antes_bueno, despues_bueno), ("malo", antes_malo, despues_malo)]:
        spec += [(vendedor, "2017-03-01", k < tardes_antes, False) for k in range(n)]
        spec += [(vendedor, "2018-03-01", k < tardes_despues, k < tardes_despues) for k in range(n)]
    spec += [("nuevo", "2018-04-01", False, False)] * 2
    return spec


def test_el_orden_solo_usa_pedidos_anteriores_al_corte():
    """Anti-fuga: cambiar lo que pasó después del corte no puede mover el puntaje."""
    base = riesgo_vendedores(pedidos_un_vendedor(_tablas(_spec(despues_bueno=0, despues_malo=0))))
    otro = riesgo_vendedores(pedidos_un_vendedor(_tablas(_spec(despues_bueno=12, despues_malo=12))))
    pd.testing.assert_series_equal(base["puntaje"], otro["puntaje"])


def test_sin_historial_suficiente_no_se_ordena():
    vend = riesgo_vendedores(pedidos_un_vendedor(_tablas(_spec())))
    assert set(vend.index) == {"bueno", "malo"}  # 'nuevo' no tiene pedidos antes del corte
    assert vend.loc["bueno", "puntaje"] < vend.loc["malo", "puntaje"]


def test_el_suavizado_no_premia_un_cero_con_pocos_pedidos():
    """0 % en 10 pedidos queda detrás de 2 % en 200: el primero es casi pura suerte."""
    spec = [("poco", "2017-02-01", False, False)] * 10
    spec += [("mucho", "2017-02-01", k < 4, False) for k in range(200)]
    spec += [("medio", "2017-02-01", k < 30, False) for k in range(200)]  # sube la media a ~8 %
    vend = riesgo_vendedores(pedidos_un_vendedor(_tablas(spec)))
    assert vend.loc["poco", "retraso_hist"] == 0
    assert vend.loc["mucho", "puntaje"] < vend.loc["poco", "puntaje"]


def test_los_pedidos_de_varios_vendedores_quedan_afuera():
    tablas = _tablas(_spec())
    extra = tablas["items"].iloc[[0]].assign(order_item_id=2, seller_id="malo")
    tablas["items"] = pd.concat([tablas["items"], extra], ignore_index=True)
    pedidos = pedidos_un_vendedor(tablas)
    assert "o0" not in set(pedidos["order_id"])
    assert pedidos["order_id"].is_unique


def test_preparar_devuelve_tercios_y_resumen():
    # tres vendedores con 12 pedidos después del corte: cada tercio es uno de ellos
    spec = _spec() + [("regular", "2017-03-01", k < 3, False) for k in range(12)]
    spec += [("regular", "2018-03-01", k < 3, False) for k in range(12)]
    candidatos, tercios, resumen = preparar(_tablas(spec))
    assert list(tercios.index) == ["Menor riesgo", "Medio", "Mayor riesgo"]
    assert tercios["retraso"].tolist() == pytest.approx([1 / 12, 3 / 12, 5 / 12])
    # 38 pedidos después del corte; 36 son de vendedores con historial
    assert resumen["cobertura"] == pytest.approx(36 / 38)
    assert set(candidatos["seller_id"]) == {"bueno", "regular", "malo"}


def test_retraso_que_empieza_en_el_vendedor():
    # 4 pedidos: 2 tarde con despacho tarde, 1 tarde con despacho a tiempo, 1 a tiempo
    spec = [("a", "2017-05-01", True, True), ("a", "2017-05-01", True, True),
            ("a", "2017-05-01", True, False), ("a", "2017-05-01", False, False)]
    r = retraso_que_empieza_en_el_vendedor(_tablas(spec))
    assert r["pct_retrasos_con_despacho_tarde"] == pytest.approx(2 / 3)
    assert r["retraso_si_despacho_tarde"] == pytest.approx(1.0)
    assert r["retraso_si_despacho_a_tiempo"] == pytest.approx(0.5)


def test_corte_por_defecto():
    assert CORTE == pd.Timestamp("2018-01-01")
    assert TARGET_COL == "pedido_retrasado"


def test_recompra_cuenta_solo_compras_posteriores_dentro_del_plazo():
    """Cliente A: primer pedido tarde, vuelve a los 30 días. B: a tiempo, vuelve a los 400. C: a tiempo, no vuelve."""
    spec = [("v", "2017-03-01", True, False), ("v", "2017-03-31", False, False),
            ("v", "2017-03-01", False, False), ("v", "2018-04-05", False, False),
            ("v", "2017-03-01", False, False)]
    tablas = _tablas(spec)
    tablas["customers"]["customer_unique_id"] = ["A", "A", "B", "B", "C"]
    r = recompra_tras_primer_pedido(tablas, dias=180, hasta="2017-06-30")
    assert r.loc[1, "clientes"] == 1 and r.loc[1, "recompra"] == pytest.approx(1.0)
    assert r.loc[0, "clientes"] == 2 and r.loc[0, "recompra"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- sistema de recomendación
CORTE_TEST = pd.Timestamp("2018-06-01")


def _pedidos_con_riesgo(vendedores, n_antes=10, n_despues=6, categoria="cama_mesa_banho", estado="RJ"):
    """Pedidos ya armados como los de preparar_clasificador.

    vendedores: lista de (seller_id, riesgo predicho antes del corte, retrasos de n_despues después).
    """
    filas = []
    for vendedor, riesgo, tardes in vendedores:
        for i in range(n_antes):
            filas.append((vendedor, "2018-03-01", riesgo, 0))
        for i in range(n_despues):
            filas.append((vendedor, "2018-07-01", 0.5, int(i < tardes)))
    pedidos = pd.DataFrame(filas, columns=["seller_id", "order_approved_at", "prob_retraso", TARGET_COL])
    pedidos["order_id"] = [f"o{i}" for i in range(len(pedidos))]
    pedidos["order_approved_at"] = pd.to_datetime(pedidos["order_approved_at"])
    pedidos["product_category_name"] = categoria
    pedidos["customer_state"] = estado
    pedidos["price"] = 50.0
    return pedidos


def test_el_riesgo_del_vendedor_solo_usa_pedidos_anteriores_al_corte():
    """Anti-fuga: cambiar riesgo o retrasos después del corte no mueve el puntaje."""
    base = _pedidos_con_riesgo([("a", 0.1, 0), ("b", 0.3, 0)])
    otro = base.copy()
    despues = otro["order_approved_at"] >= CORTE_TEST
    otro.loc[despues, "prob_retraso"] = 0.99
    otro.loc[despues, TARGET_COL] = 1
    pd.testing.assert_frame_equal(puntaje_vendedores(base, CORTE_TEST), puntaje_vendedores(otro, CORTE_TEST))


def test_el_clasificador_acierta_cuando_su_riesgo_coincide_con_el_retraso_posterior():
    """Seis vendedores: los 3 de menor riesgo predicho no se retrasan en test; los otros 3 sí."""
    pedidos = _pedidos_con_riesgo([(f"v{i}", 0.05 * (i + 1), 0 if i < 3 else 3) for i in range(6)])
    tabla = evaluar_recomendacion(pedidos, puntaje_vendedores(pedidos, CORTE_TEST), CORTE_TEST, k=3)
    resumen = resumen_evaluacion(tabla)
    assert len(tabla) == 1
    assert resumen.loc["clasificador", "precision"] == pytest.approx(1.0)
    assert resumen.loc["clasificador", "recall"] == pytest.approx(1.0)
    assert resumen.loc["azar", "precision"] == pytest.approx(0.5)
    assert resumen.loc["clasificador", "retraso"] < resumen.loc["azar", "retraso"]


def test_sin_candidatos_suficientes_no_se_evalua():
    pedidos = _pedidos_con_riesgo([(f"v{i}", 0.1, 0 if i < 2 else 3) for i in range(4)])  # 4 < 2K
    with pytest.raises(ValueError):
        evaluar_recomendacion(pedidos, puntaje_vendedores(pedidos, CORTE_TEST), CORTE_TEST, k=3)


def test_recomendar_ordena_por_riesgo_predicho_y_respeta_el_contexto():
    pedidos = pd.concat([
        _pedidos_con_riesgo([("a", 0.30, 0), ("b", 0.10, 0), ("c", 0.20, 0)]),
        _pedidos_con_riesgo([("d", 0.01, 0)], estado="SP"),  # el más seguro, pero de otro estado
    ], ignore_index=True)
    pedidos["order_id"] = [f"o{i}" for i in range(len(pedidos))]
    vend = puntaje_vendedores(pedidos, CORTE_TEST)
    lista = recomendar(pedidos, vend, CORTE_TEST, "cama_mesa_banho", "RJ", k=2)
    assert list(lista.index) == ["b", "c"]
    assert lista["ventas_aqui"].tolist() == [10, 10]


def test_sin_historial_suficiente_no_se_recomienda():
    pedidos = _pedidos_con_riesgo([("a", 0.1, 0)], n_antes=3)  # menos de MIN_PEDIDOS antes del corte
    vend = puntaje_vendedores(pedidos, CORTE_TEST)
    assert recomendar(pedidos, vend, CORTE_TEST, "cama_mesa_banho", "RJ").empty

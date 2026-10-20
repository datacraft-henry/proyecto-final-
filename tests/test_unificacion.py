import pandas as pd

from src.target import TARGET_COL
from src.unificacion import agregar_target


def _orders():
    """Mini tabla de pedidos con un caso por situación."""
    return pd.DataFrame(
        {
            "order_id": ["A", "B", "C", "D"],
            "order_status": ["delivered", "delivered", "canceled", "delivered"],
            "order_purchase_timestamp": ["2017-06-01", "2017-06-01", "2017-06-01", "2016-10-01"],
            "order_delivered_customer_date": ["2017-06-11", "2017-06-08", None, "2016-10-11"],
            "order_estimated_delivery_date": ["2017-06-10"] * 3 + ["2016-10-10"],
        }
    )


def _pedidos():
    """Tabla grande simplificada: los mismos 4 pedidos con otra columna cualquiera."""
    return pd.DataFrame({"order_id": ["A", "B", "C", "D"], "pago_total": [100, 200, 300, 400]})


def test_no_se_pierden_pedidos():
    resultado = agregar_target(_pedidos(), _orders())
    assert len(resultado) == 4


def test_valores_del_target():
    resultado = agregar_target(_pedidos(), _orders()).set_index("order_id")[TARGET_COL]
    assert resultado["A"] == 1
    assert resultado["B"] == 0
    assert pd.isna(resultado["C"])
    assert pd.isna(resultado["D"])


def test_order_id_sigue_siendo_unico():
    resultado = agregar_target(_pedidos(), _orders())
    assert resultado["order_id"].is_unique
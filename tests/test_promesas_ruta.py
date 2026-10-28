import pandas as pd
import pytest

from src.features import PREDICTION_TIME_COL
from src.promesas_ruta import (
    apply_route_promises,
    build_route_orders,
    evaluate_route_promises,
    fit_route_promises,
)


def test_fit_route_promises_calculates_p95_by_route_and_rounds_up():
    train = pd.DataFrame(
        {
            "seller_state": ["SP"] * 20 + ["MG"] * 20,
            "customer_state": ["RJ"] * 20 + ["SP"] * 20,
            "delivery_days": [10] * 19 + [20] + [5] * 20,
        }
    )

    result = fit_route_promises(train).set_index(
        ["seller_state", "customer_state"]
    )

    assert result.loc[("SP", "RJ"), "train_orders"] == 20
    assert result.loc[("SP", "RJ"), "promised_days_route"] == 11
    assert result.loc[("MG", "SP"), "promised_days_route"] == 5


def test_fit_route_promises_rejects_missing_delivery_days():
    train = pd.DataFrame(
        {
            "seller_state": ["SP"],
            "customer_state": ["RJ"],
        }
    )

    with pytest.raises(ValueError, match="delivery_days"):
        fit_route_promises(train)


def test_build_route_orders_uses_main_seller_and_one_row_per_order(raw_tables):
    result = build_route_orders(raw_tables).set_index("order_id")

    assert len(result) == 3
    assert result.index.is_unique

    # o1 tiene dos vendedores: s1 vale 100 y s2 vale 50.
    # Por eso se toma s1, cuyo estado es SP.
    assert result.loc["o1", "seller_state"] == "SP"
    assert result.loc["o1", "customer_state"] == "SP"
    assert result.loc["o1", "delivery_days"] == 15
    assert result.loc["o1", "original_promised_days"] == 20
    assert result.loc["o2", "pedido_retrasado"] == 1


def test_apply_route_promises_uses_route_and_global_fallback():
    route_promises = pd.DataFrame(
        {
            "seller_state": ["SP"],
            "customer_state": ["RJ"],
            "promised_days_route": [11],
        }
    )
    test = pd.DataFrame(
        {
            "order_id": [
                "sigue_tarde",
                "evitado_global",
                "nuevo_incumplimiento",
            ],
            "seller_state": ["SP", "BA", "SP"],
            "customer_state": ["RJ", "SC", "RJ"],
            "delivery_days": [12, 31, 12],
            "original_promised_days": [10, 20, 15],
            "pedido_retrasado": [1, 1, 0],
        }
    )

    result = apply_route_promises(
        test,
        route_promises,
        global_promise_days=32,
    ).set_index("order_id")

    assert result.loc["sigue_tarde", "promise_source"] == "route"
    assert result.loc["sigue_tarde", "late_with_route_promise"] == 1
    assert result.loc["evitado_global", "promise_source"] == "global"
    assert result.loc["evitado_global", "late_with_route_promise"] == 0
    assert result.loc["evitado_global", "avoidable_promise_breach"] == 1
    assert result.loc["nuevo_incumplimiento", "new_promise_breach"] == 1
    assert result.loc["nuevo_incumplimiento", "promise_days_change"] == -4


def _route_orders_for_temporal_test(test_delivery_days: int) -> pd.DataFrame:
    dates = pd.date_range(
        "2017-01-01",
        periods=40,
        freq="D",
    )

    return pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(40)],
            PREDICTION_TIME_COL: dates,
            "seller_state": ["SP"] * 40,
            "customer_state": ["RJ"] * 40,
            "delivery_days": [10] * 24 + [20] * 7 + [test_delivery_days] * 9,
            "original_promised_days": [10] * 40,
            "pedido_retrasado": [0] * 24 + [1] * 7 + [1] * 9,
        }
    )


def test_route_promises_do_not_use_test_delivery_outcomes():
    base = _route_orders_for_temporal_test(test_delivery_days=100)
    changed_test = _route_orders_for_temporal_test(test_delivery_days=300)

    promises_before, _, summary_before = evaluate_route_promises(base)
    promises_after, _, summary_after = evaluate_route_promises(changed_test)

    pd.testing.assert_frame_equal(promises_before, promises_after)
    assert summary_before["global_promise_days"] == 10
    assert summary_after["global_promise_days"] == 10

import pandas as pd
import pytest

from app.vista import (
    LEVEL_ALL,
    MIN_BAR_SHARE,
    ROW_HEIGHT,
    TABLE_MAX_HEIGHT,
    filter_orders,
    format_factor,
    impact_weights,
    late_summary,
    lift_high,
    one_in,
    order_exists,
    plural,
    table_height,
    table_view,
)
from src.features import PREDICTION_TIME_COL
from src.target import TARGET_COL


@pytest.fixture
def scored():
    """Pedidos ya puntuados, del más riesgoso al menos riesgoso, como los deja orders_with_risk."""
    return pd.DataFrame(
        {
            "semaforo": ["🔴", "🟡", "🟢", "🟢"],
            "nivel_riesgo": ["alto", "medio", "bajo", "bajo"],
            "order_id": ["ABC123", "abd456", "xyz789", "xyz000"],
            PREDICTION_TIME_COL: pd.to_datetime(["2018-06-01"] * 4),
            "customer_state": ["SP", "RJ", "MG", "SP"],
            "promised_days": [10, 12, 9, 15],
            "max_distance_km": [1200.4, 300.6, 50.0, 80.0],
            "total_price": [100.0, 250.5, 30.0, 45.0],
            "total_freight": [20.0, 30.0, 5.0, 8.0],
            "prob_retraso": [0.62, 0.35, 0.08, 0.05],
            TARGET_COL: [1, 0, 0, 0],
        }
    )


def test_filter_combines_level_and_search_ignoring_case(scored):
    assert list(filter_orders(scored, "bajo", "XYZ")["order_id"]) == ["xyz789", "xyz000"]
    assert list(filter_orders(scored, "alto", "xyz")["order_id"]) == []
    assert list(filter_orders(scored, "alto", "")["order_id"]) == ["ABC123"]


def test_all_levels_option_ignores_the_level(scored):
    assert len(filter_orders(scored, LEVEL_ALL, "")) == 4
    assert list(filter_orders(scored, LEVEL_ALL, "ab")["order_id"]) == ["ABC123", "abd456"]


def test_order_exists_regardless_of_filters(scored):
    assert order_exists(scored, "xyz0")
    assert not order_exists(scored, "nope")
    assert not order_exists(scored, "")


def test_table_view_scores_from_0_to_100_and_shows_real_delay_on_demand(scored):
    view = table_view(scored, show_real=False)
    assert list(view["Puntaje"]) == [62, 35, 8, 5]
    assert "¿Se retrasó?" not in view.columns
    assert list(table_view(scored, show_real=True)["¿Se retrasó?"]) == ["Sí", "No", "No", "No"]


def test_table_height_fits_the_rows_up_to_a_cap():
    one_row = ROW_HEIGHT * 2 + 3  # cabecera + una fila + bordes
    assert table_height(1) == one_row
    assert table_height(0) == one_row  # nunca más chica que una fila
    assert table_height(3) == ROW_HEIGHT * 4 + 3
    assert table_height(1865) == TABLE_MAX_HEIGHT
    assert table_height(5, max_height=100) == 100


def test_late_summary_counts_real_delays_only_when_shown(scored):
    assert late_summary(table_view(scored, show_real=True)) == "Se retrasaron 1 (25,0 %). "
    assert late_summary(table_view(scored, show_real=False)) == ""
    assert late_summary(table_view(scored.iloc[:0], show_real=True)) == ""


def test_counts_are_written_with_thousands_dot_and_plural():
    assert plural(1, "pedido") == "1 pedido"
    assert plural(1865, "pedido") == "1.865 pedidos"
    assert plural(0, "pedido") == "0 pedidos"


def test_lift_is_share_of_delays_over_share_of_orders():
    summary = pd.DataFrame(
        {"pct_pedidos": [0.097, 0.3, 0.603], "pct_retrasos": [0.211, 0.4, 0.389]}, index=["alto", "medio", "bajo"]
    )
    assert lift_high(summary) == pytest.approx(0.211 / 0.097)
    assert format_factor(lift_high(summary)) == "2,2×"
    assert lift_high(summary.drop(columns="pct_retrasos")) is None


def test_impact_weights_are_relative_to_the_biggest_factor_shown():
    factors = pd.DataFrame(
        {
            "variable": ["a", "b", "c", "d", "e"],
            "etiqueta": list("ABCDE"),
            "valor": ["1"] * 5,
            "aporte": [0.8, 0.4, 0.01, -0.2, -0.6],
        }
    )
    up, down = impact_weights(factors, n_up=2, n_down=1)
    assert list(up["variable"]) == ["a", "b"]
    assert list(down["variable"]) == ["e"]  # primero el que más baja el riesgo
    assert list(up["peso"]) == pytest.approx([1.0, 0.5])
    assert down["peso"].iloc[0] == pytest.approx(0.75)

    _, tiny = impact_weights(factors.assign(aporte=[0.8, 0.4, 0.01, -0.001, -0.6]), n_up=2, n_down=2)
    assert tiny["peso"].min() == pytest.approx(MIN_BAR_SHARE)  # siempre se ve algo de barra


def test_one_in():
    assert one_in(0.097) == "1 de cada 10"
    assert one_in(0.8) == "todos"

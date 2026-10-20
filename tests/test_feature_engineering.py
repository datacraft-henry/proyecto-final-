import pandas as pd
import pytest

from src.feature_engineering import (
    FEATURE_BUILDERS,
    MIN_SELLER_HISTORY,
    FeatureContext,
    build_model_dataset,
    haversine_km,
    model_feature_lists,
    product_features,
    route_features,
    seller_history_features,
    seller_recent_load_features,
    holiday_features,
)
from src.features import PREDICTION_TIME_COL, leaky_columns
from src.target import TARGET_COL

ANCHORS = ["aprobacion", "compra"]


@pytest.fixture
def full_tables(raw_tables):
    """El mini dataset del conftest más códigos postales, geolocalización y categorías."""
    customers = raw_tables["customers"].assign(customer_zip_code_prefix=["01000", "01000", "20000"])
    sellers = raw_tables["sellers"].assign(seller_zip_code_prefix=["01000", "20000"])
    products = raw_tables["products"].assign(product_category_name=["moveis", None])
    geolocation = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": ["01000", "01000", "20000"],
            "geolocation_lat": [-23.5, -23.7, -22.9],
            "geolocation_lng": [-46.6, -46.6, -43.2],
        }
    )
    return {**raw_tables, "customers": customers, "sellers": sellers, "products": products,
            "geolocation": geolocation}


def _context(tables, anchor="aprobacion"):
    dataset = build_model_dataset(tables, anchor)
    return FeatureContext(base=dataset, tables=tables), dataset.set_index("order_id")


def test_haversine_matches_known_distance():
    # São Paulo - Río de Janeiro: unos 360 km en línea recta
    assert haversine_km(-23.55, -46.63, -22.91, -43.17) == pytest.approx(360, abs=10)
    assert haversine_km(-23.5, -46.6, -23.5, -46.6) == pytest.approx(0)


def test_route_features_distance_regions_and_cross_region(full_tables):
    _, dataset = _context(full_tables)
    # o2: cliente SP (01000) y vendedor s1 SP (01000): misma zona
    assert dataset.loc["o2", "max_distance_km"] == pytest.approx(0)
    assert dataset.loc["o2", "cross_region"] == "False"
    # o1: tiene vendedores en SP y RJ; se toma la distancia máxima
    assert dataset.loc["o1", "max_distance_km"] > 300
    assert dataset.loc["o3", "customer_region"] == "Sudeste"


def test_product_features_use_most_expensive_item(full_tables):
    _, dataset = _context(full_tables)
    assert dataset.loc["o1", "main_category"] == "moveis"  # p1 (100) le gana a p2 (50)
    assert dataset.loc["o3", "main_category"] == "unknown"  # p2 no tiene categoría


@pytest.mark.parametrize("anchor", ANCHORS)
def test_dataset_has_one_row_per_order_and_all_features(full_tables, anchor):
    dataset = build_model_dataset(full_tables, anchor)
    numeric, categorical = model_feature_lists(anchor)
    assert dataset["order_id"].is_unique
    assert set(numeric + categorical) <= set(dataset.columns)
    assert TARGET_COL in dataset.columns


@pytest.mark.parametrize("anchor", ANCHORS)
def test_no_builder_produces_leaky_columns(full_tables, anchor):
    numeric, categorical = model_feature_lists(anchor)
    leaky = set(leaky_columns(anchor))
    assert not leaky & set(numeric + categorical)
    for builder in FEATURE_BUILDERS:
        assert not leaky & set(builder.numeric + builder.categorical), builder.name


def _seller_tables(n_orders: int, late_every: int = 2):
    """Un vendedor con pedidos diarios; cada pedido se entrega 3 días después de la compra."""
    purchase = pd.date_range("2017-03-01 10:00", periods=n_orders, freq="D")
    delivered = purchase + pd.Timedelta(days=3)
    estimated = [
        (d - pd.Timedelta(days=1) if i % late_every == 0 else d + pd.Timedelta(days=1)).normalize()
        for i, d in enumerate(delivered)
    ]
    ids = [f"o{i}" for i in range(n_orders)]
    orders = pd.DataFrame(
        {
            "order_id": ids,
            "customer_id": "c1",
            "order_status": "delivered",
            "order_purchase_timestamp": purchase,
            "order_approved_at": purchase,
            "order_delivered_customer_date": delivered,
            "order_estimated_delivery_date": estimated,
        }
    )
    items = pd.DataFrame({"order_id": ids, "seller_id": "s1", "product_id": "p1", "price": 10.0})
    return {"orders": orders, "items": items}


def _history(tables):
    base = tables["orders"][["order_id"]].assign(**{PREDICTION_TIME_COL: tables["orders"]["order_approved_at"]})
    return seller_history_features(FeatureContext(base=base, tables=tables)).set_index("order_id")


def _seller_load(tables):
    base = tables["orders"][["order_id"]].assign(
        **{PREDICTION_TIME_COL: tables["orders"]["order_approved_at"]}
    )
    return seller_recent_load_features(
        FeatureContext(base=base, tables=tables)
    ).set_index("order_id")


def _holiday_calendar(tables):
    base = pd.DataFrame(
        {
            "order_id": [
                "feriado_actual",
                "previo_navidad",
                "cruce_anio",
                "dia_normal",
            ],
            PREDICTION_TIME_COL: [
                pd.Timestamp("2018-09-07 10:00"),
                pd.Timestamp("2018-12-24 10:00"),
                pd.Timestamp("2018-12-27 10:00"),
                pd.Timestamp("2018-09-08 10:00"),
            ],
        }
    )

    return holiday_features(
        FeatureContext(base=base, tables=tables)
    ).set_index("order_id")


def test_seller_history_counts_only_orders_delivered_before_prediction():
    history = _history(_seller_tables(12))
    # o4 se predice el día 4; solo o0 (entregado el día 3) llegó antes
    assert history.loc["o4", "seller_prev_orders"] == 1
    assert history.loc["o0", "seller_prev_orders"] == 0
    # con menos de MIN_SELLER_HISTORY pedidos previos la tasa queda vacía
    assert pd.isna(history.loc["o4", "seller_prev_late_rate"])
    late_order = f"o{MIN_SELLER_HISTORY + 3}"
    assert history.loc[late_order, "seller_prev_late_rate"] == pytest.approx(0.6)


def test_seller_recent_load_counts_only_previous_windows():
    load = _seller_load(_seller_tables(35))

    for days in (7, 14, 30):
        feature = f"seller_recent_approved_orders_{days}d"
        assert load.loc["o0", feature] == 0
        assert load.loc["o4", feature] == 4

    assert load.loc["o7", "seller_recent_approved_orders_7d"] == 7

    assert load.loc["o14", "seller_recent_approved_orders_7d"] == 7
    assert load.loc["o14", "seller_recent_approved_orders_14d"] == 14

    assert load.loc["o30", "seller_recent_approved_orders_7d"] == 7
    assert load.loc["o30", "seller_recent_approved_orders_14d"] == 14
    assert load.loc["o30", "seller_recent_approved_orders_30d"] == 30

    assert load.loc["o31", "seller_recent_approved_orders_7d"] == 7
    assert load.loc["o31", "seller_recent_approved_orders_14d"] == 14
    assert load.loc["o31", "seller_recent_approved_orders_30d"] == 30


def test_seller_history_builder_is_registered():
    builders = {builder.name: builder for builder in FEATURE_BUILDERS}

    assert "historial_vendedor" in builders
    assert builders["historial_vendedor"].numeric == (
        "seller_prev_orders",
        "seller_prev_late_rate",
    )

    numeric, _ = model_feature_lists()
    assert {"seller_prev_orders", "seller_prev_late_rate"} <= set(numeric)


def test_seller_recent_load_does_not_use_delivery_outcomes():
    tables = _seller_tables(12)
    before = _seller_load(tables)

    changed_orders = tables["orders"].assign(
        order_delivered_customer_date=pd.Timestamp("2099-01-01"),
        order_estimated_delivery_date=pd.Timestamp("2016-01-01"),
    )
    after = _seller_load({**tables, "orders": changed_orders})

    pd.testing.assert_frame_equal(before, after)


def test_holiday_features_marks_current_and_future_holidays(full_tables):
    result = _holiday_calendar(full_tables)

    assert result.loc["feriado_actual", "is_national_holiday"] == 1
    assert result.loc["feriado_actual", "n_national_holidays_next_7d"] == 0

    assert result.loc["previo_navidad", "is_national_holiday"] == 0
    assert result.loc["previo_navidad", "n_national_holidays_next_7d"] == 1

    assert result.loc["cruce_anio", "n_national_holidays_next_7d"] == 1

    assert result.loc["dia_normal", "is_national_holiday"] == 0
    assert result.loc["dia_normal", "n_national_holidays_next_7d"] == 0


def test_holiday_features_do_not_use_delivery_outcomes(full_tables):
    before = _holiday_calendar(full_tables)

    changed_tables = {
        **full_tables,
        "orders": full_tables["orders"].assign(
            order_delivered_customer_date=pd.Timestamp("2099-01-01"),
            order_estimated_delivery_date=pd.Timestamp("2016-01-01"),
        ),
    }
    after = _holiday_calendar(changed_tables)

    pd.testing.assert_frame_equal(before, after)


def test_seller_history_ignores_future_outcomes():
    """Cambiar el resultado de pedidos futuros no puede cambiar las features de un pedido anterior."""
    tables = _seller_tables(12)
    before = _history(tables)
    future = tables["orders"]["order_purchase_timestamp"] >= pd.Timestamp("2017-03-09")
    changed_orders = tables["orders"].assign(
        order_estimated_delivery_date=tables["orders"]["order_estimated_delivery_date"].where(
            ~future, pd.Timestamp("2016-01-01")
        )
    )
    after = _history({**tables, "orders": changed_orders})
    past = before.index[:6]
    pd.testing.assert_frame_equal(before.loc[past], after.loc[past])


def test_route_and_product_builders_return_one_row_per_order(full_tables):
    context, _ = _context(full_tables)
    for build in (route_features, product_features, seller_recent_load_features, holiday_features):
        assert build(context)["order_id"].is_unique


def test_coordinates_outside_brazil_are_ignored(full_tables):
    """Un punto erróneo en Europa no puede mover el centroide ni inflar la distancia."""
    bad_point = pd.DataFrame(
        {"geolocation_zip_code_prefix": ["01000"], "geolocation_lat": [48.8], "geolocation_lng": [2.3]}
    )
    tables = {**full_tables, "geolocation": pd.concat([full_tables["geolocation"], bad_point])}
    _, clean = _context(full_tables)
    _, with_bad_point = _context(tables)
    pd.testing.assert_series_equal(clean["max_distance_km"], with_bad_point["max_distance_km"])


def test_zip_without_valid_coordinates_has_no_distance(full_tables):
    geolocation = full_tables["geolocation"].assign(
        geolocation_lat=lambda d: d["geolocation_lat"].where(d["geolocation_zip_code_prefix"] != "20000", 48.8)
    )
    _, dataset = _context({**full_tables, "geolocation": geolocation})
    assert pd.isna(dataset.loc["o3", "max_distance_km"])  # cliente en 20000
    assert dataset.loc["o2", "max_distance_km"] == pytest.approx(0)

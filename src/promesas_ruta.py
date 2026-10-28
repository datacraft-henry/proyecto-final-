"""Promesas de entrega por ruta estado-vendedor a estado-cliente.

Este módulo no modifica el modelo de clasificación. Calcula una propuesta de
días a prometer usando el percentil 95 del tiempo real de entrega observado
solamente en el conjunto de entrenamiento.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import RAW_DIR, load_raw_tables
from src.features import APPROVED_COL, PREDICTION_TIME_COL
from src.target import (
    DELIVERED_COL,
    ESTIMATED_COL,
    PURCHASE_COL,
    TARGET_COL,
    build_target,
)
from src.validation import temporal_three_way_split

ROUTE_COLUMNS = ["seller_state", "customer_state"]
DELIVERY_DAYS_COL = "delivery_days"
PROMISE_DAYS_COL = "promised_days_route"
TRAIN_ORDERS_COL = "train_orders"
ORIGINAL_PROMISE_DAYS_COL = "original_promised_days"
PROMISE_SOURCE_COL = "promise_source"
ROUTE_LATE_COL = "late_with_route_promise"
AVOIDED_BREACH_COL = "avoidable_promise_breach"
NEW_BREACH_COL = "new_promise_breach"
PROMISE_CHANGE_COL = "promise_days_change"
ROUTE_QUANTILE = 0.95


def fit_route_promises(
    train: pd.DataFrame,
    quantile: float = ROUTE_QUANTILE,
) -> pd.DataFrame:
    """Calcula días a prometer por ruta usando únicamente pedidos de train.

    Para cada combinación estado del vendedor -> estado del cliente calcula el
    percentil indicado de los días reales de entrega y redondea hacia arriba al
    día entero siguiente.
    """
    required = set(ROUTE_COLUMNS + [DELIVERY_DAYS_COL])
    missing = required - set(train.columns)

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    if not 0 < quantile < 1:
        raise ValueError("quantile debe estar entre 0 y 1.")

    if train[DELIVERY_DAYS_COL].isna().any():
        raise ValueError(f"{DELIVERY_DAYS_COL} no puede tener valores nulos.")

    if (train[DELIVERY_DAYS_COL] < 0).any():
        raise ValueError(
            f"{DELIVERY_DAYS_COL} no puede tener valores negativos."
        )

    route_table = (
        train.groupby(ROUTE_COLUMNS, dropna=False)[DELIVERY_DAYS_COL]
        .agg(
            train_orders="size",
            delivery_days_p95=lambda values: values.quantile(quantile),
        )
        .reset_index()
    )

    route_table[PROMISE_DAYS_COL] = (
        np.ceil(route_table["delivery_days_p95"]).astype(int)
    )
    route_table["delivery_days_p95"] = route_table[
        "delivery_days_p95"
    ].round(2)

    return route_table


def _main_seller(items: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el vendedor principal: el del ítem más caro de cada pedido."""
    return (
        items.sort_values(["order_id", "price"], ascending=[True, False])
        .drop_duplicates("order_id")
        [["order_id", "seller_id"]]
    )


def build_route_orders(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Construye una fila por pedido con su ruta y tiempos observados.

    La ruta se define como estado del vendedor principal -> estado del cliente.
    El vendedor principal es el del ítem más caro, igual que en las features
    actuales del proyecto, para no tener más de una ruta por pedido.
    """
    required_tables = {"orders", "items", "sellers", "customers"}
    missing = required_tables - set(tables)

    if missing:
        raise ValueError(f"Faltan tablas requeridas: {sorted(missing)}")

    orders = build_target(tables["orders"])
    base_columns = [
        "order_id",
        "customer_id",
        PURCHASE_COL,
        APPROVED_COL,
        DELIVERED_COL,
        ESTIMATED_COL,
        TARGET_COL,
    ]
    base = orders[base_columns].copy()
    base[PREDICTION_TIME_COL] = pd.to_datetime(
        base[APPROVED_COL],
        errors="coerce",
    )

    # Mismo universo temporal del modelo: pedidos con aprobación registrada.
    base = base.dropna(subset=[PREDICTION_TIME_COL])

    main_seller = _main_seller(tables["items"])
    sellers = tables["sellers"][["seller_id", "seller_state"]]
    customers = tables["customers"][["customer_id", "customer_state"]]

    route_orders = (
        base.merge(
            main_seller,
            on="order_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            sellers,
            on="seller_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            customers,
            on="customer_id",
            how="left",
            validate="many_to_one",
        )
    )

    if route_orders[ROUTE_COLUMNS].isna().any().any():
        raise ValueError("Hay pedidos sin estado de vendedor o de cliente.")

    date_columns = [PURCHASE_COL, DELIVERED_COL, ESTIMATED_COL]
    for column in date_columns:
        route_orders[column] = pd.to_datetime(
            route_orders[column],
            errors="coerce",
        )

    if route_orders[date_columns].isna().any().any():
        raise ValueError("Hay pedidos sin fechas válidas para calcular la ruta.")

    purchase_days = route_orders[PURCHASE_COL].dt.normalize()
    delivered_days = route_orders[DELIVERED_COL].dt.normalize()
    estimated_days = route_orders[ESTIMATED_COL].dt.normalize()

    route_orders[DELIVERY_DAYS_COL] = (
        delivered_days - purchase_days
    ).dt.days
    route_orders[ORIGINAL_PROMISE_DAYS_COL] = (
        estimated_days - purchase_days
    ).dt.days

    if not route_orders["order_id"].is_unique:
        raise ValueError("La tabla de rutas debe tener una sola fila por pedido.")

    return route_orders.reset_index(drop=True)


def fit_global_promise(
    train: pd.DataFrame,
    quantile: float = ROUTE_QUANTILE,
) -> int:
    """Calcula el P95 global de train para rutas nunca vistas."""
    if DELIVERY_DAYS_COL not in train.columns:
        raise ValueError(f"Falta la columna {DELIVERY_DAYS_COL}.")

    return int(np.ceil(train[DELIVERY_DAYS_COL].quantile(quantile)))


def apply_route_promises(
    test: pd.DataFrame,
    route_promises: pd.DataFrame,
    global_promise_days: int,
) -> pd.DataFrame:
    """Aplica en test promesas ajustadas únicamente con información de train.

    Una ruta no vista en train recibe el P95 global de train como respaldo.
    """
    required_test = {
        "order_id",
        *ROUTE_COLUMNS,
        DELIVERY_DAYS_COL,
        ORIGINAL_PROMISE_DAYS_COL,
        TARGET_COL,
    }
    missing_test = required_test - set(test.columns)

    if missing_test:
        raise ValueError(
            f"Faltan columnas requeridas en test: {sorted(missing_test)}"
        )

    required_promises = {*ROUTE_COLUMNS, PROMISE_DAYS_COL}
    missing_promises = required_promises - set(route_promises.columns)

    if missing_promises:
        raise ValueError(
            "Faltan columnas requeridas en route_promises: "
            f"{sorted(missing_promises)}"
        )

    fitted_promises = route_promises[
        ROUTE_COLUMNS + [PROMISE_DAYS_COL]
    ].copy()

    if fitted_promises.duplicated(ROUTE_COLUMNS).any():
        raise ValueError("La tabla de promesas tiene rutas duplicadas.")

    result = test.merge(
        fitted_promises,
        on=ROUTE_COLUMNS,
        how="left",
        validate="many_to_one",
    )

    result[PROMISE_SOURCE_COL] = np.where(
        result[PROMISE_DAYS_COL].notna(),
        "route",
        "global",
    )
    result[PROMISE_DAYS_COL] = (
        result[PROMISE_DAYS_COL]
        .fillna(global_promise_days)
        .astype(int)
    )

    result[ROUTE_LATE_COL] = (
        result[DELIVERY_DAYS_COL] > result[PROMISE_DAYS_COL]
    ).astype(int)

    avoidable_breach = np.logical_and(
        result[TARGET_COL].eq(1),
        result[ROUTE_LATE_COL].eq(0),
    )
    result[AVOIDED_BREACH_COL] = avoidable_breach.astype(int)

    new_breach = np.logical_and(
        result[TARGET_COL].eq(0),
        result[ROUTE_LATE_COL].eq(1),
    )
    result[NEW_BREACH_COL] = new_breach.astype(int)

    result[PROMISE_CHANGE_COL] = (
        result[PROMISE_DAYS_COL] - result[ORIGINAL_PROMISE_DAYS_COL]
    )

    return result


def evaluate_route_promises(
    route_orders: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Entrena las promesas en train y las evalúa una única vez en test.

    La validación se conserva separada para respetar las tres cohortes
    temporales del proyecto, pero no interviene: el percentil 95 está definido
    antes de mirar los datos.
    """
    if PREDICTION_TIME_COL not in route_orders.columns:
        raise ValueError(
            f"Falta la columna temporal {PREDICTION_TIME_COL}."
        )

    train, validation, test = temporal_three_way_split(route_orders)
    route_promises = fit_route_promises(train)
    global_promise_days = fit_global_promise(train)
    scored_test = apply_route_promises(
        test,
        route_promises,
        global_promise_days,
    )

    original_breaches = int(scored_test[TARGET_COL].sum())
    route_breaches = int(scored_test[ROUTE_LATE_COL].sum())
    avoided_breaches = int(scored_test[AVOIDED_BREACH_COL].sum())
    new_breaches = int(scored_test[NEW_BREACH_COL].sum())

    summary = {
        "train_orders": int(len(train)),
        "validation_orders": int(len(validation)),
        "test_orders": int(len(test)),
        "route_count": int(len(route_promises)),
        "global_promise_days": global_promise_days,
        "original_breaches": original_breaches,
        "original_breach_rate": float(scored_test[TARGET_COL].mean()),
        "route_promise_breaches": route_breaches,
        "route_promise_breach_rate": float(
            scored_test[ROUTE_LATE_COL].mean()
        ),
        "avoidable_breaches": avoided_breaches,
        "new_breaches": new_breaches,
        "net_breaches_reduced": original_breaches - route_breaches,
        "median_promise_days_change": float(
            scored_test[PROMISE_CHANGE_COL].median()
        ),
        "mean_promise_days_change": float(
            scored_test[PROMISE_CHANGE_COL].mean()
        ),
        "route_specific_coverage": float(
            scored_test[PROMISE_SOURCE_COL].eq("route").mean()
        ),
    }

    return route_promises, scored_test, summary


def main() -> None:
    """Genera la tabla de promesas por ruta y la evalúa temporalmente."""
    parser = argparse.ArgumentParser(
        description=(
            "Calcula días a prometer por ruta con el P95 de train "
            "y evalúa la propuesta en test."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Carpeta que contiene los CSV originales de Olist.",
    )
    parser.add_argument(
        "--routes-output",
        type=Path,
        default=Path("reports/promesas_por_ruta.csv"),
        help="Salida CSV con la tabla de rutas.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/evaluacion_promesas_ruta.json"),
        help="Salida JSON con el resumen de test.",
    )
    args = parser.parse_args()

    tables = load_raw_tables(args.raw_dir)
    route_orders = build_route_orders(tables)
    route_promises, _, summary = evaluate_route_promises(route_orders)

    args.routes_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    route_promises.sort_values(
        [TRAIN_ORDERS_COL, "seller_state", "customer_state"],
        ascending=[False, True, True],
    ).to_csv(args.routes_output, index=False)

    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Tabla de rutas: {args.routes_output}")
    print(f"Resumen: {args.summary_output}")
    print()
    print(
        f"Test: {summary['test_orders']} pedidos | "
        f"rutas de train: {summary['route_count']}"
    )
    print(
        "Incumplimientos con promesa original: "
        f"{summary['original_breaches']} "
        f"({summary['original_breach_rate']:.2%})"
    )
    print(
        "Incumplimientos con P95 por ruta: "
        f"{summary['route_promise_breaches']} "
        f"({summary['route_promise_breach_rate']:.2%})"
    )
    print(
        "Incumplimientos evitables: "
        f"{summary['avoidable_breaches']} | "
        f"Nuevos incumplimientos: {summary['new_breaches']} | "
        f"Reducción neta: {summary['net_breaches_reduced']}"
    )
    print(
        "Cambio mediano de promesa: "
        f"{summary['median_promise_days_change']:.1f} días"
    )


if __name__ == "__main__":
    main()

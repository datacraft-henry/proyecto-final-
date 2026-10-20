"""Funciones reutilizables del script de unificación."""
import pandas as pd

from src.target import TARGET_COL, build_target


def agregar_target(pedidos: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Agrega pedido_retrasado a la tabla de pedidos sin perder filas.

    Los pedidos que build_target no puede etiquetar (no entregados, sin fecha
    o fuera del período de estudio) quedan con NaN, no con 0.
    """
    etiquetados = build_target(orders)
    solo_target = etiquetados[["order_id", TARGET_COL]]
    return pedidos.merge(solo_target, on="order_id", how="left")
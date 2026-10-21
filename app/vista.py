"""Preparación de lo que se muestra en pantalla, sin Streamlit, para poder testearlo.

La lógica de negocio (puntaje, nivel, factores) está en `app/riesgo.py`; acá solo
se filtra, se da formato y se calculan los pesos visuales.
"""

import pandas as pd

from src.features import PREDICTION_TIME_COL
from src.target import TARGET_COL

LEVEL_ALL = "todos"
SCORE_SCALE = 100  # el puntaje se muestra de 0 a 100
MIN_BAR_SHARE = 0.08  # ancho mínimo de una barra de impacto, para que siempre se vea
ROW_HEIGHT = 35  # alto de cada fila de la grilla (y de su cabecera), en píxeles
TABLE_FRAME = 3  # bordes de la grilla
TABLE_MAX_HEIGHT = 420


def table_height(n_rows: int, max_height: int = TABLE_MAX_HEIGHT) -> int:
    """Alto justo para las filas que hay (cabecera incluida), con tope para no estirar la tabla."""
    return min(max_height, ROW_HEIGHT * (max(n_rows, 1) + 1) + TABLE_FRAME)


def format_int(value) -> str:
    return f"{int(value):,}".replace(",", ".")


def format_pct(value: float) -> str:
    return f"{100 * value:.1f} %".replace(".", ",")


def format_factor(value: float) -> str:
    """Multiplicador con coma decimal: 2.17 -> '2,2×'."""
    return f"{value:.1f}×".replace(".", ",")


def plural(count: int, noun: str) -> str:
    """'1 pedido', '1.865 pedidos'."""
    return f"{format_int(count)} {noun}{'' if count == 1 else 's'}"


def filter_orders(orders: pd.DataFrame, level: str, search: str) -> pd.DataFrame:
    """Pedidos del nivel elegido (o todos) cuyo código contiene lo buscado."""
    listed = orders if level == LEVEL_ALL else orders[orders["nivel_riesgo"] == level]
    if search:
        listed = listed[listed["order_id"].str.contains(search, case=False, regex=False)]
    return listed


def order_exists(orders: pd.DataFrame, search: str) -> bool:
    """Si lo buscado coincide con algún pedido, sin importar los filtros."""
    return bool(search) and bool(orders["order_id"].str.contains(search, case=False, regex=False).any())


def table_view(orders: pd.DataFrame, show_real: bool) -> pd.DataFrame:
    """Tabla con nombres de columna para operaciones y puntaje de 0 a 100, junto al nivel."""
    view = pd.DataFrame(
        {
            "Nivel": orders["semaforo"] + " " + orders["nivel_riesgo"].str.capitalize(),
            "Puntaje": (orders["prob_retraso"] * SCORE_SCALE).round(0),
            "Pedido": orders["order_id"],
            "Aprobado": orders[PREDICTION_TIME_COL],
            "Estado": orders["customer_state"],
            "Plazo": orders["promised_days"],
            "Distancia": orders["max_distance_km"].round(0),
            "Precio": orders["total_price"],
            "Flete": orders["total_freight"],
        }
    )
    if show_real and TARGET_COL in orders.columns:
        # Junto al puntaje, para que se vea sin desplazar la tabla.
        view.insert(2, "¿Se retrasó?", orders[TARGET_COL].map({1: "Sí", 0: "No"}))
    return view


def late_summary(view: pd.DataFrame) -> str:
    """'Se retrasaron 142 (7,6 %). ' cuando la tabla muestra el dato real; vacío si no."""
    if "¿Se retrasó?" not in view.columns or view.empty:
        return ""
    late = view["¿Se retrasó?"] == "Sí"
    return f"Se retrasaron {format_int(late.sum())} ({format_pct(late.mean())}). "


def lift_high(summary: pd.DataFrame) -> float | None:
    """Cuántas veces más retrasos concentra el nivel alto que pedidos (2,2× = el doble y pico)."""
    if "pct_retrasos" not in summary.columns:
        return None
    share_orders = summary.loc["alto", "pct_pedidos"]
    return float(summary.loc["alto", "pct_retrasos"] / share_orders) if share_orders else None


def one_in(share: float) -> str:
    """Una proporción dicha como '1 de cada N' (0,097 -> '1 de cada 10')."""
    n = round(1 / share)
    return "todos" if n <= 1 else f"1 de cada {n}"


def impact_weights(factors: pd.DataFrame, n_up: int, n_down: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Los factores que más suben y más bajan el riesgo, con un peso de 0 a 1 para la barra.

    El peso es relativo al aporte más grande entre los factores mostrados, así las
    barras de los dos bloques se comparan entre sí.
    """
    up = factors[factors["aporte"] > 0].head(n_up)
    down = factors[factors["aporte"] < 0].sort_values("aporte").head(n_down)
    biggest = max([*up["aporte"].abs(), *down["aporte"].abs()], default=0) or 1.0

    def weight(part: pd.DataFrame) -> pd.Series:
        return (part["aporte"].abs() / biggest).clip(lower=MIN_BAR_SHARE)

    return up.assign(peso=weight(up)), down.assign(peso=weight(down))

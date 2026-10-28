"""Lógica de la app de riesgo por pedido, separada de Streamlit para poder testearla.

Usa el artefacto de src/umbral.py (models/modelo_principal_calibrado.joblib):
modelo reentrenado con train + validación, calibrador de Platt y cortes de riesgo
fijados en validación. El nivel (alto, medio, bajo) es el dato principal; la
probabilidad calibrada está inflada en los meses nuevos (ver docs/metricas_evaluacion.md).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.calibration import LEVELS
from src.features import PREDICTION_TIME_COL
from src.target import TARGET_COL
from src.umbral import CALIBRATED_MODEL_PATH, predict_risk

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / CALIBRATED_MODEL_PATH
SEMAFORO = {"alto": "🔴", "medio": "🟡", "bajo": "🟢"}
# Columnas que se muestran en la tabla de pedidos, además del riesgo.
DETAIL_COLUMNS = [
    "order_id",
    PREDICTION_TIME_COL,
    "customer_state",
    "promised_days",
    "max_distance_km",
    "total_price",
    "total_freight",
    "n_items",
]
# Nombre de cada variable del modelo para alguien de operaciones.
FEATURE_LABELS = {
    "purchase_dayofweek": "Día de la compra",
    "purchase_hour": "Hora de la compra",
    "promised_days": "Plazo prometido",
    "n_items": "Cantidad de productos",
    "n_sellers": "Cantidad de vendedores",
    "total_price": "Precio total",
    "total_freight": "Costo del flete",
    "freight_ratio": "Flete sobre el precio",
    "total_weight_g": "Peso total",
    "total_volume_cm3": "Volumen total",
    "max_installments": "Cuotas",
    "approval_hours": "Demora en aprobar el pago",
    "max_distance_km": "Distancia al vendedor",
    "n_national_holidays_next_7d": "Feriados en la semana siguiente",
    "customer_state": "Estado del cliente",
    "main_payment_type": "Medio de pago",
    "seller_other_state": "Vendedor en otro estado",
}
_WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_PAYMENT_TYPES = {
    "credit_card": "tarjeta de crédito",
    "boleto": "boleto bancario",
    "voucher": "voucher",
    "debit_card": "tarjeta de débito",
}


def _number(value: float, decimals: int = 0) -> str:
    """Número con punto de miles y coma decimal."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def _holidays(value: float) -> str:
    count = int(value)
    if count == 0:
        return "ninguno"
    return f"{count} feriado" if count == 1 else f"{count} feriados"


def format_feature_value(feature: str, value) -> str:
    """Valor de una variable escrito para leerlo de corrido."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "sin dato"
    formats = {
        "purchase_dayofweek": lambda v: _WEEKDAYS[int(v)],
        "purchase_hour": lambda v: f"{int(v)} h",
        "promised_days": lambda v: f"{int(v)} días",
        "n_items": lambda v: _number(v),
        "n_sellers": lambda v: _number(v),
        "total_price": lambda v: f"R$ {_number(v, 2)}",
        "total_freight": lambda v: f"R$ {_number(v, 2)}",
        "freight_ratio": lambda v: f"{_number(100 * v)} %",
        "total_weight_g": lambda v: f"{_number(v / 1000, 1)} kg",
        "total_volume_cm3": lambda v: f"{_number(v / 1000, 1)} litros",
        "max_installments": lambda v: _number(v),
        "approval_hours": lambda v: f"{_number(v, 1)} h",
        "max_distance_km": lambda v: f"{_number(v)} km",
        "n_national_holidays_next_7d": _holidays,
        "main_payment_type": lambda v: _PAYMENT_TYPES.get(str(v), str(v)),
        "seller_other_state": lambda v: "sí" if str(v).lower() in ("1", "true", "sí", "si") else "no",
    }
    return formats.get(feature, str)(value)


def load_artifact(path: Path = ARTIFACT_PATH) -> dict:
    """Carga el modelo calibrado. Falla con un mensaje claro si no se generó."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No está {path}. Generalo con `python -m src.umbral`.")
    return joblib.load(path)


def orders_with_risk(artifact: dict, orders: pd.DataFrame) -> pd.DataFrame:
    """Pedidos con probabilidad calibrada, nivel y semáforo, del más riesgoso al menos riesgoso.

    Si los pedidos traen el target (datos históricos), se conserva para comparar.
    """
    risk = predict_risk(artifact, orders)
    columns = [c for c in [*DETAIL_COLUMNS, TARGET_COL] if c in orders.columns]
    result = orders[columns].join(risk)
    result.insert(0, "semaforo", result["nivel_riesgo"].map(SEMAFORO))
    return result.sort_values("prob_retraso", ascending=False).reset_index(drop=True)


def summary_by_level(scored: pd.DataFrame) -> pd.DataFrame:
    """Por nivel: cantidad y % de pedidos y, si hay target, tasa real y % de los retrasos que capta."""
    grouped = scored.groupby("nivel_riesgo")
    summary = pd.DataFrame({"pedidos": grouped.size()}).reindex(list(LEVELS), fill_value=0)
    total = summary["pedidos"].sum()
    summary["pct_pedidos"] = summary["pedidos"] / total if total else 0.0
    if TARGET_COL in scored.columns:
        late = grouped[TARGET_COL].sum().reindex(list(LEVELS), fill_value=0)
        total_late = late.sum()
        summary["tasa_retraso_real"] = grouped[TARGET_COL].mean().reindex(list(LEVELS))
        summary["pct_retrasos"] = late / total_late if total_late else 0.0
    summary.insert(0, "semaforo", [SEMAFORO[level] for level in LEVELS])
    return summary


def _source_feature(transformed_name: str, features: list[str]) -> str:
    """Variable original de una columna transformada ('cat__customer_state_SP' -> 'customer_state')."""
    name = transformed_name.split("__", 1)[-1]
    matches = [f for f in features if name == f or name.startswith(f"{f}_")]
    return max(matches, key=len)


def feature_contributions(artifact: dict, orders: pd.DataFrame) -> pd.DataFrame:
    """Aporte de cada variable al riesgo de cada pedido, frente al pedido promedio de `orders`.

    Sirve para la regresión logística: el aporte es el coeficiente por la diferencia entre
    el valor del pedido y el promedio, en la escala que ve el modelo. Las categorías de una
    misma variable se suman. Una fila por pedido (en el orden de `orders`), una columna por variable.
    """
    pipeline, features = artifact["model"], artifact["features"]
    preprocess = pipeline.named_steps["preprocess"]
    coefs = pipeline.named_steps["model"].coef_[0]
    matrix = preprocess.transform(orders[features])
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    contributions = pd.DataFrame(coefs * (matrix - matrix.mean(axis=0)))
    sources = [_source_feature(n, features) for n in preprocess.get_feature_names_out()]
    return contributions.T.groupby(sources).sum().T


def factors_table(order: pd.Series, contributions: pd.Series) -> pd.DataFrame:
    """Una fila por variable, de la que más sube el riesgo a la que más lo baja, con su valor escrito."""
    result = pd.DataFrame(
        {
            "variable": contributions.index,
            "etiqueta": [FEATURE_LABELS.get(f, f) for f in contributions.index],
            "valor": [format_feature_value(f, order[f]) for f in contributions.index],
            "aporte": contributions.to_numpy(dtype=float),
        }
    )
    return result.sort_values("aporte", ascending=False).reset_index(drop=True)


def risk_factors(artifact: dict, orders: pd.DataFrame, order_id: str) -> pd.DataFrame:
    """Qué variables suben o bajan el riesgo de un pedido frente al pedido promedio de `orders`."""
    row = int(np.flatnonzero(orders["order_id"].to_numpy() == order_id)[0])
    contributions = feature_contributions(artifact, orders).iloc[row]
    return factors_table(orders.iloc[row], contributions)

"""API de riesgo de retraso: POST /predict devuelve la probabilidad y el nivel de cada pedido.

Usa el mismo artefacto y la misma función que la app de Streamlit
(models/modelo_principal_calibrado.joblib y src.umbral.predict_risk), así que
la API y la app dan siempre el mismo resultado.

Cómo correrla, desde la raíz del repo:

    python -m src.umbral                 # genera el artefacto (una sola vez)
    uvicorn api.main:app --reload        # documentación en http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.umbral import CALIBRATED_MODEL_PATH, predict_risk

MODEL_PATH_ENV = "MODELO_PATH"
MAX_PEDIDOS = 1000
TEXT_FIELDS = ["order_id", "customer_state", "main_payment_type", "seller_other_state"]


class Pedido(BaseModel):
    """Un pedido con las variables que se conocen al aprobar el pago.

    Cualquier campo puede venir vacío: el pipeline imputa los faltantes con lo
    aprendido en train. Un campo que no esté acá (por ejemplo la fecha de
    entrega) se rechaza con error 422: así la API no deja entrar fuga de datos.
    """

    model_config = ConfigDict(extra="forbid")

    order_id: str | None = Field(None, description="Identificador del pedido, solo para devolverlo en la respuesta")
    purchase_dayofweek: int | None = Field(None, ge=0, le=6, description="Día de la semana de la compra (0 = lunes)")
    purchase_hour: int | None = Field(None, ge=0, le=23, description="Hora de la compra")
    promised_days: float | None = Field(None, ge=0, description="Días prometidos entre la compra y la fecha estimada")
    approval_hours: float | None = Field(None, ge=0, description="Horas entre la compra y la aprobación del pago")
    n_items: int | None = Field(None, ge=1)
    n_sellers: int | None = Field(None, ge=1)
    total_price: float | None = Field(None, ge=0, description="Precio total del pedido (R$)")
    total_freight: float | None = Field(None, ge=0, description="Flete total (R$)")
    freight_ratio: float | None = Field(None, ge=0, description="Flete sobre precio")
    total_weight_g: float | None = Field(None, ge=0)
    total_volume_cm3: float | None = Field(None, ge=0)
    max_installments: int | None = Field(None, ge=0, description="Máximo de cuotas del pago")
    max_distance_km: float | None = Field(None, ge=0, description="Distancia máxima cliente-vendedor en km")
    n_national_holidays_next_7d: int | None = Field(
        None, ge=0, le=7, description="Feriados nacionales de Brasil en los 7 días desde la aprobación"
    )
    customer_state: str | None = Field(None, min_length=2, max_length=2, description="Estado del cliente, por ejemplo SP")
    main_payment_type: str | None = Field(None, description="credit_card, boleto, voucher o debit_card")
    seller_other_state: Literal["True", "False", "unknown"] | None = Field(
        None, description="Si algún vendedor está en otro estado que el cliente"
    )


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pedidos: list[Pedido] = Field(..., min_length=1, max_length=MAX_PEDIDOS)


class Prediccion(BaseModel):
    order_id: str | None
    prob_retraso: float = Field(..., description="Probabilidad calibrada. Leerla como orden, no como tasa exacta")
    nivel_riesgo: Literal["alto", "medio", "bajo"]


class PredictResponse(BaseModel):
    predicciones: list[Prediccion]


def model_path() -> Path:
    return Path(os.environ.get(MODEL_PATH_ENV, CALIBRATED_MODEL_PATH))


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = model_path()
    app.state.artifact = joblib.load(path) if path.exists() else None
    yield


app = FastAPI(
    title="DataCraft · Riesgo de retraso Olist",
    description="Probabilidad calibrada y nivel de riesgo (alto, medio, bajo) por pedido, al aprobar el pago.",
    version="1.0.0",
    lifespan=lifespan,
)


def _artifact(request: Request) -> dict:
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail=f"No hay modelo cargado. Generalo con `python -m src.umbral` (busca {model_path()}).",
        )
    return artifact


@app.get("/health")
def health(request: Request) -> dict:
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        return {"status": "sin_modelo", "modelo_path": str(model_path())}
    return {
        "status": "ok",
        "entrenado_con": artifact.get("entrenado_con"),
        "anchor": artifact.get("anchor"),
        "features": artifact["features"],
        "cortes_riesgo": artifact["cuts"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    artifact = _artifact(request)
    rows = [p.model_dump() for p in body.pedidos]
    orders = pd.DataFrame(rows)
    missing = [f for f in artifact["features"] if f not in orders.columns]
    if missing:
        # El artefacto pide una variable que el esquema no conoce: el modelo cambió y la API no.
        raise HTTPException(status_code=500, detail=f"El modelo espera variables que la API no recibe: {missing}")
    for column in orders.columns.difference(TEXT_FIELDS):
        orders[column] = pd.to_numeric(orders[column])  # un campo numérico todo vacío llega como object
    risk = predict_risk(artifact, orders)
    return PredictResponse(
        predicciones=[
            Prediccion(order_id=oid, prob_retraso=round(float(p), 4), nivel_riesgo=n)
            for oid, p, n in zip(orders["order_id"], risk["prob_retraso"], risk["nivel_riesgo"])
        ]
    )

"""Saca pedidos reales del test para la demo de la API.

Uso, desde la raíz del repo, con el modelo ya generado (python -m src.umbral):
    python -m scripts.demo_pedidos

Escribe reports/demo_pedidos.json con tres pedidos reales del período de test, listos
para pegar en /docs o mandar con curl:
  - alto_retrasado: el modelo lo marca alto y de verdad llegó tarde (acierto)
  - bajo_a_tiempo:  el modelo lo marca bajo y llegó a tiempo (acierto)
  - alto_a_tiempo:  el modelo lo marca alto y llegó a tiempo (falsa alarma, para ser honestas)
Además imprime el contexto real de cada uno (fechas y si se atrasó) para contarlo.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from api.main import Pedido
from src.data_loading import RAW_DIR, load_raw_tables
from src.feature_engineering import build_model_dataset, load_geolocation
from src.target import TARGET_COL
from src.umbral import CALIBRATED_MODEL_PATH, predict_risk
from src.validation import temporal_three_way_split

SEMILLA = 28  # fija, para que siempre salgan los mismos pedidos
CAMPOS_API = list(Pedido.model_fields)

artifact = joblib.load(CALIBRATED_MODEL_PATH)
tables = {**load_raw_tables(RAW_DIR), "geolocation": load_geolocation(RAW_DIR)}
dataset = build_model_dataset(tables, artifact["anchor"])
_, _, test = temporal_three_way_split(dataset)

test = test.join(predict_risk(artifact, test))
casos = {
    "alto_retrasado": (test["nivel_riesgo"] == "alto") & (test[TARGET_COL] == 1),
    "bajo_a_tiempo": (test["nivel_riesgo"] == "bajo") & (test[TARGET_COL] == 0),
    "alto_a_tiempo": (test["nivel_riesgo"] == "alto") & (test[TARGET_COL] == 0),
}

orders = tables["orders"].set_index("order_id")
fechas = ["order_purchase_timestamp", "order_estimated_delivery_date", "order_delivered_customer_date"]

pedidos = []
for nombre, filtro in casos.items():
    fila = test[filtro].sample(1, random_state=SEMILLA).iloc[0]
    pedido = {c: (None if pd.isna(fila.get(c)) else fila.get(c)) for c in CAMPOS_API if c != "order_id"}
    pedido = {"order_id": nombre, **{k: (v.item() if hasattr(v, "item") else v) for k, v in pedido.items()}}
    pedidos.append(pedido)

    real = orders.loc[fila["order_id"], fechas]
    print(f"\n{nombre}  (order_id real: {fila['order_id']})")
    print(f"  modelo: prob {fila['prob_retraso']:.3f}, nivel {fila['nivel_riesgo']}")
    print(f"  compra {str(real.iloc[0])[:10]} | prometido {str(real.iloc[1])[:10]} | entregado {str(real.iloc[2])[:10]}"
          f" | {'SE ATRASÓ' if fila[TARGET_COL] == 1 else 'a tiempo'}")
    print(f"  estado {pedido['customer_state']}, {pedido['promised_days']:.0f} días prometidos, "
          f"{pedido['max_distance_km']:.0f} km, vendedor en otro estado: {pedido['seller_other_state']}")

Path("reports/demo_pedidos.json").write_text(json.dumps({"pedidos": pedidos}, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nListo: reports/demo_pedidos.json (pegalo completo en POST /predict desde /docs)")
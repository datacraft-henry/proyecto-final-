"""Monitoreo mensual del nivel de riesgo alto con el modelo congelado.

Uso (desde la raíz del repo, después de python -m src.umbral):
    python -m src.monitoreo

Aplica models/modelo_principal_calibrado.joblib a los pedidos de test y, por mes,
compara el % de pedidos en nivel alto con el diseño (HIGH_SHARE, 5 % en validación)
y la tasa de retraso de ese nivel contra la del mes. Si el % se aleja mucho del 5 %
o el lift cae, es la señal para recalibrar los cortes. Validación queda afuera porque
el modelo final ya se entrenó con esos meses. Escribe reports/monitoreo_nivel_alto.json.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.calibration import HIGH_SHARE
from src.data_loading import RAW_DIR, load_raw_tables
from src.feature_engineering import build_model_dataset, load_geolocation
from src.features import PREDICTION_TIME_COL
from src.target import TARGET_COL
from src.umbral import CALIBRATED_MODEL_PATH, predict_risk
from src.validation import temporal_three_way_split

# Con menos pedidos en el mes las tasas son ruido (el test arranca el 27 de mayo)
MIN_ORDERS_PER_MONTH = 1000


def monthly_high_risk(scored: pd.DataFrame, min_orders: int = MIN_ORDERS_PER_MONTH) -> list[dict]:
    """Una fila por mes: pedidos, % en alto, tasa de retraso en alto y del mes, lift y % de retrasos captados.

    scored necesita PREDICTION_TIME_COL, TARGET_COL y nivel_riesgo. Se omiten los
    meses con menos de min_orders pedidos.
    """
    month = scored[PREDICTION_TIME_COL].dt.to_period("M").astype(str)
    rows = []
    for mes, group in scored.groupby(month):
        if len(group) < min_orders:
            continue
        late = group[TARGET_COL]
        high = group["nivel_riesgo"] == "alto"
        late_rate = float(late.mean())
        high_late_rate = float(late[high].mean()) if high.any() else 0.0
        rows.append(
            {
                "mes": mes,
                "pedidos": len(group),
                "pct_alto": float(high.mean()),
                "pct_alto_diseno": HIGH_SHARE,
                "tasa_retraso_alto": high_late_rate,
                "tasa_retraso_mes": late_rate,
                "lift_alto": high_late_rate / late_rate if late_rate else 0.0,
                "pct_retrasos_captados": float(late[high].sum() / late.sum()) if late.sum() else 0.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    if not CALIBRATED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Falta {CALIBRATED_MODEL_PATH}. Corré antes python -m src.umbral.")
    artifact = joblib.load(CALIBRATED_MODEL_PATH)
    tables = {**load_raw_tables(args.raw_dir), "geolocation": load_geolocation(args.raw_dir)}
    _, _, test = temporal_three_way_split(build_model_dataset(tables, artifact["anchor"]))
    scored = test[[PREDICTION_TIME_COL, TARGET_COL]].join(predict_risk(artifact, test))
    rows = monthly_high_risk(scored)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "monitoreo_nivel_alto.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Nivel alto por mes (objetivo: {HIGH_SHARE:.0%} de los pedidos)")
    for row in rows:
        print(
            f"  {row['mes']}  {row['pedidos']:5d} pedidos  alto {row['pct_alto']:5.1%}  "
            f"retraso alto {row['tasa_retraso_alto']:5.1%} vs mes {row['tasa_retraso_mes']:5.1%}  "
            f"lift {row['lift_alto']:.1f}  capta {row['pct_retrasos_captados']:5.1%}"
        )
    print(f"\nReporte: {out}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()

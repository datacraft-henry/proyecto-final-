import pandas as pd
import pytest

from src.features import PREDICTION_TIME_COL
from src.monitoreo import monthly_high_risk
from src.target import TARGET_COL


def _scored(month: str, levels: list[str], late: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            PREDICTION_TIME_COL: pd.Timestamp(f"{month}-10"),
            "nivel_riesgo": levels,
            TARGET_COL: late,
        }
    )


def test_monthly_high_risk_share_rate_and_lift():
    june = _scored("2018-06", ["alto", "alto", "bajo", "bajo", "medio"], [1, 0, 0, 0, 0])
    july = _scored("2018-07", ["bajo"] * 4, [1, 0, 0, 0])
    rows = monthly_high_risk(pd.concat([june, july]), min_orders=1)

    assert [r["mes"] for r in rows] == ["2018-06", "2018-07"]
    assert rows[0]["pct_alto"] == pytest.approx(0.4)
    assert rows[0]["tasa_retraso_alto"] == pytest.approx(0.5)
    assert rows[0]["lift_alto"] == pytest.approx(0.5 / 0.2)
    assert rows[0]["pct_retrasos_captados"] == pytest.approx(1.0)
    # mes sin pedidos en alto: no divide por cero
    assert rows[1]["pct_alto"] == 0
    assert rows[1]["lift_alto"] == 0


def test_monthly_high_risk_skips_months_with_few_orders():
    small = _scored("2018-05", ["alto"], [1])
    full = _scored("2018-06", ["bajo"] * 3, [0, 0, 1])
    rows = monthly_high_risk(pd.concat([small, full]), min_orders=2)
    assert [r["mes"] for r in rows] == ["2018-06"]

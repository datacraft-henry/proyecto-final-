"""Modo web de la app: archivos chicos ya puntuados, para publicarla en Streamlit Community Cloud.

En la nube no están los CSV de Kaggle ni el modelo (.joblib), que no se versionan. Por eso
la PC que tiene los datos y el modelo congelado genera una vez, con las mismas funciones que
usa la app, todo lo que las dos pestañas muestran:

    python -m app.datos_web

Escribe en app/datos_web/ (sí se versiona):
  - pedidos_test.parquet: los pedidos del período de test con su riesgo, nivel, variables y el
    aporte de cada variable al puntaje (el inspector "Qué sube y qué baja").
  - ranking_vendedores.parquet: para cada categoría y destino que ofrece la pestaña de vendedores,
    la lista completa en el orden de src.recomendador.recomendar.
  - resumen.json: la tabla de Precision@K y Recall@K, los intervalos, el diagnóstico de despacho,
    los cortes de riesgo y las fechas.

La app entra en modo web cuando no encuentra el modelo y sí estos archivos (o con
DATACRAFT_MODO_WEB=1). Las cifras son las de la PC que los generó, la fuente única.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.riesgo import ARTIFACT_PATH, feature_contributions, orders_with_risk
from app.vendedores import TOP_K
from src.recomendador import CRITERIOS

WEB_DIR = Path(__file__).resolve().parent / "datos_web"
ORDERS_FILE = WEB_DIR / "pedidos_test.parquet"
RANKING_FILE = WEB_DIR / "ranking_vendedores.parquet"
SUMMARY_FILE = WEB_DIR / "resumen.json"
CONTRIBUTION_PREFIX = "aporte__"
COMBO_COLUMNS = ["product_category_name", "customer_state"]
ENV_FLAG = "DATACRAFT_MODO_WEB"


def web_mode(artifact_path: Path = ARTIFACT_PATH, summary_file: Path = SUMMARY_FILE) -> bool:
    """True si la app tiene que leer los archivos ya puntuados en lugar de los CSV y el modelo."""
    if os.environ.get(ENV_FLAG) == "1":
        return True
    return not Path(artifact_path).exists() and Path(summary_file).exists()


# ----------------------------------------------------------------------------- generar (en la PC)


def scored_orders_snapshot(artifact: dict, orders: pd.DataFrame) -> pd.DataFrame:
    """Pedidos puntuados como los muestra la app, más sus variables y el aporte de cada una."""
    contributions = feature_contributions(artifact, orders).add_prefix(CONTRIBUTION_PREFIX)
    extra = orders[["order_id", *artifact["features"]]].join(contributions.set_axis(orders.index))
    scored = orders_with_risk(artifact, orders)
    extra = extra.drop(columns=[c for c in artifact["features"] if c in scored.columns])
    return scored.merge(extra, on="order_id", how="left", validate="one_to_one")


def sellers_ranking(orders: pd.DataFrame, sellers: pd.DataFrame, cut, combos: pd.DataFrame, seller_state) -> pd.DataFrame:
    """La lista completa de vendedores de cada combinación, en el orden de `recomendar`."""
    from src.recomendador import recomendar

    parts = [
        recomendar(orders, sellers, cut, category, state, k=len(sellers))
        .rename_axis("seller_id")
        .reset_index()
        .assign(product_category_name=category, customer_state=state, posicion=lambda d: range(1, len(d) + 1))
        for category, state in combos[COMBO_COLUMNS].itertuples(index=False)
    ]
    ranking = pd.concat(parts, ignore_index=True)
    ranking["seller_state"] = ranking["seller_id"].map(seller_state)
    return ranking[[*COMBO_COLUMNS, "posicion", "seller_id", "seller_state", "riesgo_predicho", "ventas_aqui",
                    "precio_mediano", "pedidos_hist"]]


def _interval(values) -> list[float]:
    return [float(v) for v in values]


def build_summary(artifact: dict, rec: dict, orders: pd.DataFrame) -> dict:
    """Lo que no es una tabla grande: cortes, fechas, evaluación del recomendador y diagnóstico."""
    metrics = rec["metrics"].reindex(CRITERIOS)
    categories = set(rec["combos"]["product_category_name"])
    return {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "modelo": {"entrenado_con": artifact["entrenado_con"], "anchor": artifact["anchor"], "cuts": artifact["cuts"]},
        "pedidos": {"n": int(len(orders))},
        "vendedores": {
            "cut": rec["cut"].isoformat(),
            "test_end": rec["test_end"].isoformat(),
            "top_k": TOP_K,
            "n_combos": int(rec["n_combos"]),
            "metrics": {name: {k: float(v) for k, v in row.items()} for name, row in metrics.iterrows()},
            "vs_random": _interval(rec["vs_random"]),
            "vs_history": _interval(rec["vs_history"]),
            "vs_popularity": _interval(rec["vs_popularity"]),
            "dispatch": {k: float(v) for k, v in rec["dispatch"].items()},
            "names": {k: v for k, v in rec["names"].items() if k in categories},
            "combos": rec["combos"].to_dict(orient="records"),
        },
    }


def generate(out_dir: Path = WEB_DIR) -> dict:
    """Puntúa con el modelo congelado y escribe los tres archivos del modo web. Devuelve el resumen."""
    from app.pestana_vendedores import compute_recommender
    from app.riesgo import load_artifact
    from src.data_loading import load_raw_tables
    from src.feature_engineering import build_model_dataset, load_geolocation
    from src.validation import temporal_three_way_split

    artifact = load_artifact()
    tables = {**load_raw_tables(), "geolocation": load_geolocation()}
    _, _, test = temporal_three_way_split(build_model_dataset(tables, artifact["anchor"]))
    orders = test.reset_index(drop=True)
    rec = compute_recommender(artifact, tables)

    out_dir.mkdir(parents=True, exist_ok=True)
    scored_orders_snapshot(artifact, orders).to_parquet(out_dir / ORDERS_FILE.name, index=False)
    sellers_ranking(rec["orders"], rec["sellers"], rec["cut"], rec["combos"], rec["seller_state"]).to_parquet(
        out_dir / RANKING_FILE.name, index=False
    )
    summary = build_summary(artifact, rec, orders)
    (out_dir / SUMMARY_FILE.name).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------- leer (en la nube)


def load_summary(path: Path = SUMMARY_FILE) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"No está {path}. Generalo con `python -m app.datos_web`.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_artifact_info(path: Path = SUMMARY_FILE) -> dict:
    """Lo que la app lee del artefacto sin cargar el modelo: con qué se entrenó y los cortes de riesgo."""
    return load_summary(path)["modelo"]


def load_scored_orders(path: Path = ORDERS_FILE) -> pd.DataFrame:
    """Los pedidos puntuados, del más riesgoso al menos riesgoso, con variables y aportes."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No está {path}. Generalo con `python -m app.datos_web`.")
    return pd.read_parquet(path)


def contributions_of(orders: pd.DataFrame, order_id: str) -> tuple[pd.Series, pd.Series]:
    """(fila del pedido, aporte de cada variable) para armar el inspector sin el modelo."""
    order = orders.loc[orders["order_id"] == order_id].iloc[0]
    columns = [c for c in orders.columns if c.startswith(CONTRIBUTION_PREFIX)]
    contributions = order[columns].astype(float)
    return order, contributions.set_axis([c.removeprefix(CONTRIBUTION_PREFIX) for c in columns])


def load_recommender(ranking_path: Path = RANKING_FILE, summary_path: Path = SUMMARY_FILE) -> dict:
    """El mismo diccionario que arma la pestaña de vendedores, leído de los archivos."""
    if not Path(ranking_path).exists():
        raise FileNotFoundError(f"No está {ranking_path}. Generalo con `python -m app.datos_web`.")
    summary = load_summary(summary_path)
    sellers = summary["vendedores"]
    ranking = pd.read_parquet(ranking_path)
    metrics = pd.DataFrame.from_dict(sellers["metrics"], orient="index").reindex(CRITERIOS)
    return {
        "ranking": ranking,
        "cut": pd.Timestamp(sellers["cut"]),
        "test_end": pd.Timestamp(sellers["test_end"]),
        "combos": pd.DataFrame(sellers["combos"]),
        "metrics": metrics,
        "n_combos": sellers["n_combos"],
        "vs_random": tuple(sellers["vs_random"]),
        "vs_history": tuple(sellers["vs_history"]),
        "vs_popularity": tuple(sellers["vs_popularity"]),
        "dispatch": sellers["dispatch"],
        "seller_state": ranking.drop_duplicates("seller_id").set_index("seller_id")["seller_state"],
        "names": sellers["names"],
        "cuts": summary["modelo"]["cuts"],
    }


def ranked_sellers(ranking: pd.DataFrame, category: str, state: str) -> pd.DataFrame:
    """La lista de una combinación como la devuelve `recomendar`: indexada por seller_id, en orden."""
    rows = ranking[(ranking["product_category_name"] == category) & (ranking["customer_state"] == state)]
    return (
        rows.sort_values("posicion")
        .set_index("seller_id")[["ventas_aqui", "precio_mediano", "riesgo_predicho", "pedidos_hist"]]
    )


if __name__ == "__main__":
    result = generate()
    for path in (ORDERS_FILE, RANKING_FILE, SUMMARY_FILE):
        print(f"{path.relative_to(WEB_DIR.parents[1])}: {path.stat().st_size / 1e6:.2f} MB")
    print(f"Pedidos de test: {result['pedidos']['n']:,}")
    print(f"Combinaciones de vendedores: {len(result['vendedores']['combos'])}")
    for name, row in result["vendedores"]["metrics"].items():
        print(f"  {name:12s} Precision@{TOP_K} {row['precision']:.1%} · Recall@{TOP_K} {row['recall']:.1%}"
              f" · retraso {row['retraso']:.2%}")

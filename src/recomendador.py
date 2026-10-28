"""Sistema de recomendación: vendedores ordenados por su riesgo de retraso.

Antes de comprar, en la categoría que busca y para su estado, el cliente ve los vendedores
con menor riesgo según el clasificador (decisión 12, docs/recomendador.md). La tasa de
retraso histórica del vendedor (riesgo_vendedores) y la popularidad quedan como baselines.
Todo se ordena con pedidos ANTES de un corte y se mide DESPUÉS, con el target del proyecto.

Uso:
    python -m src.recomendador          # cifras de docs/recomendador.md

La pestaña "Vendedores recomendados" de la app (app/pestana_vendedores.py) usa
preparar_clasificador, puntaje_vendedores y recomendar. Es una asociación observacional:
no prueba que recomendar a un vendedor reduzca el retraso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.target import TARGET_COL, build_target

CORTE = pd.Timestamp("2018-01-01")  # historial antes, medición desde acá
MIN_PEDIDOS = 10  # pedidos antes del corte para tener historial
PESO_SUAVIZADO = 20  # pedidos "de la media" que se suman, para no premiar un 0 % con pocos pedidos
TERCIOS = ["Menor riesgo", "Medio", "Mayor riesgo"]


def pedidos_un_vendedor(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Una fila por pedido con target y un solo vendedor: así el retraso se atribuye sin ambigüedad.

    Columnas: las de build_target más seller_id, product_id, price (del ítem más caro),
    customer_state y product_category_name.
    """
    pedidos = build_target(tablas["orders"])
    items = tablas["items"]
    n_vendedores = items.groupby("order_id")["seller_id"].nunique()
    un_vendedor = n_vendedores[n_vendedores == 1].index
    principal = (
        items[items["order_id"].isin(un_vendedor)]
        .sort_values(["order_id", "price"], ascending=[True, False])
        .drop_duplicates("order_id")[["order_id", "seller_id", "product_id", "price"]]
    )
    productos = tablas["products"][["product_id"]].copy()
    productos["product_category_name"] = tablas["products"].get("product_category_name")
    return (
        pedidos.merge(principal, on="order_id")
        .merge(tablas["customers"][["customer_id", "customer_state"]], on="customer_id")
        .merge(productos, on="product_id", how="left")
    )


def riesgo_vendedores(
    pedidos: pd.DataFrame,
    corte: pd.Timestamp = CORTE,
    min_pedidos: int = MIN_PEDIDOS,
    peso: float = PESO_SUAVIZADO,
    fecha_col: str = "order_purchase_timestamp",
) -> pd.DataFrame:
    """Riesgo de cada vendedor con historial, calculado SOLO con pedidos comprados antes del corte.

    puntaje = (retrasos + peso × tasa media) / (pedidos + peso): la tasa del vendedor
    suavizada hacia la media, de modo que un vendedor con pocos pedidos y 0 % no quede primero.
    Devuelve, indexado por seller_id: retrasos, pedidos_hist, retraso_hist y puntaje.
    """
    fecha = pd.to_datetime(pedidos[fecha_col])
    historial = pedidos[fecha < corte]
    media = historial[TARGET_COL].mean()
    vend = historial.groupby("seller_id")[TARGET_COL].agg(retrasos="sum", pedidos_hist="size")
    vend = vend[vend["pedidos_hist"] >= min_pedidos].copy()
    vend["retraso_hist"] = vend["retrasos"] / vend["pedidos_hist"]
    vend["puntaje"] = (vend["retrasos"] + peso * media) / (vend["pedidos_hist"] + peso)
    return vend


def tercios_despues_del_corte(pedidos: pd.DataFrame, vend: pd.DataFrame, corte: pd.Timestamp = CORTE) -> pd.DataFrame:
    """Retraso y precio de los pedidos posteriores al corte, por tercio del puntaje del vendedor.

    `ajustado_por_estado` estandariza la tasa de cada tercio a la mezcla de estados del cliente
    de todos los pedidos medidos, para que la geografía no explique la diferencia.
    """
    fecha = pd.to_datetime(pedidos["order_purchase_timestamp"])
    medidos = pedidos[fecha >= corte].merge(vend[["puntaje"]], left_on="seller_id", right_index=True)
    medidos["tercio"] = pd.qcut(medidos["puntaje"].rank(method="first"), 3, labels=TERCIOS)
    pesos = medidos["customer_state"].value_counts(normalize=True)

    def ajustado(grupo: pd.DataFrame) -> float:
        tasa = grupo.groupby("customer_state")[TARGET_COL].mean()
        w = pesos[tasa.index]
        return float((tasa * w).sum() / w.sum())

    por_tercio = medidos.groupby("tercio", observed=True)
    salida = por_tercio.agg(
        retraso=(TARGET_COL, "mean"), precio_mediano=("price", "median"), pedidos=(TARGET_COL, "size")
    )
    salida["ajustado_por_estado"] = por_tercio[[TARGET_COL, "customer_state"]].apply(ajustado)
    return salida


def retraso_que_empieza_en_el_vendedor(tablas: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Cuánto del retraso empieza en el despacho: entrega a la transportadora después del plazo límite.

    Devuelve la fracción de pedidos retrasados con despacho tarde y la tasa de retraso con y sin
    despacho tarde. La fecha de despacho es posterior a la compra: sirve para este diagnóstico,
    nunca como variable del modelo (decisión 4).
    """
    pedidos = build_target(tablas["orders"]).set_index("order_id")[[TARGET_COL]]
    limite = pd.to_datetime(tablas["items"].groupby("order_id")["shipping_limit_date"].max())
    despacho = pd.to_datetime(tablas["orders"].set_index("order_id")["order_delivered_carrier_date"])
    d = pedidos.join(limite).join(despacho).dropna()
    tarde = d["order_delivered_carrier_date"] > d["shipping_limit_date"]
    return {
        "pct_retrasos_con_despacho_tarde": float(tarde[d[TARGET_COL] == 1].mean()),
        "retraso_si_despacho_tarde": float(d.loc[tarde, TARGET_COL].mean()),
        "retraso_si_despacho_a_tiempo": float(d.loc[~tarde, TARGET_COL].mean()),
        "pedidos": int(len(d)),
    }


def candidatos_por_combinacion(pedidos: pd.DataFrame, vend: pd.DataFrame, corte: pd.Timestamp = CORTE) -> pd.DataFrame:
    """Vendedores con historial que vendieron cada categoría a cada estado, con su precio y su retraso posterior."""
    fecha = pd.to_datetime(pedidos["order_purchase_timestamp"])
    despues = pedidos[fecha >= corte].groupby("seller_id")[TARGET_COL].agg(retraso_post="mean", pedidos_post="size")
    con_hist = pedidos[pedidos["seller_id"].isin(vend.index)].copy()
    con_hist["product_category_name"] = con_hist["product_category_name"].fillna("sin_categoria")
    return (
        con_hist.groupby(["product_category_name", "customer_state", "seller_id"])
        .agg(precio_mediano=("price", "median"), ventas_aqui=("order_id", "size"))
        .reset_index()
        .merge(vend, left_on="seller_id", right_index=True)
        .merge(despues, left_on="seller_id", right_index=True, how="left")
    )


def preparar(tablas: dict[str, pd.DataFrame], corte: pd.Timestamp = CORTE):
    """Todo lo que muestra la app: (candidatos, tercios, resumen)."""
    pedidos = pedidos_un_vendedor(tablas)
    vend = riesgo_vendedores(pedidos, corte)
    tercios = tercios_despues_del_corte(pedidos, vend, corte)
    fecha = pd.to_datetime(pedidos["order_purchase_timestamp"])
    despues = pedidos[fecha >= corte]
    resumen = {
        "media_hist": float(pedidos.loc[fecha < corte, TARGET_COL].mean()),
        "n_vendedores": int(len(vend)),
        "cobertura": float(despues["seller_id"].isin(vend.index).mean()),
        "pedidos_un_vendedor": int(len(pedidos)),
        "pedidos_total": int(len(build_target(tablas["orders"]))),
        **retraso_que_empieza_en_el_vendedor(tablas),
    }
    return candidatos_por_combinacion(pedidos, vend, corte), tercios, resumen


def recompra_tras_primer_pedido(
    tablas: dict[str, pd.DataFrame], dias: int = 180, hasta: str = "2018-02-28"
) -> pd.DataFrame:
    """Qué parte de los clientes vuelve a comprar según si su primer pedido llegó tarde.

    Toma el primer pedido de cada cliente (`customer_unique_id`) con target, comprado hasta
    `hasta` para que todos tengan `dias` de seguimiento, y marca si hizo otra compra en ese plazo.
    Devuelve, por valor del target: clientes, recompra y recompra ajustada por estado del cliente
    (estandarizada a la mezcla de estados de todos los primeros pedidos). Es observacional.
    """
    clientes = tablas["customers"][["customer_id", "customer_unique_id", "customer_state"]]
    pedidos = tablas["orders"][["order_id", "customer_id", "order_purchase_timestamp"]].merge(clientes, on="customer_id")
    pedidos["compra"] = pd.to_datetime(pedidos["order_purchase_timestamp"])
    pedidos = pedidos.sort_values("compra", kind="stable")
    primeros = pedidos.drop_duplicates("customer_unique_id").merge(
        build_target(tablas["orders"])[["order_id", TARGET_COL]], on="order_id"
    )
    primeros = primeros[primeros["compra"] <= pd.Timestamp(hasta) + pd.Timedelta(days=1)]
    siguientes = pedidos[["customer_unique_id", "compra"]].merge(
        primeros[["customer_unique_id", "compra"]], on="customer_unique_id", suffixes=("", "_primera")
    )
    plazo = siguientes["compra"] - siguientes["compra_primera"]
    vuelven = siguientes.loc[(plazo > pd.Timedelta(0)) & (plazo <= pd.Timedelta(days=dias)), "customer_unique_id"]
    primeros["vuelve"] = primeros["customer_unique_id"].isin(vuelven)
    pesos = primeros["customer_state"].value_counts(normalize=True)

    def ajustada(grupo: pd.DataFrame) -> float:
        tasa = grupo.groupby("customer_state")["vuelve"].mean()
        w = pesos[tasa.index]
        return float((tasa * w).sum() / w.sum())

    por_target = primeros.groupby(TARGET_COL)
    salida = por_target["vuelve"].agg(clientes="size", recompra="mean")
    salida["recompra_ajustada_por_estado"] = por_target[["vuelve", "customer_state"]].apply(ajustada)
    return salida


def resenas_por_retraso(tablas: dict[str, pd.DataFrame], resenas: pd.DataFrame) -> pd.DataFrame:
    """Reseñas de 1 estrella y nota media según si el pedido llegó tarde (una fila por reseña).

    `resenas` es olist_order_reviews_dataset.csv, que load_raw_tables no carga a propósito: la
    reseña es posterior a la entrega y sirve para justificar la recomendación, nunca como
    variable del modelo (decisión 4).
    """
    resenas = resenas[["order_id", "review_score"]].merge(
        build_target(tablas["orders"])[["order_id", TARGET_COL]], on="order_id"
    )
    return resenas.groupby(TARGET_COL).agg(
        resenas=("review_score", "size"),
        una_estrella=("review_score", lambda x: float((x == 1).mean())),
        nota_media=("review_score", "mean"),
    )



# --------------------------------------------------------------------------- sistema de recomendación
# El recomendador ordena a los vendedores con el riesgo que predice el clasificador (el modelo
# calibrado y congelado de src/umbral.py). La tasa histórica (riesgo_vendedores) y la popularidad
# quedan como baselines. Todo se mide en el período de test, que el clasificador no vio.

FECHA_PREDICCION = "order_approved_at"  # el clasificador predice al aprobar el pago
MIN_PEDIDOS_TEST = 5  # pedidos del vendedor en test para medir su retraso
CRITERIOS = ["clasificador", "historial", "popularidad", "azar"]


def preparar_clasificador(tablas: dict[str, pd.DataFrame], artefacto: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Pedidos de un vendedor con `prob_retraso` del clasificador, y el inicio del período de test.

    `tablas` necesita también "geolocation" (para la distancia). El inicio del test es el mismo
    que usa el clasificador (temporal_three_way_split): lo anterior es historial, lo posterior
    se usa solo para medir.
    """
    from src.features import PREDICTION_TIME_COL
    from src.feature_engineering import build_model_dataset
    from src.umbral import predict_risk
    from src.validation import temporal_three_way_split

    dataset = build_model_dataset(tablas, artefacto["anchor"])
    _, _, test = temporal_three_way_split(dataset)
    inicio_test = pd.to_datetime(test[PREDICTION_TIME_COL]).min()
    riesgo = predict_risk(artefacto, dataset)["prob_retraso"]
    probs = pd.DataFrame({"order_id": dataset["order_id"], "prob_retraso": riesgo.to_numpy()})
    pedidos = pedidos_un_vendedor(tablas).merge(probs, on="order_id")  # trae order_approved_at de build_target
    pedidos["product_category_name"] = pedidos["product_category_name"].fillna("sin_categoria")
    return pedidos, inicio_test


def puntaje_vendedores(
    pedidos: pd.DataFrame, corte: pd.Timestamp, min_pedidos: int = MIN_PEDIDOS, fecha_col: str = FECHA_PREDICCION
) -> pd.DataFrame:
    """Por vendedor, con pedidos ANTES del corte: riesgo medio predicho, puntaje histórico y pedidos.

    - `riesgo_predicho`: media de `prob_retraso` de sus pedidos (el criterio del recomendador).
    - `puntaje`: tasa de retraso suavizada (baseline de historial, ver riesgo_vendedores).
    - `pedidos_hist`: cantidad de pedidos (baseline de popularidad).
    Solo vendedores con al menos `min_pedidos` pedidos antes del corte.
    """
    historial = pedidos[pd.to_datetime(pedidos[fecha_col]) < corte]
    riesgo = historial.groupby("seller_id")["prob_retraso"].mean().rename("riesgo_predicho")
    vend = riesgo_vendedores(pedidos, corte, min_pedidos, fecha_col=fecha_col)
    return vend[["puntaje", "pedidos_hist"]].join(riesgo)


def _orden(candidatos: pd.DataFrame, criterio: str) -> pd.DataFrame:
    if criterio == "clasificador":
        return candidatos.sort_values("riesgo_predicho", kind="stable")
    if criterio == "historial":
        return candidatos.sort_values("puntaje", kind="stable")
    if criterio == "popularidad":
        return candidatos.sort_values("pedidos_hist", ascending=False, kind="stable")
    raise ValueError(f"Criterio desconocido: {criterio}")


def evaluar_recomendacion(
    pedidos: pd.DataFrame,
    vend: pd.DataFrame,
    corte: pd.Timestamp,
    k: int = 3,
    min_pedidos_post: int = MIN_PEDIDOS_TEST,
    fecha_col: str = FECHA_PREDICCION,
) -> pd.DataFrame:
    """Una fila por combinación de categoría y estado del cliente medida después del corte.

    Simula la pantalla del cliente: en cada combinación se muestran K vendedores según cada
    criterio. Candidatos: vendedores con historial (`vend`) que vendieron ahí después del corte,
    con al menos `min_pedidos_post` pedidos después del corte (no sabemos quién más ofrecía el
    producto). Un vendedor es "bueno" si después del corte se retrasó menos que la tasa media
    posterior. Solo cuentan combinaciones con al menos 2K candidatos y algún vendedor bueno.

    Columnas por criterio (clasificador, historial, popularidad, azar): precision_*, recall_* y
    retraso_* (tasa de retraso posterior media de los K mostrados, promediada por vendedor). Al
    azar se usa el valor esperado: proporción de buenos, K / candidatos y retraso medio.
    """
    post = pedidos[pd.to_datetime(pedidos[fecha_col]) >= corte]
    tasa_post = post[TARGET_COL].mean()
    resultado = post.groupby("seller_id")[TARGET_COL].agg(retraso_post="mean", pedidos_post="size")
    medibles = vend.join(resultado[resultado["pedidos_post"] >= min_pedidos_post], how="inner")
    medibles["bueno"] = medibles["retraso_post"] < tasa_post

    combinaciones = (
        post[post["seller_id"].isin(medibles.index)]
        .groupby(["product_category_name", "customer_state"])["seller_id"]
        .unique()
    )
    filas = []
    for (categoria, estado), vendedores in combinaciones.items():
        cand = medibles.loc[vendedores]
        if len(cand) < 2 * k or not cand["bueno"].any():
            continue
        fila = {"categoria": categoria, "estado": estado, "candidatos": len(cand)}
        for criterio in CRITERIOS[:-1]:
            top = _orden(cand, criterio).head(k)
            fila[f"precision_{criterio}"] = top["bueno"].mean()
            fila[f"recall_{criterio}"] = top["bueno"].sum() / cand["bueno"].sum()
            fila[f"retraso_{criterio}"] = top["retraso_post"].mean()
        fila["precision_azar"] = cand["bueno"].mean()
        fila["recall_azar"] = k / len(cand)
        fila["retraso_azar"] = cand["retraso_post"].mean()
        filas.append(fila)
    if not filas:
        raise ValueError("No hay combinaciones con suficientes candidatos para medir")
    tabla = pd.DataFrame(filas)
    tabla.attrs.update(k=k, tasa_post=float(tasa_post))
    return tabla


def resumen_evaluacion(tabla: pd.DataFrame) -> pd.DataFrame:
    """Promedio por criterio de Precision@K, Recall@K y retraso de los mostrados."""
    return pd.DataFrame(
        {m: [tabla[f"{m}_{c}"].mean() for c in CRITERIOS] for m in ("precision", "recall", "retraso")},
        index=CRITERIOS,
    )


def ic_diferencia(tabla: pd.DataFrame, a: str, b: str, metrica: str = "retraso", n: int = 2000, semilla: int = 0):
    """Diferencia media de `metrica` entre los criterios a y b, con IC 95 % por bootstrap de combinaciones.

    Optimista: un vendedor grande aparece en varias combinaciones, que no son independientes.
    """
    dif = (tabla[f"{metrica}_{a}"] - tabla[f"{metrica}_{b}"]).to_numpy()
    rng = np.random.default_rng(semilla)
    medias = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(n)]
    return float(dif.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def recomendar(
    pedidos: pd.DataFrame,
    vend: pd.DataFrame,
    corte: pd.Timestamp,
    categoria: str,
    estado: str,
    k: int = 3,
    fecha_col: str = FECHA_PREDICCION,
) -> pd.DataFrame:
    """Los K vendedores de menor riesgo predicho que ya vendieron esa categoría a ese estado.

    Usa solo pedidos anteriores al corte (lo que se sabría al recomendar). Devuelve, indexado por
    seller_id: riesgo_predicho, ventas_aqui (pedidos de esa categoría a ese estado), precio_mediano
    y pedidos_hist, del menor al mayor riesgo.
    """
    historial = pedidos[pd.to_datetime(pedidos[fecha_col]) < corte]
    aqui = historial[(historial["product_category_name"] == categoria) & (historial["customer_state"] == estado)]
    ventas = aqui.groupby("seller_id").agg(ventas_aqui=("order_id", "size"), precio_mediano=("price", "median"))
    candidatos = ventas.join(vend[["riesgo_predicho", "pedidos_hist"]], how="inner")
    return candidatos.sort_values("riesgo_predicho", kind="stable").head(k)


if __name__ == "__main__":
    import joblib

    from src.data_loading import RAW_DIR, load_raw_tables
    from src.feature_engineering import load_geolocation
    from src.umbral import CALIBRATED_MODEL_PATH

    tablas = load_raw_tables()
    pedidos, inicio_test = preparar_clasificador(
        {**tablas, "geolocation": load_geolocation()}, joblib.load(CALIBRATED_MODEL_PATH)
    )
    vend = puntaje_vendedores(pedidos, inicio_test)
    print(f"Historial hasta {inicio_test:%d/%m/%Y}: {len(vend)} vendedores con al menos {MIN_PEDIDOS} pedidos")
    for k in (3, 5):
        tabla = evaluar_recomendacion(pedidos, vend, inicio_test, k=k)
        print(f"\nK={k} en test: {len(tabla)} combinaciones (mediana {tabla['candidatos'].median():.0f} candidatos), "
              f"tasa de retraso {tabla.attrs['tasa_post']:.1%}")
        for criterio, fila in resumen_evaluacion(tabla).iterrows():
            print(f"  {criterio:12s} Precision@{k} {fila['precision']:.1%} · Recall@{k} {fila['recall']:.1%}"
                  f" · retraso de los mostrados {fila['retraso']:.2%}")
        if k == 3:
            for a, b in [("clasificador", "azar"), ("historial", "azar"), ("popularidad", "azar"),
                         ("clasificador", "historial")]:
                d, lo, hi = ic_diferencia(tabla, a, b)
                print(f"  retraso {a} − {b}: {d * 100:+.2f} pp (IC 95 % {lo * 100:+.2f} a {hi * 100:+.2f})")
    for dias, hasta in [(180, "2018-02-28"), (365, "2017-08-31")]:
        r = recompra_tras_primer_pedido(tablas, dias, hasta)
        print(f"\nRecompra a {dias} días (primeros pedidos hasta {hasta}):")
        for target, fila in r.iterrows():
            print(
                f"  {'tarde' if target == 1 else 'a tiempo':8s} {fila['clientes']:,.0f} clientes · "
                f"recompra {fila['recompra']:.1%} · ajustada por estado {fila['recompra_ajustada_por_estado']:.1%}"
            )
    r = resenas_por_retraso(tablas, pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv"))
    print("\nReseñas:")
    for target, fila in r.iterrows():
        print(f"  {'tarde' if target == 1 else 'a tiempo':8s} {fila['resenas']:,.0f} reseñas · "
              f"1 estrella {fila['una_estrella']:.1%} · nota media {fila['nota_media']:.2f}")

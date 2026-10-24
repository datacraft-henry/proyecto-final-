"""Prototipo de visión: ordenar vendedores por su riesgo de retraso histórico.

Es el tercer uso de la misma señal de riesgo (ver docs/recomendador.md): antes de
comprar, mostrar al cliente los vendedores con menos retraso. No usa el modelo: ordena
a cada vendedor por su tasa de retraso ANTES de un corte de fecha y mide si ese orden
se sostiene DESPUÉS del corte, con el target del proyecto (src/target.py).

Uso:
    from src.recomendador import preparar
    candidatos, tercios, resumen = preparar(tablas)

La app está en prototipos/recomendador_vendedores.py. Es una asociación observacional:
no prueba que recomendar a un vendedor reduzca el retraso.
"""

from __future__ import annotations

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
) -> pd.DataFrame:
    """Riesgo de cada vendedor con historial, calculado SOLO con pedidos comprados antes del corte.

    puntaje = (retrasos + peso × tasa media) / (pedidos + peso): la tasa del vendedor
    suavizada hacia la media, de modo que un vendedor con pocos pedidos y 0 % no quede primero.
    Devuelve, indexado por seller_id: retrasos, pedidos_hist, retraso_hist y puntaje.
    """
    fecha = pd.to_datetime(pedidos["order_purchase_timestamp"])
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

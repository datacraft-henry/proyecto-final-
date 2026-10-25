"""Pestaña de vendedores recomendados de la app (app/streamlit_app.py).

Compone la pantalla con src/recomendador.py: para la categoría y el estado del cliente,
los vendedores que ya vendieron ahí, en el orden de `recomendar`, y cómo rinde el
recomendador en test frente a historial, popularidad y azar. Los helpers sin Streamlit
están en app/vendedores.py.
"""

import pandas as pd
import streamlit as st

from app.vendedores import (
    TOP_K,
    categorias,
    combinaciones,
    de_cada_100,
    es_significativa,
    estados_de,
    nombre_categoria,
    nombre_estado,
    resumen_negocio,
    resumen_top,
    tabla_metricas,
    tabla_vendedores,
    texto_diferencia,
    texto_reales,
    texto_vendedores,
)
from app.vista import ROW_HEIGHT, format_int, format_pct, table_height
from src.data_loading import RAW_DIR, load_raw_tables
from src.feature_engineering import load_geolocation
from src.recomendador import (
    FECHA_PREDICCION,
    MIN_PEDIDOS,
    evaluar_recomendacion,
    ic_diferencia,
    preparar_clasificador,
    puntaje_vendedores,
    recomendar,
    resumen_evaluacion,
)

DEFAULT_CATEGORY, DEFAULT_STATE = "cama_mesa_banho", "RJ"  # la categoría que más se vende y un estado con varios vendedores


@st.cache_resource(show_spinner="Calculando el riesgo de los vendedores (la primera vez tarda cerca de un minuto)...")
def get_recommender(_artifact: dict) -> dict:
    """Lo que usa la pestaña de vendedores, calculado una vez con src.recomendador.

    Pedidos con el riesgo del clasificador, vendedores con puntaje, el corte (inicio del test),
    las combinaciones que se pueden elegir y la evaluación en test frente a los baselines.
    El artefacto no se hashea (guion bajo): es el mismo modelo en toda la sesión.
    """
    tables = {**load_raw_tables(), "geolocation": load_geolocation()}
    orders, cut = preparar_clasificador(tables, _artifact)
    sellers = puntaje_vendedores(orders, cut)
    evaluation = evaluar_recomendacion(orders, sellers, cut, k=TOP_K)
    history = orders[(pd.to_datetime(orders[FECHA_PREDICCION]) < cut) & orders["seller_id"].isin(sellers.index)]
    translation_file = RAW_DIR / "product_category_name_translation.csv"
    names = {}
    if translation_file.exists():
        translation = pd.read_csv(translation_file)
        names = dict(zip(translation["product_category_name"], translation["product_category_name_english"]))
    return {
        "orders": orders,
        "sellers": sellers,
        "cut": cut,
        "combos": combinaciones(history),
        "metrics": resumen_evaluacion(evaluation),
        "n_combos": len(evaluation),
        "vs_random": ic_diferencia(evaluation, "clasificador", "azar"),
        "vs_history": ic_diferencia(evaluation, "clasificador", "historial"),
        "vs_popularity": ic_diferencia(evaluation, "clasificador", "popularidad"),
        "seller_state": tables["sellers"].set_index("seller_id")["seller_state"],
        "names": names,
        "cuts": _artifact["cuts"],
    }


def show_sellers_page(p: dict, artifact: dict) -> None:
    """Pestaña del recomendador: para una categoría y el estado del cliente, los vendedores de menor riesgo."""
    try:
        rec = get_recommender(artifact)
    except FileNotFoundError as error:
        st.error(f"No se pudo cargar el recomendador. {error}")
        st.stop()
    cut = rec["cut"]

    st.markdown(
        '<p class="dc-hero">¿A qué vendedor conviene comprarle?</p>'
        '<p class="dc-muted">Para la categoría que busca el cliente y su estado, los vendedores que ya vendieron ahí, '
        "ordenados por el riesgo de retraso que el clasificador predice para sus pedidos. "
        f"Usa solo pedidos aprobados antes del {cut:%d/%m/%Y}.</p>",
        unsafe_allow_html=True,
    )
    category, state = show_sellers_toolbar(rec["combos"], rec["names"])
    listed = recomendar(rec["orders"], rec["sellers"], cut, category, state, k=len(rec["sellers"]))
    show_sellers_cards(listed, p)
    st.markdown(f"### Los {TOP_K} de menor riesgo")
    show_sellers_table(listed.head(TOP_K), state, rec["seller_state"], rec["cuts"])
    st.info(
        "**Se retrasan menos frente a lo que prometen, no entregan más rápido.** En parte el recomendador "
        "elige a quien promete más días: en test, los recomendados prometían unos 2 días más que los candidatos en general.",
        icon=":material/schedule:",
    )
    with st.expander(f"Ver los {texto_vendedores(len(listed))} para elegir"):
        show_sellers_table(listed, state, rec["seller_state"], rec["cuts"])
    show_evaluation(rec)
    show_sellers_help(cut)


def show_sellers_toolbar(combos: pd.DataFrame, names: dict) -> tuple[str, str]:
    """Categoría y estado del cliente; solo combinaciones con más vendedores que los que se recomiendan."""
    category_options = sorted(categorias(combos), key=lambda c: nombre_categoria(c, names))
    with st.container(key="dc-toolbar"):
        st.markdown('<div class="dc-toolbar-title">Compra</div>', unsafe_allow_html=True)
        left, right = st.columns(2, vertical_alignment="bottom")
        category = left.selectbox(
            "Categoría del producto",
            category_options,
            index=category_options.index(DEFAULT_CATEGORY) if DEFAULT_CATEGORY in category_options else 0,
            format_func=lambda c: nombre_categoria(c, names),
        )
        states = estados_de(combos, category)
        state = right.selectbox(
            "Estado del cliente",
            states,
            index=states.index(DEFAULT_STATE) if DEFAULT_STATE in states else 0,
            format_func=nombre_estado,
        )
    return category, state


def show_sellers_cards(listed: pd.DataFrame, p: dict) -> None:
    """Cuántos vendedores hay para elegir y cómo son los recomendados frente a todos."""
    top = resumen_top(listed)
    cards = [
        (p["text"], "Vendedores para elegir", format_int(top["candidatos"]), f"⭐ Se recomiendan los {top['top']} de menor riesgo."),
        (
            p["level"]["bajo"],
            "Pedidos que se espera que lleguen tarde",
            de_cada_100(top["riesgo_top"]),
            f"Con los recomendados. Con todos: {de_cada_100(top['riesgo_todos'])}.",
        ),
        (
            p["accent"],
            "Precio mediano de los recomendados",
            texto_reales(top["precio_top"]),
            f"De todos: {texto_reales(top['precio_todos'])}.",
        ),
    ]
    for column, (color, title, number, note) in zip(st.columns(len(cards)), cards):
        column.markdown(
            f'<div class="dc-card" style="--lvl:{color}">'
            f'<div class="t">{title}</div><div class="n">{number}</div><div class="p">{note}</div></div>',
            unsafe_allow_html=True,
        )


def show_sellers_table(listed: pd.DataFrame, state: str, seller_state: pd.Series, cuts: dict) -> None:
    view = tabla_vendedores(listed, state, seller_state, cuts)
    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        height=table_height(len(view)),
        row_height=ROW_HEIGHT,
        column_config={
            "#": st.column_config.NumberColumn(width=40),
            "": st.column_config.TextColumn(width=36),
            "Vendedor": st.column_config.TextColumn(width=96, help="Primeros 8 caracteres del código del vendedor"),
            "Riesgo": st.column_config.TextColumn(
                width=90, help="Con los mismos cortes que los pedidos a revisar. La lista va del menor al mayor riesgo."
            ),
            "Llegan tarde": st.column_config.TextColumn(
                help="De cada 100 pedidos suyos, cuántos espera el modelo que lleguen después de la fecha prometida."
            ),
            "Ventas aquí": st.column_config.NumberColumn(help="Pedidos de esta categoría a este estado."),
            "Precio mediano": st.column_config.NumberColumn(format="R$ %.2f", help="De sus ventas aquí."),
            "Pedidos (historial)": st.column_config.NumberColumn(help="Todos sus pedidos antes del corte."),
        },
    )


def show_evaluation(rec: dict) -> None:
    """Si sirve recomendar así, en palabras; cómo se midió y las cifras técnicas en un desplegable."""
    st.markdown("### ¿Sirve recomendar así?")
    business = resumen_negocio(rec["metrics"])
    if business:
        st.markdown(
            f'<p class="dc-lead">Si el cliente le compra a uno de los {TOP_K} recomendados, '
            f'<b>{business["modelo"]} de cada 1.000</b> pedidos llegan tarde. Si elige un vendedor cualquiera, '
            f'<b>{business["azar"]} de cada 1.000</b>. Son <b>{business["evitados"]} pedidos tarde menos cada 1.000</b>'
            + (f' (1 de cada {business["uno_de_cada"]} retrasos).' if business["uno_de_cada"] else ".")
            + "</p>",
            unsafe_allow_html=True,
        )
        if business["reglas"]:
            low, high = business["reglas"]
            rules = f"{low} de cada 1.000" if low == high else f"entre {low} y {high} de cada 1.000"
            st.markdown(
                '<p class="dc-muted">Para ser honestos: elegir al vendedor que menos se atrasó antes, o al que más '
                f"vende, da casi lo mismo ({rules}). El modelo suma poco frente a esas reglas simples, porque la "
                "mayoría de los retrasos se produce en el transporte y no en el vendedor.</p>",
                unsafe_allow_html=True,
            )
    with st.expander("Cómo lo medimos"):
        st.markdown(
            f"Lo probamos con pedidos desde el {rec['cut']:%d/%m/%Y}, que el modelo nunca vio. En "
            f"{format_int(rec['n_combos'])} combinaciones de categoría y estado comparamos cuatro formas de elegir "
            f"{TOP_K} vendedores. Un vendedor \"cumple\" si se atrasó menos que el promedio."
        )
        st.dataframe(tabla_metricas(rec["metrics"]), width="stretch")
        lines = [
            "- **Frente a elegir uno cualquiera**, la mejora "
            + ("no es casualidad." if es_significativa(rec["vs_random"]) else "podría ser casualidad.")
        ]
        for name, key in (("el que menos se atrasó antes", "vs_history"), ("el que más vende", "vs_popularity")):
            lines.append(
                f"- **Frente a elegir {name}**, "
                + ("la mejora no es casualidad." if es_significativa(rec[key])
                   else "la diferencia es tan chica que podría ser casualidad.")
            )
        st.markdown("\n".join(lines))
        technical = rec["metrics"].loc["clasificador"]
        st.caption(
            f"Para el jurado técnico: Precision@{TOP_K} {format_pct(technical['precision'])}, "
            f"Recall@{TOP_K} {format_pct(technical['recall'])}. Diferencia en el retraso frente al azar "
            f"{texto_diferencia(rec['vs_random'])}; frente al historial {texto_diferencia(rec['vs_history'])}; "
            f"frente a la popularidad {texto_diferencia(rec['vs_popularity'])}."
        )


def show_sellers_help(cut: pd.Timestamp) -> None:
    with st.expander("Límites del recomendador"):
        st.markdown(
            "- **Premia en parte a quien promete más días.** Recomienda cumplir lo prometido, no entregar más rápido.\n"
            "- **El vendedor explica solo una parte del retraso.** 7 de cada 10 retrasos ocurren en el transporte, "
            "no en el despacho del vendedor.\n"
            f"- **Solo vendedores con historial:** al menos {MIN_PEDIDOS} pedidos antes del {cut:%d/%m/%Y}. "
            "Los nuevos no se pueden recomendar.\n"
            "- **Solo vemos lo vendido,** no qué vendedores ofrecían el mismo producto ni a qué precio.\n"
            "- **Es observacional.** El efecto de recomendar se mediría con una prueba A/B.\n"
            "- Detalle y cifras en `docs/recomendador.md` (`python -m src.recomendador` las reproduce)."
        )

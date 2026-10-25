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
    es_significativa,
    estados_de,
    nombre_categoria,
    nombre_estado,
    resumen_top,
    tabla_metricas,
    tabla_vendedores,
    texto_diferencia,
    texto_pct,
    texto_reales,
    texto_vendedores,
)
from app.vista import ROW_HEIGHT, format_int, table_height
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
    # La misma escala en las dos tablas: la barra de los recomendados se compara con la de todos.
    risk_max = 100 * float(listed["riesgo_predicho"].max()) if "riesgo_predicho" in listed else 100.0
    show_sellers_table(listed.head(TOP_K), state, rec["seller_state"], risk_max)
    st.info(
        "**Se retrasan menos frente a lo que prometen, no entregan más rápido.** En parte el recomendador "
        "elige a quien promete más días: en test, los recomendados prometían unos 2 días más que los candidatos en general.",
        icon=":material/schedule:",
    )
    with st.expander(f"Ver los {texto_vendedores(len(listed))} para elegir"):
        show_sellers_table(listed, state, rec["seller_state"], risk_max)
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
            "Riesgo predicho de los recomendados",
            texto_pct(top["riesgo_top"]),
            f"Promedio de todos: {texto_pct(top['riesgo_todos'])}.",
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


def show_sellers_table(listed: pd.DataFrame, state: str, seller_state: pd.Series, risk_max: float) -> None:
    view = tabla_vendedores(listed, state, seller_state)
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
            "Riesgo predicho": st.column_config.ProgressColumn(
                format="%.1f %%",
                min_value=0,
                max_value=max(1.0, risk_max),
                help="Riesgo de retraso medio que el clasificador predice para sus pedidos. Ordena la lista: menor, primero.",
            ),
            "Ventas aquí": st.column_config.NumberColumn(help="Pedidos de esta categoría a este estado."),
            "Precio mediano": st.column_config.NumberColumn(format="R$ %.2f", help="De sus ventas aquí."),
            "Pedidos (historial)": st.column_config.NumberColumn(help="Todos sus pedidos antes del corte."),
        },
    )


def show_evaluation(rec: dict) -> None:
    """Precision@K, Recall@K y retraso de los mostrados en test, frente a historial, popularidad y azar."""
    st.markdown("### ¿Cuánto ayuda recomendar así?")
    st.markdown(
        f'<p class="dc-muted">Medido en el período de test (desde el {rec["cut"]:%d/%m/%Y}), que el clasificador '
        f"no vio: en {format_int(rec['n_combos'])} combinaciones de categoría y estado se mostraron {TOP_K} "
        "vendedores con cada criterio. Un vendedor es bueno si en test se retrasó menos que la media.</p>",
        unsafe_allow_html=True,
    )
    st.dataframe(tabla_metricas(rec["metrics"]), width="stretch")
    lines = [f"- **Frente al azar**, el retraso de los mostrados cambia {texto_diferencia(rec['vs_random'])}."]
    for name, key in (("historial", "vs_history"), ("popularidad", "vs_popularity")):
        ic = rec[key]
        verdict = (
            "La diferencia es significativa."
            if es_significativa(ic)
            else f"La diferencia no es significativa: no se distingue de recomendar por {name}."
        )
        lines.append(f"- **Frente a la {name}**, cambia {texto_diferencia(ic)}. {verdict}")
    st.markdown("\n".join(lines))


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

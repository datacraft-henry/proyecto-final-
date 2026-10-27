"""Pestaña de vendedores recomendados de la app (app/streamlit_app.py).

Compone la pantalla con src/recomendador.py para el responsable del marketplace: para una
categoría y un destino, los vendedores que ya vendieron ahí, en el orden de `recomendar`,
qué tener en cuenta antes de elegir y qué gana Olist con las recomendaciones, medido con
pedidos históricos. Las métricas quedan en un desplegable de respaldo técnico. Los
helpers sin Streamlit están en app/vendedores.py.
"""

from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from app.datos_web import load_recommender, ranked_sellers, web_mode
from app.vendedores import (
    CRITERIOS,
    TOP_K,
    categorias,
    combinaciones,
    comparacion_criterios,
    es_significativa,
    estados_de,
    nivel_riesgo,
    nombre_categoria,
    nombre_estado,
    origen,
    puntaje,
    tabla_metricas,
    tabla_vendedores,
    texto_diferencia,
    texto_reales,
    texto_vendedores,
)
from app.vista import ROW_HEIGHT, SCORE_SCALE, format_int, format_pct, table_height
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
    retraso_que_empieza_en_el_vendedor,
)

DEFAULT_CATEGORY, DEFAULT_STATE = "cama_mesa_banho", "RJ"  # la categoría que más se vende y un estado con varios vendedores
CHART_ROW = 34  # alto de cada barra del gráfico de la evaluación, en píxeles


def compute_recommender(artifact: dict, tables: dict) -> dict:
    """Lo que usa la pestaña de vendedores, calculado con src.recomendador.

    Pedidos con el riesgo del clasificador, vendedores con puntaje, el corte (inicio del test),
    las combinaciones que se pueden elegir, la evaluación en test frente a los baselines y el
    diagnóstico de despacho. También lo usa app/datos_web.py para generar el modo web.
    """
    orders, cut = preparar_clasificador(tables, artifact)
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
        "test_end": pd.to_datetime(orders[FECHA_PREDICCION]).max(),
        "combos": combinaciones(history),
        "metrics": resumen_evaluacion(evaluation),
        "n_combos": len(evaluation),
        "vs_random": ic_diferencia(evaluation, "clasificador", "azar"),
        "vs_history": ic_diferencia(evaluation, "clasificador", "historial"),
        "vs_popularity": ic_diferencia(evaluation, "clasificador", "popularidad"),
        "dispatch": retraso_que_empieza_en_el_vendedor(tables),
        "seller_state": tables["sellers"].set_index("seller_id")["seller_state"],
        "names": names,
        "cuts": artifact["cuts"],
    }


@st.cache_resource(show_spinner="Calculando el riesgo de los vendedores (la primera vez tarda cerca de un minuto)...")
def get_recommender(_artifact: dict) -> dict:
    """Se calcula una vez. El artefacto no se hashea (guion bajo): es el mismo modelo en toda la sesión.

    En modo web se lee de los archivos ya puntuados (app/datos_web.py) y el artefacto no se usa.
    """
    if web_mode():
        return load_recommender()
    return compute_recommender(_artifact, {**load_raw_tables(), "geolocation": load_geolocation()})


def listed_sellers(rec: dict, category: str, state: str) -> pd.DataFrame:
    """Todos los vendedores de la combinación, en el orden del recomendador."""
    if "ranking" in rec:
        return ranked_sellers(rec["ranking"], category, state)
    return recomendar(rec["orders"], rec["sellers"], rec["cut"], category, state, k=len(rec["sellers"]))


def show_sellers_page(p: dict, artifact: dict) -> None:
    """Pestaña del recomendador: para una categoría y un destino, los vendedores a considerar."""
    try:
        rec = get_recommender(artifact)
    except FileNotFoundError as error:
        st.error(f"No se pudo cargar el recomendador. {error}")
        st.stop()
    cut = rec["cut"]

    st.markdown(
        '<p class="dc-hero">Vendedores a considerar para esta categoría y destino</p>'
        '<p class="dc-muted">Ayuda a decidir a qué vendedores mirar primero cuando un cliente busca un producto. '
        f"Es una demostración con pedidos históricos de Olist aprobados antes del {cut:%d/%m/%Y}: "
        "no reemplaza confirmar con el vendedor el producto, el stock, el precio y el plazo.</p>",
        unsafe_allow_html=True,
    )
    category, state = show_sellers_toolbar(rec["combos"], rec["names"])
    listed = listed_sellers(rec, category, state)
    st.markdown(
        f'<p class="dc-lead"><b>{texto_vendedores(len(listed))}</b> con historial ya vendieron '
        f"{escape(nombre_categoria(category, rec['names']))} con destino {escape(nombre_estado(state))}. "
        f"Estos son los {TOP_K} de menor puntaje de riesgo.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dc-note"><b>Cómo se ordena.</b> La categoría y el destino definen quién aparece: '
        "los vendedores que ya vendieron ahí. El orden usa el puntaje de riesgo medio que el modelo dio a "
        "<i>todos</i> los pedidos anteriores de cada vendedor, en cualquier categoría y destino. "
        "No es una predicción para esta ruta ni para este producto.</div>",
        unsafe_allow_html=True,
    )
    show_seller_cards(listed.head(TOP_K), state, rec, p)
    show_before_choosing(cut)
    with st.expander(f"Comparar los {texto_vendedores(len(listed))}"):
        show_sellers_table(listed, state, rec["seller_state"], rec["cuts"])
    show_evaluation(rec, p)
    show_technical(rec)


def show_sellers_toolbar(combos: pd.DataFrame, names: dict) -> tuple[str, str]:
    """Categoría y destino; solo combinaciones con más vendedores que los que se destacan."""
    category_options = sorted(categorias(combos), key=lambda c: nombre_categoria(c, names))
    with st.container(key="dc-toolbar"):
        st.markdown('<div class="dc-toolbar-title">Búsqueda</div>', unsafe_allow_html=True)
        left, right = st.columns(2, vertical_alignment="bottom")
        category = left.selectbox(
            "Categoría del producto",
            category_options,
            index=category_options.index(DEFAULT_CATEGORY) if DEFAULT_CATEGORY in category_options else 0,
            format_func=lambda c: nombre_categoria(c, names),
        )
        states = estados_de(combos, category)
        state = right.selectbox(
            "Destino (estado del cliente)",
            states,
            index=states.index(DEFAULT_STATE) if DEFAULT_STATE in states else 0,
            format_func=nombre_estado,
        )
    return category, state


def show_seller_cards(top: pd.DataFrame, state: str, rec: dict, p: dict) -> None:
    """Una tarjeta por vendedor destacado: posición, nivel, origen, ventas y precio histórico."""
    levels = nivel_riesgo(top["riesgo_predicho"], rec["cuts"]) if "riesgo_predicho" in top else [""] * len(top)
    for column, (position, (seller, row)), level in zip(st.columns(TOP_K), enumerate(top.iterrows(), 1), levels):
        level_key = level.split(" ")[-1].lower() if level else "bajo"
        facts = [
            ("Origen", origen(rec["seller_state"].get(seller), state)),
            ("Ventas en este contexto", format_int(row["ventas_aqui"]) if "ventas_aqui" in row else "—"),
            ("Precio histórico de referencia", texto_reales(row.get("precio_mediano"))),
            ("Pedidos anteriores", format_int(row["pedidos_hist"]) if "pedidos_hist" in row else "—"),
        ]
        score = puntaje(row.get("riesgo_predicho"))
        column.markdown(
            f'<div class="dc-seller" style="--lvl:{p["level"][level_key]};--soft:{p["level_soft"][level_key]};'
            f'--ink:{p["level_text"][level_key]}">'
            f'<div class="dc-seller-top"><span class="dc-rank">⭐ {position}.º</span>'
            f'<span class="dc-badge">{level}</span></div>'
            f'<div class="dc-seller-id" title="Código completo del vendedor">{escape(str(seller))}</div>'
            f'<div class="dc-score">Puntaje de riesgo <b>{score if score is not None else "—"}</b>/{SCORE_SCALE}</div>'
            '<div class="dc-facts">'
            + "".join(f"<div><small>{label}</small><b>{escape(str(value))}</b></div>" for label, value in facts)
            + "</div></div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "Puntaje de riesgo: de 0 a 100, sirve para ordenar; no es la probabilidad de que un pedido llegue tarde. "
        "Precio histórico de referencia: la mediana de lo que cada vendedor cobró en sus ventas de esta categoría "
        "a este destino, en cualquier producto. No es el precio actual ni del mismo producto."
    )


def show_before_choosing(cut: pd.Timestamp) -> None:
    st.markdown("#### Qué tener en cuenta antes de elegir")
    st.markdown(
        "- **Cumplir la fecha prometida no es entregar más rápido.** En la evaluación, los vendedores destacados "
        "prometían en promedio unos 2 días más que el resto: en parte se los elige por prometer plazos más largos.\n"
        f"- **Solo aparecen vendedores con historial:** al menos {MIN_PEDIDOS} pedidos antes del {cut:%d/%m/%Y}. "
        "Los vendedores nuevos no se pueden evaluar.\n"
        "- **Confirmá producto, stock, precio y plazo con el vendedor.** La pantalla solo muestra lo que vendió "
        "antes; no sabe qué ofrece hoy ni a qué precio."
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
            "#": st.column_config.NumberColumn(width=40, help="Posición en el orden del recomendador"),
            "": st.column_config.TextColumn(width=36),
            "Vendedor": st.column_config.TextColumn(width=150, help="Código completo del vendedor"),
            "Riesgo": st.column_config.TextColumn(width=90, help="Con los mismos cortes que los pedidos a revisar."),
            "Puntaje": st.column_config.NumberColumn(
                width=70, help="Puntaje de riesgo medio de sus pedidos anteriores, de 0 a 100. Sirve para ordenar."
            ),
            "Ventas en este contexto": st.column_config.NumberColumn(help="Pedidos de esta categoría a este destino."),
            "Precio histórico de referencia": st.column_config.NumberColumn(
                format="R$ %.2f", help="Mediana de lo que cobró en sus ventas aquí. No es el precio actual."
            ),
            "Pedidos anteriores": st.column_config.NumberColumn(help="Todos sus pedidos antes del corte."),
        },
    )


def show_evaluation(rec: dict, p: dict) -> None:
    """Qué gana Olist: el resultado frente al azar, el gráfico y los motivos de negocio."""
    st.markdown("### Qué gana Olist con las recomendaciones")
    comparison = comparacion_criterios(rec["metrics"])
    if comparison.empty:
        return
    rates = dict(zip(comparison["criterio"], comparison["tasa"]))
    labels = dict(zip(comparison["criterio"], comparison["etiqueta"]))
    st.markdown(
        f'<p class="dc-muted">Repetimos la elección con pedidos aprobados entre el {rec["cut"]:%d/%m/%Y} y el '
        f'{rec["test_end"]:%d/%m/%Y}, que el modelo no vio, en {format_int(rec["n_combos"])} combinaciones de '
        f"categoría y destino. Para cada forma de elegir, medimos la tasa media de retraso de los {TOP_K} "
        "vendedores que mostraba.</p>"
        f'<p class="dc-lead">{_headline(rates, labels, rec)}</p>',
        unsafe_allow_html=True,
    )
    st.altair_chart(_comparison_chart(comparison, p), width="stretch")
    st.caption(
        "Tasas medias por vendedor y combinación, en pedidos históricos ya entregados. Muestra qué pasó con esos "
        "vendedores; el impacto en compras nuevas se confirma con una prueba A/B (ver próximos pasos)."
    )
    st.markdown(
        "#### Por qué le sirve a Olist\n"
        "- **Menos retrasos, mejores reseñas.** El retraso es lo que más castiga la nota: de las reseñas de "
        "pedidos que llegaron tarde, el 53,7 % es de 1 estrella, contra el 6,6 % de los que llegaron a tiempo.\n"
        "- **Sin costo extra para el cliente.** Los vendedores recomendados no son más caros ni están más lejos "
        "que el resto de los candidatos.\n"
        "- **Pensado para clientes nuevos.** El 96,9 % de los clientes de Olist compra una sola vez. El "
        "recomendador no necesita su historial: elige según la categoría y el estado de destino.\n"
        "- **Un modelo, dos usos.** La misma señal de riesgo que marca qué pedidos vigilar después del pago "
        "ordena a los vendedores antes de la compra. Olist mantiene un solo modelo.\n"
        "- **Margen para crecer.** Con más meses de datos y la información de la transportadora, la señal "
        "puede afinarse todavía más."
    )


def _headline(rates: dict, labels: dict, rec: dict) -> str:
    """La frase principal: el modelo contra elegir al azar, y si fue el de menor retraso."""
    model, random_pick = CRITERIOS["clasificador"], CRITERIOS["azar"]
    if model not in rates or random_pick not in rates:
        return ""
    text = (
        f"Los {TOP_K} vendedores que recomienda el modelo se atrasaron un {labels[model]} de las veces, contra "
        f"un {labels[random_pick]} de un vendedor elegido al azar en la misma lista"
    )
    if es_significativa(rec["vs_random"]) and rates[random_pick] > 0:
        cut = 1 - rates[model] / rates[random_pick]
        text += f": <b>{round(100 * cut)} % menos retrasos</b> para el cliente."
    else:
        text += "."
    if min(rates, key=rates.get) == model:
        text += " De las cuatro formas de elegir que probamos, fue la de menor retraso."
    return text


def _comparison_chart(comparison: pd.DataFrame, p: dict) -> alt.LayerChart:
    """Barras horizontales con el eje desde cero, para no agrandar diferencias chicas."""
    data = comparison.assign(destacado=comparison["criterio"] == "Nuestro modelo")
    top = max(0.05, float(data["tasa"].max()) * 1.25)
    y = alt.Y(
        "criterio:N",
        sort=list(data["criterio"]),
        title=None,
        axis=alt.Axis(labelColor=p["text"], labelFontSize=13, labelLimit=260, domain=False, ticks=False),
    )
    x = alt.X(
        "tasa:Q",
        title="Tasa media de retraso",
        scale=alt.Scale(domain=[0, top]),
        axis=alt.Axis(format="%", labelColor=p["muted"], titleColor=p["muted"], gridColor=p["line"], tickCount=5),
    )
    base = alt.Chart(data).encode(y=y)
    bars = base.mark_bar(size=CHART_ROW - 12, cornerRadius=4).encode(
        x=x,
        color=alt.condition(alt.datum.destacado, alt.value(p["accent"]), alt.value(p["muted"])),
        tooltip=[alt.Tooltip("criterio:N", title="Forma de elegir"), alt.Tooltip("etiqueta:N", title="Tasa media")],
    )
    labels = base.mark_text(align="left", dx=6, color=p["text"], fontSize=13, fontWeight="bold").encode(
        x="tasa:Q", text="etiqueta:N"
    )
    return (
        (bars + labels)
        .properties(height=alt.Step(CHART_ROW), background="transparent")
        .configure_view(stroke=None)
    )


def show_technical(rec: dict) -> None:
    """Métricas y argumentos para quien quiera el respaldo de la propuesta."""
    dispatch = rec["dispatch"]
    with st.expander("Respaldo técnico: cómo lo medimos y por qué elegir este modelo"):
        st.markdown(
            f"**Diseño.** En cada combinación de categoría y estado del cliente con al menos {2 * TOP_K} candidatos "
            f"medibles se muestran {TOP_K} vendedores con cada criterio. Un vendedor es *bueno* si en test se retrasó "
            "menos que la tasa media posterior. Precision@K: qué parte de los mostrados es buena; Recall@K: qué parte "
            "de los buenos aparece entre los mostrados; tasa media: promedio por vendedor y luego por combinación."
        )
        st.dataframe(tabla_metricas(rec["metrics"]), width="stretch")
        st.markdown(_technical_result(rec))
        st.markdown(
            "**Por qué un modelo y no una regla fija.**\n"
            "- **Explica el riesgo, no solo lo cuenta.** El puntaje resume cómo son los pedidos del vendedor: plazo "
            "prometido, distancia al cliente, si vende fuera de su estado, peso y flete. Una regla como «el que más "
            "vende» o «el que menos se atrasó» no dice por qué.\n"
            "- **Reutiliza lo que ya está construido.** Es el mismo clasificador calibrado que prioriza pedidos para "
            "logística; no hay un segundo modelo que entrenar ni mantener.\n"
            "- **Escala con los datos.** Cada variable nueva que sume Olist (transportadora, depósito del vendedor, "
            "capacidad de despacho) mejora a la vez la alerta de pedidos y la recomendación."
        )
        st.markdown(
            "**Despacho y retraso (diagnóstico, no entra al modelo).** De los pedidos que llegaron tarde, el "
            f"{format_pct(dispatch['pct_retrasos_con_despacho_tarde'])} tuvo un vendedor que entregó a la "
            "transportadora después del plazo límite registrado. En el resto, el despacho figura dentro del plazo: "
            "ahí es donde los datos de la transportadora (modalidad, tramos, eventos de entrega) más pueden sumar. "
            f"Con despacho tarde, el pedido llegó tarde el "
            f"{format_pct(dispatch['retraso_si_despacho_tarde'])} de las veces; con despacho a tiempo, el "
            f"{format_pct(dispatch['retraso_si_despacho_a_tiempo'])}."
        )
        st.markdown(
            "**Próximos pasos.**\n"
            "- Medir el efecto de recomendar con una prueba A/B: hoy la comparación es sobre datos históricos.\n"
            "- Sumar datos de la transportadora, para explicar los retrasos en que el vendedor despachó a tiempo.\n"
            "- Ampliar el período de evaluación: el actual es de pocos meses y con pocos retrasos por vendedor.\n"
            "- Incorporar la oferta completa (qué vendedores ofrecían el mismo producto y a qué precio), "
            "porque hoy solo vemos lo vendido.\n\n"
            "Detalle y cifras en `docs/recomendador.md` (`python -m src.recomendador` las reproduce)."
        )


def _technical_result(rec: dict) -> str:
    """Dónde queda el modelo en la tabla y la mejora frente al azar, solo con lo que los datos sostienen."""
    metrics = rec["metrics"]
    lines = []
    if "precision" in metrics.columns and metrics["precision"].idxmax() == "clasificador":
        lines.append(f"la mayor Precision@{TOP_K} ({format_pct(metrics.loc['clasificador', 'precision'])})")
    if "retraso" in metrics.columns and metrics["retraso"].idxmin() == "clasificador":
        lines.append(f"la menor tasa de retraso de los mostrados ({format_pct(metrics.loc['clasificador', 'retraso'])})")
    text = f"**Resultado.** Nuestro modelo obtuvo {' y '.join(lines)} de los cuatro criterios. " if lines else ""
    text += f"Frente a elegir al azar, la tasa de retraso cambia {texto_diferencia(rec['vs_random'])}, por bootstrap de combinaciones"
    if es_significativa(rec["vs_random"]):
        text += ": el intervalo no incluye el cero, la mejora es estadísticamente significativa."
    else:
        text += "."
    return text

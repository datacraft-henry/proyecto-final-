"""App de riesgo de retraso por pedido, para el equipo de operaciones.

Uso, desde la raíz del repo, con los CSV en data/raw y el modelo generado
(`python -m src.umbral`):
    streamlit run app/streamlit_app.py

Arma los pedidos desde los CSV de Kaggle (período de test, los meses que el
modelo no vio). Publicada en Streamlit Community Cloud, donde no están ni los
CSV ni el modelo, lee los archivos ya puntuados de app/datos_web/ (modo web,
ver app/datos_web.py).

La lógica de negocio vive en app/riesgo.py, los helpers de pantalla en
app/vista.py (pedidos) y app/vendedores.py (vendedores recomendados) y las
paletas y el CSS en app/estilos.py: este archivo compone la pantalla. El modo
claro u oscuro vive en `st.session_state.theme_mode`.

Tiene dos pestañas: los pedidos a revisar (operaciones, en este archivo) y los
vendedores recomendados según el riesgo que predice el clasificador
(src/recomendador.py, compuesta en app/pestana_vendedores.py). Solo corre la
pestaña abierta.
"""

import json
import sys
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # para importar src y app

from app.datos_web import contributions_of, load_artifact_info, load_scored_orders, web_mode  # noqa: E402
from app.estilos import PALETTES, build_css  # noqa: E402
from app.riesgo import (  # noqa: E402
    SEMAFORO,
    factors_table,
    format_feature_value,
    load_artifact,
    orders_with_risk,
    risk_factors,
    summary_by_level,
)
from app.pestana_vendedores import show_sellers_page  # noqa: E402
from app.vista import (  # noqa: E402
    LEVEL_ALL,
    ROW_HEIGHT,
    SCORE_SCALE,
    filter_orders,
    format_factor,
    format_int,
    format_pct,
    impact_weights,
    late_summary,
    lift_high,
    main_view,
    one_in,
    open_order,
    order_exists,
    plural,
    table_height,
    table_view,
)
from src.calibration import LEVELS  # noqa: E402
from src.data_loading import load_raw_tables  # noqa: E402
from src.feature_engineering import build_model_dataset, load_geolocation  # noqa: E402
from src.features import PREDICTION_TIME_COL  # noqa: E402
from src.target import TARGET_COL  # noqa: E402
from src.validation import temporal_three_way_split  # noqa: E402

LEVEL_INFO = {
    "alto": ("Riesgo alto", "Revisar hoy."),
    "medio": ("Riesgo medio", "Seguir de cerca."),
    "bajo": ("Riesgo bajo", "Seguimiento habitual."),
}
# Qué hacer con el pedido abierto. Son pasos del proceso de revisión, no causas: el modelo
# no sabe por qué se retrasaría un pedido.
LEVEL_STEPS = {
    "alto": "Revisarlo hoy: verificar con el vendedor el estado del despacho y, si no llega a la fecha prometida, avisar al cliente.",
    "medio": "Seguirlo de cerca: volver a mirarlo antes de la fecha prometida.",
    "bajo": "Seguimiento habitual: no requiere una revisión especial.",
}
FACTORS_UP, FACTORS_DOWN = 3, 2
ARROWS = {"up": "↑", "down": "↓"}
CHART_STEP, CHART_BAR = 40, 28  # alto de cada fila del gráfico y de la barra dentro de la fila
OPEN_MARK = "▶"  # marca, en la tabla, la fila del pedido abierto en el inspector
TABLE_HEIGHT = 600  # tope de la tabla de pedidos, cerca del alto del inspector para no dejar hueco
COPY_LABEL, COPIED_LABEL = "📋 Copiar ID del pedido", "✅ ¡ID copiado!"
DEFAULT_THEME = "light"
TAB_ORDERS, TAB_SELLERS = "🚚 Pedidos a revisar", "🏪 Vendedores recomendados"
# Qué ofrece el botón de la cabecera en cada modo: el modo al que se pasa.
THEME_SWITCH = {"light": ("🌙 Oscuro", "Cambiar a modo oscuro"), "dark": ("☀️ Claro", "Cambiar a modo claro")}


# ----------------------------------------------------------------------------- datos


@st.cache_resource
def get_artifact() -> dict:
    """El modelo calibrado; en modo web, solo con qué se entrenó y los cortes de riesgo."""
    return load_artifact_info() if web_mode() else load_artifact()


@st.cache_data(show_spinner="Preparando los pedidos...")
def get_orders() -> pd.DataFrame:
    """Pedidos del período de test (los que el modelo no vio), con todas sus variables."""
    if web_mode():
        return load_scored_orders()
    tables = {**load_raw_tables(), "geolocation": load_geolocation()}
    dataset = build_model_dataset(tables, get_artifact()["anchor"])
    _, _, test = temporal_three_way_split(dataset)
    return test.reset_index(drop=True)


@st.cache_data
def get_scored_orders() -> pd.DataFrame:
    if web_mode():
        return get_orders()  # ya vienen puntuados y ordenados
    return orders_with_risk(get_artifact(), get_orders())


def get_factors(orders: pd.DataFrame, order_id: str) -> pd.DataFrame:
    """Qué sube y qué baja el puntaje de un pedido: con el modelo o, en modo web, ya calculado."""
    if web_mode():
        return factors_table(*contributions_of(orders, order_id))
    return risk_factors(get_artifact(), orders, order_id)


# ----------------------------------------------------------------------------- tema


def palette() -> dict:
    return PALETTES[st.session_state.get("theme_mode", DEFAULT_THEME)]


def _toggle_theme() -> None:
    st.session_state.theme_mode = "dark" if st.session_state.theme_mode == "light" else "light"


def sync_streamlit_theme(mode: str) -> bool:
    """Lleva el tema nativo de Streamlit (grilla, inputs, botones, expander) al modo elegido.

    Las opciones `theme.*` viajan al navegador al inicio de cada corrida, así que
    cuando cambian hace falta una corrida más. Devuelve si cambió algo.
    """
    changed = False
    for option, value in PALETTES[mode]["streamlit"].items():
        if st._config.get_option(f"theme.{option}") != value:
            st._config.set_option(f"theme.{option}", value)
            changed = True
    return changed


def apply_theme() -> None:
    """Aplica el modo de la sesión. Las opciones `theme.*` son del servidor (compartidas entre
    pestañas), así que cada corrida las vuelve a alinear con la sesión y, si cambiaron, corre
    una vez más para que el navegador reciba el tema nuevo; nunca dos veces seguidas."""
    mode = st.session_state.setdefault("theme_mode", DEFAULT_THEME)
    changed = sync_streamlit_theme(mode)
    if changed and not st.session_state.get("theme_rerun_pending"):
        st.session_state["theme_rerun_pending"] = True
        st.rerun()
    st.session_state["theme_rerun_pending"] = False


# ----------------------------------------------------------------------------- filtros


def _reset_filters(seed: dict) -> None:
    """Vuelve a crear los filtros con estos valores: los widgets cambian de clave y arrancan de cero."""
    st.session_state["seed"] = seed
    st.session_state["gen"] = st.session_state.get("gen", 0) + 1


def _widget_key(name: str) -> str:
    return f"{name}-{st.session_state.get('gen', 0)}"


def _seed() -> dict:
    return st.session_state.get("seed", {})


def _clear_search() -> None:
    """Borra lo buscado y conserva el período y el nivel elegidos."""
    keep = {name: st.session_state.get(_widget_key(name)) for name in ("period", "level")}
    _reset_filters({name: value for name, value in keep.items() if value is not None})


def show_header() -> None:
    """Marca a la izquierda y, en la misma barra, el botón de modo claro / oscuro."""
    p = palette()
    st.markdown(build_css(p), unsafe_allow_html=True)
    with st.container(key="dc-header"):
        brand, switch = st.columns([6, 1.2], vertical_alignment="center")
        brand.markdown(
            '<div class="dc-brand"><div class="dc-logo">DataCraft<span>.</span></div>'
            '<div class="dc-sub">Riesgo de retraso por pedido · Demostración con datos históricos de Olist</div></div>',
            unsafe_allow_html=True,
        )
        label, tip = THEME_SWITCH[p["name"]]
        switch.button(label, key="theme_toggle", help=tip, on_click=_toggle_theme, width="stretch")


def show_toolbar(scored: pd.DataFrame) -> tuple[pd.DataFrame, str, bool, tuple]:
    """Filtros globales en una barra: período, búsqueda y dato histórico."""
    dates = scored[PREDICTION_TIME_COL].dt.date
    full_period = (dates.min(), dates.max())
    with st.container(key="dc-toolbar"):
        st.markdown('<div class="dc-toolbar-title">Filtros</div>', unsafe_allow_html=True)
        left, middle, right = st.columns([1.7, 2.7, 1.2], vertical_alignment="bottom")
        period = left.date_input(
            "Pedidos aprobados entre",
            _seed().get("period", full_period),
            min_value=full_period[0],
            max_value=full_period[1],
            format="DD/MM/YYYY",
            key=_widget_key("period"),
        )
        search_col, find_col, clear_col = middle.columns([2.8, 0.9, 1.1], vertical_alignment="bottom")
        search = search_col.text_input(
            "Buscar un pedido",
            _seed().get("search", ""),
            placeholder="Pegá el id y apretá Enter",
            key=_widget_key("search"),
        )
        # El text_input ya busca al apretar Enter; el botón solo vuelve a correr la
        # pantalla, para quien espera un botón al lado del campo.
        find_col.button("Buscar", key="find_order", type="primary", help="O apretá Enter en el campo")
        if search.strip():
            clear_col.button(
                "✕ Limpiar",
                key="clear_search",
                type="tertiary",
                help="Borra el código y deja el período y el nivel como están",
                on_click=_clear_search,
            )
        show_real = right.toggle(
            "Mostrar si se retrasó (histórico)",
            key="show_real",
            help="Resultado real de estos pedidos ya entregados. En un pedido nuevo todavía no se sabe.",
        )
    start, end = period if len(period) == 2 else (period[0], period[0])
    return scored[(dates >= start) & (dates <= end)], search.strip(), show_real, full_period


# ----------------------------------------------------------------------------- resumen


def show_demo_note() -> None:
    """Qué es esta pantalla: una demostración con pedidos que ya se entregaron."""
    st.markdown(
        '<div class="dc-note"><b>Demostración con pedidos históricos.</b> Estos pedidos ya se entregaron. '
        "La lista muestra en qué orden se habrían revisado al aprobar cada pago: el puntaje usa solo datos "
        "conocidos en ese momento. Lo que pasó después se ve únicamente con «Mostrar si se retrasó» y en la "
        "evaluación histórica, al final de la página.</div>",
        unsafe_allow_html=True,
    )


def show_hero(summary: pd.DataFrame, n_orders: int) -> None:
    """Una sola frase: cuántos pedidos hay que revisar y sobre cuántos."""
    high = summary.loc["alto"]
    st.markdown(
        f'<p class="dc-hero">Hay <b>{format_int(high["pedidos"])}</b> pedidos en riesgo alto para revisar '
        f'<span class="dc-hero-sub">de {format_int(n_orders)} en el período elegido.</span></p>',
        unsafe_allow_html=True,
    )


def show_level_cards(summary: pd.DataFrame, show_real: bool) -> None:
    """Una tarjeta por nivel: cuántos son, qué parte del total y qué hacer. El dato real solo si se pidió."""
    colors = palette()["level"]
    for column, level in zip(st.columns(len(LEVELS)), LEVELS):
        row = summary.loc[level]
        title, action = LEVEL_INFO[level]
        late = (
            f" · histórico: se retrasó el {format_pct(row['tasa_retraso_real'])}"
            if show_real and "tasa_retraso_real" in summary.columns and row["pedidos"]
            else ""
        )
        column.markdown(
            f'<div class="dc-card" style="--lvl:{colors[level]}">'
            f'<div class="t">{SEMAFORO[level]} {title}</div>'
            f'<div class="n">{format_int(row["pedidos"])}</div>'
            f'<div class="p">{format_pct(row["pct_pedidos"])} de los pedidos{late}</div>'
            f'<div class="a">{action}</div></div>',
            unsafe_allow_html=True,
        )


def show_history(summary: pd.DataFrame) -> None:
    """Evaluación histórica: dónde cayeron los retrasos que ya ocurrieron. Siempre se rotula como tal,
    esté o no activado «Mostrar si se retrasó», porque usa el resultado real de los pedidos."""
    if "pct_retrasos" not in summary.columns:
        return
    with st.expander("📊 Evaluación histórica: ¿el nivel alto concentró los retrasos?"):
        st.markdown(
            '<p class="dc-muted">Usa el resultado real de estos pedidos, que ya se entregaron. '
            "Sirve para saber cuánto ayudó ordenar así en el pasado; en un pedido nuevo ese dato todavía no existe.</p>",
            unsafe_allow_html=True,
        )
        chart_col, lift_col = st.columns([3, 1.15], gap="medium", vertical_alignment="center")
        with chart_col:
            st.altair_chart(_share_chart(summary), width="stretch")
            st.caption(
                "La parte roja de los retrasos es más grande que la de los pedidos: el nivel alto concentró retrasos."
            )
        with lift_col:
            show_lift(summary)


def show_lift(summary: pd.DataFrame) -> None:
    lift = lift_high(summary)
    if lift is None:
        return
    high = summary.loc["alto"]
    st.markdown(
        f'<div class="dc-lift"><div class="n">{format_factor(lift)}</div>'
        '<div class="t">más retrasos que revisando al azar (histórico)</div>'
        f'<div class="s">{SEMAFORO["alto"]} Revisando {one_in(high["pct_pedidos"])} pedidos (los rojos) '
        f'aparece {one_in(high["pct_retrasos"])} de todos los pedidos que se retrasaron.</div></div>',
        unsafe_allow_html=True,
    )


def _share_chart(summary: pd.DataFrame) -> alt.LayerChart:
    p = palette()
    rows = [
        {"grupo": group, "nivel": LEVEL_INFO[level][0], "orden": i, "parte": summary.loc[level, column]}
        for group, column in [("Pedidos", "pct_pedidos"), ("Retrasos", "pct_retrasos")]
        for i, level in enumerate(LEVELS)
    ]
    data = pd.DataFrame(rows)
    data["fin"] = data.groupby("grupo")["parte"].cumsum()
    data["inicio"] = data["fin"] - data["parte"]
    data["etiqueta"] = data["parte"].map(format_pct)
    color = alt.Color(
        "nivel:N",
        scale=alt.Scale(domain=[LEVEL_INFO[l][0] for l in LEVELS], range=[p["level"][l] for l in LEVELS]),
        legend=alt.Legend(title=None, orient="top", labelColor=p["text"], labelFontSize=13),
    )
    y = alt.Y(
        "grupo:N",
        title=None,
        sort=["Pedidos", "Retrasos"],
        axis=alt.Axis(labelColor=p["text"], labelFontSize=13, domain=False, ticks=False, labelPadding=8),
    )
    x = alt.X("inicio:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 1]))
    base = alt.Chart(data).encode(y=y)
    bars = base.mark_bar(size=CHART_BAR, stroke=p["bg"], strokeWidth=2, cornerRadius=4).encode(
        x=x,
        x2="fin:Q",
        color=color,
        tooltip=[
            alt.Tooltip("grupo:N", title=""),
            alt.Tooltip("nivel:N", title="Nivel"),
            alt.Tooltip("etiqueta:N", title="Parte"),
        ],
    )
    labels = (
        base.transform_filter(alt.datum.orden == 0)
        .mark_text(align="left", dx=8, color=p["on_bar"], fontWeight="bold", fontSize=13)
        .encode(x=x, text="etiqueta:N")
    )
    # Línea punteada que une el final de la parte roja de una barra con el de la otra, cruzando
    # solo el hueco entre ambas: con filas de CHART_STEP píxeles, el borde inferior de la barra
    # de arriba y el superior de la de abajo quedan en posiciones fijas.
    high = summary.loc["alto"]
    gap = (CHART_STEP - CHART_BAR) / 2
    link = pd.DataFrame({"x": [high["pct_pedidos"]], "x2": [high["pct_retrasos"]]})
    connector = (
        alt.Chart(link)
        .mark_rule(color=p["level"]["alto"], strokeDash=[4, 3], strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
            x2="x2:Q",
            y=alt.value(CHART_STEP - gap),
            y2=alt.value(CHART_STEP + gap),
        )
    )
    chart = (bars + labels + connector).properties(height=alt.Step(CHART_STEP), background="transparent")
    return chart.configure_view(stroke=None)


# ----------------------------------------------------------------------------- tabla


def level_picker(summary: pd.DataFrame) -> str:
    """Un solo nivel activo (o todos), con la cantidad de pedidos de cada uno."""
    counts = {LEVEL_ALL: int(summary["pedidos"].sum()), **{l: int(summary.loc[l, "pedidos"]) for l in LEVELS}}
    labels = {LEVEL_ALL: "Todos", **{l: f"{SEMAFORO[l]} {LEVEL_INFO[l][0]}" for l in LEVELS}}
    picked = st.segmented_control(
        "Nivel de riesgo",
        [LEVEL_ALL, *LEVELS],
        default=_seed().get("level", "alto"),
        selection_mode="single",
        required=True,
        format_func=lambda l: f"{labels[l]} ({format_int(counts[l])})",
        key=_widget_key("level"),
        label_visibility="collapsed",
    )
    return picked or LEVEL_ALL


def show_table_bar(
    in_period: pd.DataFrame, summary: pd.DataFrame, search: str, show_real: bool
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Selector de nivel y descarga en la misma línea, sobre la tabla. El CSV lleva todas las columnas."""
    with st.container(key="dc-tablebar"):
        left, right = st.columns([3.2, 1], vertical_alignment="center")
        with left:
            level = level_picker(summary)
        listed = filter_orders(in_period, level, search).reset_index(drop=True)
        full = table_view(listed, show_real)
        right.download_button(
            "Descargar lista (CSV)",
            full.to_csv(index=False).encode("utf-8-sig"),
            file_name="pedidos_a_revisar.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
            disabled=listed.empty,
            help="Todas las columnas de la lista: aprobación, distancia, precio y flete incluidos.",
        )
    return listed, full, level


# La tabla principal tiene pocas columnas para que se lea bien; el resto está en el inspector y en el CSV.
TABLE_COLUMNS = {
    " ": st.column_config.TextColumn(width=28, help="Pedido abierto en el inspector"),
    "Prioridad": st.column_config.TextColumn(width=86),
    "Puntaje": st.column_config.ProgressColumn(
        format="%d", min_value=0, max_value=SCORE_SCALE, width=86, help="De 0 a 100. Sirve para ordenar la revisión: más alto, antes."
    ),
    "Pedido": st.column_config.TextColumn(width=280, help="Código completo del pedido"),
    "Destino": st.column_config.TextColumn(width=70, help="Estado del cliente"),
    "Plazo prometido": st.column_config.NumberColumn(format="%d días", width=110),
    "¿Se retrasó? (histórico)": st.column_config.TextColumn(width=150),
}


def show_table(listed: pd.DataFrame, full: pd.DataFrame, show_real: bool, table_key: str) -> tuple[str, str, int]:
    """Tabla seleccionable. Devuelve el pedido abierto, por qué está abierto y su fila en la lista.

    La selección se guarda por código de pedido (`open_order`), así sobrevive a ordenar la
    tabla, filtrar, buscar o cambiar de nivel mientras el pedido siga en la lista.
    """
    previous = st.session_state.get(table_key)
    rows = previous["selection"]["rows"] if previous else []
    order_id, reason = open_order(listed, rows[0] if rows else None, st.session_state.get("open_order"))
    st.session_state["open_order"] = order_id
    st.dataframe(
        main_view(listed, show_real, order_id, OPEN_MARK),
        hide_index=True,
        width="stretch",
        height=table_height(len(listed), TABLE_HEIGHT),  # alto parejo con el inspector; se achica con pocas filas
        row_height=ROW_HEIGHT,
        on_select="rerun",
        selection_mode="single-row",
        column_config=TABLE_COLUMNS,
        key=table_key,
    )
    st.caption(
        f"{plural(len(listed), 'pedido')}, del más riesgoso al menos riesgoso. {late_summary(full)}"
        f"Marcá la casilla de una fila para abrirla en el inspector ({OPEN_MARK} = pedido abierto). "
        "Fecha de aprobación, distancia, precio y flete están en el inspector y en el CSV."
    )
    position = int((listed["order_id"] == order_id).to_numpy().argmax())
    return order_id, reason, position


def show_empty_state(scored: pd.DataFrame, search: str, full_period: tuple) -> None:
    exists = order_exists(scored, search)
    if exists:
        title = "El pedido existe, pero está fuera de los filtros actuales"
        sub = "Está en otro período o en otro nivel de riesgo. Podés verlo quitando los filtros."
    elif search:
        title = "No encontramos ningún pedido con ese código"
        sub = "Revisá que el código esté bien copiado, o limpiá la búsqueda para volver a la lista."
    else:
        title = "No hay pedidos con esos filtros"
        sub = "Probá con otro período u otro nivel de riesgo."
    st.markdown(
        f'<div class="dc-empty"><div class="i">🔎</div><div class="t">{title}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )
    with st.container(horizontal=True, horizontal_alignment="center"):
        if exists:
            st.button(
                "Ver el pedido sin filtros",
                type="primary",
                icon=":material/filter_alt_off:",
                on_click=_reset_filters,
                args=({"search": search, "level": LEVEL_ALL, "period": full_period},),
            )
        st.button(
            "Limpiar filtros",
            type="secondary" if exists else "primary",
            icon=":material/restart_alt:",
            on_click=_reset_filters,
            args=({"period": full_period},),
        )


# ----------------------------------------------------------------------------- inspector


def show_detail(
    order_id: str, scored: pd.DataFrame, orders: pd.DataFrame, reason: str, position: int, n_listed: int, show_real: bool
) -> None:
    """Primero el pedido abierto, su prioridad y qué hacer; después los datos y las señales del puntaje."""
    p = palette()
    order = scored.loc[scored["order_id"] == order_id].iloc[0]
    level = order["nivel_riesgo"]
    title, _ = LEVEL_INFO[level]
    colors = f'--lvl:{p["level"][level]};--soft:{p["level_soft"][level]};--ink:{p["level_text"][level]}'
    where = f"fila {format_int(position + 1)} de {format_int(n_listed)}"
    hint = {
        "fila": f"Elegido en la tabla · {where}",
        "antes": f"Elegido antes · sigue en la lista, {where}",
        "primero": "El primero de la lista · elegí otra fila para cambiarlo",
    }[reason]
    with st.container(key="dc-inspector"):
        st.markdown(
            '<div class="dc-eyebrow">Pedido abierto</div>'
            f'<div class="dc-hint{"" if reason == "primero" else " picked"}" role="status">{hint}</div>'
            f'<div class="dc-oid">{escape(order_id)}</div>'
            f'<div class="dc-level" style="{colors}"><span class="dc-badge">{SEMAFORO[level]} {title}</span>'
            f'<span class="dc-score">Puntaje <b>{round(order["prob_retraso"] * SCORE_SCALE)}</b>/{SCORE_SCALE}</span></div>'
            '<div class="dc-fx-note">El puntaje sirve para ordenar la revisión; no es la probabilidad exacta de retraso.</div>'
            f'<div class="dc-action" style="{colors}"><div class="k">Acción sugerida</div>'
            f'<div class="t">{escape(LEVEL_STEPS[level])}</div></div>',
            unsafe_allow_html=True,
        )
        facts = [
            ("Aprobado", f"{order[PREDICTION_TIME_COL]:%d/%m/%Y %H:%M}"),
            ("Destino", order["customer_state"]),
            ("Plazo prometido", format_feature_value("promised_days", order["promised_days"])),
            ("Distancia", format_feature_value("max_distance_km", order["max_distance_km"])),
            ("Precio", format_feature_value("total_price", order["total_price"])),
            ("Flete", format_feature_value("total_freight", order["total_freight"])),
        ]
        if show_real and TARGET_COL in order.index:
            facts.append(("Resultado (histórico)", "Se retrasó" if order[TARGET_COL] == 1 else "Llegó a tiempo"))
        st.markdown(
            '<div class="dc-facts dc-facts-grid">'
            + "".join(f"<div><small>{label}</small><b>{escape(str(value))}</b></div>" for label, value in facts)
            + "</div>",
            unsafe_allow_html=True,
        )
        up, down = impact_weights(get_factors(orders, order_id), FACTORS_UP, FACTORS_DOWN)
        with st.expander("Qué sube y qué baja el puntaje"):
            _factor_block("Suben el puntaje", "frente al pedido promedio", up, "up")
            if not down.empty:
                _factor_block("Bajan el puntaje", "frente al pedido promedio", down, "down")
            st.markdown(
                '<div class="dc-fx-note">Son las variables que más pesan en el modelo para este pedido. '
                "No indican la causa de un retraso.</div>",
                unsafe_allow_html=True,
            )
        _copy_button(order_id)


def _factor_block(title: str, meaning: str, factors: pd.DataFrame, kind: str) -> None:
    """Chips con flecha y mini barra de impacto: se lee de un vistazo qué pesa más."""
    arrow = ARROWS[kind]
    rows = "".join(
        f'<div class="dc-fx dc-fx-{kind}" title="{meaning}">'
        f'<div class="dc-fx-row"><span class="dc-chip"><i>{arrow}</i>{escape(row.etiqueta)}</span>'
        f'<span class="dc-fx-val">{escape(row.valor)}</span></div>'
        f'<div class="dc-fx-bar"><i style="width:{row.peso:.0%}"></i></div></div>'
        for row in factors.itertuples()
    )
    st.markdown(
        f'<div class="dc-sec"><span class="dc-sec-t">{title}</span>'
        f'<span class="dc-sec-m dc-sec-{kind}">{arrow} {meaning}</span></div>{rows}',
        unsafe_allow_html=True,
    )


def _copy_button(order_id: str) -> None:
    """Botón secundario que copia el código al portapapeles (JS, en un iframe propio)."""
    p = palette()
    st.iframe(
        f"""
<style>
  body {{ margin: 0; padding-top: 6px; font-family: "Source Sans Pro", "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif; }}
  button {{ width: 100%; height: 38px; border-radius: 8px; border: 1px solid {p["muted"]}; background: {p["surface"]};
    color: {p["text"]}; font-size: 14px; font-weight: 600; cursor: pointer; transition: background .15s ease, transform .1s ease; }}
  button:hover {{ background: {p["hint_bg"]}; border-color: {p["text"]}; }}
  button:active {{ transform: scale(.985); }}
  button.ok {{ background: {p["level_soft"]["bajo"]}; border-color: {p["level"]["bajo"]}; color: {p["level_text"]["bajo"]}; }}
</style>
<button id="copy" type="button">{COPY_LABEL}</button>
<script>
  const orderId = {json.dumps(order_id)};
  const button = document.getElementById("copy");
  const done = () => {{
    button.textContent = {json.dumps(COPIED_LABEL)};
    button.classList.add("ok");
    setTimeout(() => {{ button.textContent = {json.dumps(COPY_LABEL)}; button.classList.remove("ok"); }}, 1600);
  }};
  const fallback = () => {{
    const area = document.createElement("textarea");
    area.value = orderId;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
    done();
  }};
  button.addEventListener("click", () => {{
    if (navigator.clipboard && window.isSecureContext) {{
      navigator.clipboard.writeText(orderId).then(done).catch(fallback);
    }} else {{
      fallback();
    }}
  }});
</script>
""",
        height=48,
    )


# ----------------------------------------------------------------------------- página


def show_help() -> None:
    with st.expander("Cómo leer esta pantalla"):
        st.markdown(
            "- **El nivel es lo importante.** Alto, medio y bajo salen de ordenar los pedidos por riesgo: "
            "el alto se pensó para el 5 % más riesgoso, y en estos meses marca cerca del 10 %.\n"
            "- **El puntaje sirve para ordenar, no es una probabilidad exacta.** En los meses nuevos sale "
            "más alto que la tasa real de retrasos.\n"
            "- **El detalle muestra qué variables pesan** en ese pedido frente al promedio, según el modelo. "
            "No quiere decir que esa sea la causa del retraso.\n"
            "- El modelo usa solo datos conocidos al aprobar el pago. Detalle en `docs/metricas_evaluacion.md`."
        )


def show_orders_page() -> None:
    """Pestaña de operaciones: resumen compacto, lista con inspector y, al final, la evaluación histórica."""
    try:
        orders, scored = get_orders(), get_scored_orders()
    except FileNotFoundError as error:
        st.error(f"No se pudo cargar la app. {error}")
        st.stop()

    in_period, search, show_real, full_period = show_toolbar(scored)
    show_demo_note()
    summary = summary_by_level(in_period)
    show_hero(summary, len(in_period))
    show_level_cards(summary, show_real)

    st.markdown("### Pedidos a revisar")
    listed, full, level = show_table_bar(in_period, summary, search, show_real)
    if listed.empty:
        show_empty_state(scored, search, full_period)
    else:
        dates = in_period[PREDICTION_TIME_COL]
        gen = st.session_state.get("gen", 0)
        table_key = f"orders-{gen}-{level}-{search}-{dates.min():%Y%m%d}-{dates.max():%Y%m%d}"
        table_col, detail_col = st.columns([2.2, 1], gap="small")
        with table_col:
            order_id, reason, position = show_table(listed, full, show_real, table_key)
        with detail_col:
            show_detail(order_id, scored, orders, reason, position, len(listed), show_real)
    show_history(summary)
    show_help()
    trained_with = get_artifact()["entrenado_con"].replace("validacion", "validación")
    st.caption(f"Modelo: regresión logística calibrada, entrenada con {trained_with}.")


def main() -> None:
    st.set_page_config(page_title="Riesgo de retraso · DataCraft", page_icon="🚚", layout="wide")
    apply_theme()
    show_header()
    # Con on_change="rerun" cada pestaña sabe si está abierta: solo se arma la que se ve.
    orders_tab, sellers_tab = st.tabs([TAB_ORDERS, TAB_SELLERS], key="dc-tabs", on_change="rerun")
    if orders_tab.open:
        with orders_tab:
            show_orders_page()
    if sellers_tab.open:
        with sellers_tab:
            show_sellers_page(palette(), get_artifact())


main()

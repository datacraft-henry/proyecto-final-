"""Prototipo de visión: vendedores ordenados por riesgo de retraso, antes de comprar.

Uso, desde la raíz del repo, con los 9 CSV en data/raw:
    streamlit run prototipos/recomendador_vendedores.py

Es una app aparte de la principal (app/streamlit_app.py): no la modifica ni usa el
modelo. La lógica y sus tests están en src/recomendador.py y tests/test_recomendador.py;
el detalle, en docs/recomendador.md. Todas las cifras se calculan al abrir la app.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loading import RAW_DIR, load_raw_tables  # noqa: E402
from src.recomendador import CORTE, MIN_PEDIDOS, PESO_SUAVIZADO, preparar  # noqa: E402

ESTADOS = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia", "CE": "Ceará",
    "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
    "MG": "Minas Gerais", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso", "PA": "Pará",
    "PB": "Paraíba", "PE": "Pernambuco", "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte", "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul",
    "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins",
}

DATOS_NECESARIOS = pd.DataFrame(
    [
        ("Transportadora y modalidad de envío de cada pedido", "Separar el retraso del vendedor del retraso del transporte"),
        ("Eventos de seguimiento con fecha (recolección, centros de distribución, intento de entrega)", "Saber en qué tramo de la ruta se pierde el tiempo"),
        ("Catálogo de ofertas: qué vendedores ofrecían cada producto, con precio y stock", "Recomendar entre alternativas reales, no solo entre los que vendieron"),
        ("Ubicación real del depósito del vendedor", "Calcular la distancia desde donde sale el paquete"),
        ("Tiempo de preparación declarado y capacidad del vendedor", "Anticipar demoras de despacho en picos como Black Friday"),
        ("Qué vendedores vio el cliente y cuál eligió", "Medir el efecto de recomendar con una prueba A/B"),
        ("Pedidos posteriores a 2018, con reclamos y cancelaciones", "Validar que el orden sigue vigente e incluir pedidos que nunca llegaron"),
    ],
    columns=["Dato", "Para qué"],
)


def pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.1f} %".replace(".", ",")


def reales(x: float) -> str:
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data(show_spinner="Armando los pedidos...")
def cargar():
    tablas = load_raw_tables(RAW_DIR)
    candidatos, tercios, resumen = preparar(tablas)
    traduccion = pd.read_csv(Path(RAW_DIR) / "product_category_name_translation.csv")
    nombres = dict(zip(traduccion["product_category_name"], traduccion["product_category_name_english"]))
    candidatos["categoria"] = (
        candidatos["product_category_name"].map(nombres).fillna(candidatos["product_category_name"]).str.replace("_", " ")
    )
    estados_vendedor = tablas["sellers"].set_index("seller_id")["seller_state"]
    candidatos["seller_state"] = candidatos["seller_id"].map(estados_vendedor)
    return candidatos, tercios, resumen


st.set_page_config(page_title="Vendedores con menos retraso", layout="wide")
candidatos, tercios, r = cargar()
anio_hist, anio_post = CORTE.year - 1, CORTE.year

st.caption("DataCraft · Olist · Prototipo de visión")
st.title("¿A qué vendedor conviene comprarle?")
st.write(
    "El mismo riesgo de retraso que usa logística, mostrado antes de comprar: para una categoría y el "
    "estado del cliente, los vendedores ordenados de menor a mayor riesgo, con su precio."
)
st.warning(
    f"**Prototipo, no producto.** Es una asociación observacional con datos de {anio_hist}–{anio_post}: no prueba "
    "que recomendar a estos vendedores reduzca el retraso. Se validaría con una prueba A/B."
)

combos = candidatos.groupby(["categoria", "customer_state"])["seller_id"].nunique()
combos = combos[combos >= 2].reset_index()
categorias = sorted(combos["categoria"].unique())
c1, c2 = st.columns(2)
categoria = c1.selectbox(
    "Categoría del producto", categorias, index=categorias.index("bed bath table") if "bed bath table" in categorias else 0
)
estados = sorted(combos.loc[combos["categoria"] == categoria, "customer_state"], key=lambda s: ESTADOS.get(s, s))
estado = c2.selectbox(
    "Estado del cliente", estados, index=estados.index("RJ") if "RJ" in estados else 0,
    format_func=lambda s: f"{ESTADOS.get(s, s)} ({s})",
)

lista = candidatos[(candidatos["categoria"] == categoria) & (candidatos["customer_state"] == estado)].sort_values("puntaje")
mejor = lista.iloc[0]
m1, m2, m3 = st.columns(3)
m1.metric("Vendedores para elegir", len(lista))
m2.metric(f"Retraso del recomendado ({anio_hist})", pct(mejor["retraso_hist"]),
          help=f"Mediana de los candidatos: {pct(lista['retraso_hist'].median())}")
m3.metric("Precio mediano del recomendado", reales(mejor["precio_mediano"]),
          help=f"Mediana de los candidatos: {reales(lista['precio_mediano'].median())}")

tabla = pd.DataFrame({
    "Vendedor": lista["seller_id"].str[:8],
    "Desde": [ESTADOS.get(s, s) + (" (local)" if s == estado else "") for s in lista["seller_state"]],
    f"Retraso {anio_hist}": lista["retraso_hist"] * 100,
    f"Pedidos {anio_hist}": lista["pedidos_hist"],
    f"Retraso {anio_post}": lista["retraso_post"] * 100,
    f"Pedidos {anio_post}": lista["pedidos_post"].fillna(0).astype(int),
    "Precio mediano (R$)": lista["precio_mediano"],
    "Ventas aquí": lista["ventas_aqui"],
}).reset_index(drop=True)
tabla.index = tabla.index + 1
st.dataframe(
    tabla,
    width="stretch",
    column_config={
        f"Retraso {anio_hist}": st.column_config.ProgressColumn(
            format="%.1f %%", min_value=0, max_value=max(1.0, float(tabla[f"Retraso {anio_hist}"].max()))
        ),
        f"Retraso {anio_post}": st.column_config.NumberColumn(format="%.1f %%"),
        "Precio mediano (R$)": st.column_config.NumberColumn(format="%.2f"),
    },
)
st.caption(
    f"Orden: tasa de retraso de {anio_hist} suavizada hacia la media del año ({pct(r['media_hist'])}) con un peso de "
    f"{PESO_SUAVIZADO} pedidos, para no poner primero a un vendedor con 0 % en pocos pedidos. "
    f"“Retraso {anio_post}” es sobre todas las ventas del vendedor en {anio_post}. "
    "“Ventas aquí”: pedidos de esa categoría a ese estado."
)

st.subheader("¿El orden se sostiene en el tiempo?")
st.write(f"Vendedores ordenados por su retraso en {anio_hist}, medidos con sus pedidos de {anio_post}, que no se usaron para ordenarlos.")
st.dataframe(
    pd.DataFrame({
        f"Retraso en {anio_post}": tercios["retraso"].map(pct),
        "Ajustado por estado del cliente": tercios["ajustado_por_estado"].map(pct),
        "Precio mediano": tercios["precio_mediano"].map(reales),
        "Pedidos": tercios["pedidos"],
    }),
    width="stretch",
)

st.subheader("Límites, según los datos")
st.markdown(
    f"""
- **El vendedor explica solo una parte del retraso.** De los pedidos que llegaron tarde, solo el {pct(r['pct_retrasos_con_despacho_tarde'])} tuvo un vendedor que entregó a la transportadora después de su plazo límite: en el resto, el retraso ocurrió en el transporte. Cuando el vendedor despacha tarde, el pedido llega tarde el {pct(r['retraso_si_despacho_tarde'])} de las veces, contra {pct(r['retraso_si_despacho_a_tiempo'])} cuando despacha a tiempo.
- **Funciona en promedio, no vendedor por vendedor.** El orden por tercios se sostiene, pero un vendedor con 0 % en {anio_hist} puede tener más de 25 % en {anio_post}.
- **Cobertura parcial.** Solo el {pct(r['cobertura'])} de los pedidos de {anio_post} tiene un vendedor con al menos {MIN_PEDIDOS} pedidos en {anio_hist} ({r['n_vendedores']} vendedores). Los nuevos no se pueden ordenar.
- **No sabemos qué se ofrecía.** Los datos muestran lo que se vendió, no qué vendedores ofrecían el mismo producto, a qué precio ni con qué stock.
- **Es observacional.** Los clientes no eligieron vendedor al azar. Ajustar por el estado del cliente no cambia el resultado, pero puede haber otras diferencias, como el tipo de producto o la ruta.
- **Datos de {anio_hist}–{anio_post}.** No sabemos si el orden se mantiene con la operación actual.
"""
)

st.subheader("Qué datos necesitaríamos de Olist")
st.write("Hipótesis de lo que haría falta para pasar de prototipo a producto, y qué límite resuelve cada dato:")
st.dataframe(DATOS_NECESARIOS, hide_index=True, width="stretch")
st.caption(
    f"Base: pedidos de un solo vendedor ({r['pedidos_un_vendedor']:,} de {r['pedidos_total']:,}), "
    "con el target del proyecto (src/target.py).".replace(",", ".")
)

"""Pestaña de vendedores recomendados, sin Streamlit, para poder testearla.

La categoría y el estado del cliente solo filtran candidatos (quién ya vendió ahí). El
orden lo decide `src.recomendador.recomendar`: hoy, el puntaje medio que el clasificador
dio a TODOS los pedidos anteriores del vendedor, no una predicción para esa categoría y
destino. La pestaña muestra la lista tal como llega y no vuelve a ordenarla. Las columnas
que el recomendador no devuelva no se muestran.
"""

import pandas as pd

from app.categorias import CATEGORIAS_ES
from app.riesgo import SEMAFORO
from app.vista import SCORE_SCALE, format_int, format_pct
from src.calibration import assign_risk_level

TOP_K = 3  # vendedores recomendados por pantalla
MIN_CANDIDATOS = TOP_K + 1  # con K o menos candidatos se mostrarían todos: no hay nada que elegir

ESTADOS = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia", "CE": "Ceará",
    "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
    "MG": "Minas Gerais", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso", "PA": "Pará",
    "PB": "Paraíba", "PE": "Pernambuco", "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte", "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul",
    "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins",
}

# Criterios de src.recomendador.CRITERIOS, con el nombre que se ve en pantalla.
CRITERIOS = {
    "clasificador": "Nuestro modelo",
    "historial": "El que menos se atrasó antes",
    "popularidad": "El que más vende",
    "azar": "Uno cualquiera (al azar)",
}

COLUMNAS = {
    "ventas_aqui": "Ventas en este contexto",
    "precio_mediano": "Precio histórico de referencia",
    "pedidos_hist": "Pedidos anteriores",
}


def nombre_estado(sigla: str) -> str:
    """'RJ' -> 'Rio de Janeiro (RJ)'."""
    return f"{ESTADOS.get(sigla, sigla)} ({sigla})"


def nombre_categoria(categoria: str, traduccion: dict[str, str]) -> str:
    """Nombre en español ('cama_mesa_banho' -> 'Cama, mesa y baño'); la clave interna no cambia.

    Si la categoría no tiene nombre en español, usa la traducción al inglés de Kaggle sin
    guiones bajos y, si tampoco está, la clave misma.
    """
    if categoria in CATEGORIAS_ES:
        return CATEGORIAS_ES[categoria]
    return traduccion.get(categoria, categoria).replace("_", " ")


def combinaciones(ventas: pd.DataFrame, minimo: int = MIN_CANDIDATOS) -> pd.DataFrame:
    """Categoría y estado del cliente con al menos `minimo` vendedores para elegir.

    `ventas` tiene una fila por pedido con product_category_name, customer_state y seller_id,
    ya filtrada a lo que el recomendador puede usar (historial y vendedores con puntaje).
    """
    cuantos = ventas.groupby(["product_category_name", "customer_state"])["seller_id"].nunique()
    return cuantos[cuantos >= minimo].rename("vendedores").reset_index()


def categorias(combos: pd.DataFrame) -> list[str]:
    return sorted(combos["product_category_name"].unique())


def estados_de(combos: pd.DataFrame, categoria: str) -> list[str]:
    """Estados con vendedores para esa categoría, por nombre del estado."""
    siglas = combos.loc[combos["product_category_name"] == categoria, "customer_state"]
    return sorted(siglas, key=lambda s: ESTADOS.get(s, s))


def puntaje(prob: float | None) -> int | None:
    """Riesgo medio predicho en la escala de 0 a 100 de la app (0.074 -> 7). Sirve para ordenar."""
    return None if prob is None or pd.isna(prob) else round(prob * SCORE_SCALE)


def nivel_riesgo(probs, cortes: dict) -> list[str]:
    """Semáforo y nivel con los mismos cortes que los pedidos ('🟢 Bajo', '🟡 Medio', '🔴 Alto')."""
    return [f"{SEMAFORO[n]} {n.capitalize()}" for n in assign_risk_level(probs, cortes)]


def origen(sigla, estado: str) -> str:
    """Estado del vendedor, marcando si es el mismo que el del cliente."""
    if not isinstance(sigla, str):
        return "—"
    return ESTADOS.get(sigla, sigla) + (" (mismo estado)" if sigla == estado else "")


def tabla_vendedores(
    lista: pd.DataFrame, estado: str, estado_vendedor: pd.Series, cortes: dict | None = None, top_k: int = TOP_K
) -> pd.DataFrame:
    """Tabla para pantalla, en el orden del recomendador (`lista` indexada por seller_id).

    Marca los primeros `top_k`, deja el código completo del vendedor y muestra el riesgo como
    nivel (si se pasan los `cortes` del modelo) y puntaje de 0 a 100, igual que en pedidos.
    """
    vista = pd.DataFrame(
        {
            "#": range(1, len(lista) + 1),
            "": ["⭐" if i < top_k else "" for i in range(len(lista))],
            "Vendedor": [str(s) for s in lista.index],
            "Origen": [origen(s, estado) for s in lista.index.map(estado_vendedor)],
        }
    )
    if "riesgo_predicho" in lista.columns:
        riesgo = lista["riesgo_predicho"].to_numpy()
        if cortes:
            vista["Riesgo"] = nivel_riesgo(riesgo, cortes)
        vista["Puntaje"] = [puntaje(r) for r in riesgo]
    for columna, titulo in COLUMNAS.items():
        if columna in lista.columns:
            vista[titulo] = lista[columna].to_numpy()
    return vista


def resumen_top(lista: pd.DataFrame, top_k: int = TOP_K) -> dict:
    """Cuántos candidatos hay y cómo son los destacados frente a todos.

    Los valores que el recomendador no devuelva quedan en None.
    """

    def valor(parte: pd.DataFrame, columna: str, como: str) -> float | None:
        if columna not in parte.columns or parte.empty:
            return None
        resultado = getattr(parte[columna], como)()
        return None if pd.isna(resultado) else float(resultado)

    top = lista.head(top_k)
    return {
        "candidatos": len(lista),
        "top": len(top),
        "riesgo_top": valor(top, "riesgo_predicho", "mean"),
        "riesgo_todos": valor(lista, "riesgo_predicho", "mean"),
        "precio_top": valor(top, "precio_mediano", "median"),
        "precio_todos": valor(lista, "precio_mediano", "median"),
    }


def tabla_metricas(resumen: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """Para el jurado: Precision@K, Recall@K y tasa media de retraso de los mostrados por criterio."""
    columnas = {
        "precision": f"Precision@{k}",
        "recall": f"Recall@{k}",
        "retraso": "Tasa media de retraso de los mostrados",
    }
    vista = pd.DataFrame(
        {titulo: resumen[columna].map(format_pct) for columna, titulo in columnas.items() if columna in resumen.columns}
    )
    vista.index = [CRITERIOS.get(c, c) for c in resumen.index]
    vista.index.name = "Forma de elegir"
    return vista


def comparacion_criterios(resumen: pd.DataFrame) -> pd.DataFrame:
    """Tasa media de retraso de los vendedores mostrados por cada forma de elegir, para el gráfico.

    Columnas: criterio (nombre en pantalla), tasa (0 a 1) y etiqueta ('2,6 %'), en el orden de
    CRITERIOS. Vacía si el resumen no trae la tasa.
    """
    if "retraso" not in resumen.columns:
        return pd.DataFrame(columns=["criterio", "tasa", "etiqueta"])
    orden = [c for c in CRITERIOS if c in resumen.index]
    tasas = resumen.loc[orden, "retraso"].astype(float)
    return pd.DataFrame(
        {"criterio": [CRITERIOS[c] for c in orden], "tasa": tasas.to_numpy(), "etiqueta": tasas.map(format_pct).to_numpy()}
    )


def es_significativa(ic: tuple[float, float, float]) -> bool:
    """El intervalo de confianza de la diferencia no incluye el cero."""
    _, bajo, alto = ic
    return bajo > 0 or alto < 0


def texto_diferencia(ic: tuple[float, float, float]) -> str:
    """(-0.0058, -0.0085, -0.0031) -> '−0,58 puntos (IC 95 %: −0,85 a −0,31)'."""

    def pp(x: float) -> str:
        return f"{x * 100:+.2f}".replace("-", "−").replace(".", ",")

    media, bajo, alto = ic
    return f"{pp(media)} puntos (IC 95 %: {pp(bajo)} a {pp(alto)})"


def texto_reales(valor: float | None) -> str:
    if valor is None:
        return "—"
    return "R$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def texto_vendedores(n: int) -> str:
    return f"{format_int(n)} {'vendedor' if n == 1 else 'vendedores'}"

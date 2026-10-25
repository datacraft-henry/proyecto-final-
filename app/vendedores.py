"""Pestaña de vendedores recomendados, sin Streamlit, para poder testearla.

El orden lo decide `src.recomendador.recomendar` (hoy, el riesgo medio que predice el
clasificador para los pedidos de cada vendedor): la pestaña muestra la lista tal como
llega y no vuelve a ordenarla. Las columnas que el recomendador no devuelva no se muestran.
"""

import pandas as pd

from app.vista import format_int, format_pct

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
    "clasificador": "Recomendador (clasificador)",
    "historial": "Historial del vendedor",
    "popularidad": "Popularidad",
    "azar": "Al azar",
}

COLUMNAS = {
    "riesgo_predicho": "Riesgo predicho",
    "ventas_aqui": "Ventas aquí",
    "precio_mediano": "Precio mediano",
    "pedidos_hist": "Pedidos (historial)",
}


def nombre_estado(sigla: str) -> str:
    """'RJ' -> 'Rio de Janeiro (RJ)'."""
    return f"{ESTADOS.get(sigla, sigla)} ({sigla})"


def nombre_categoria(categoria: str, traduccion: dict[str, str]) -> str:
    """Nombre en inglés de Kaggle, sin guiones bajos ('cama_mesa_banho' -> 'bed bath table')."""
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


def tabla_vendedores(
    lista: pd.DataFrame, estado: str, estado_vendedor: pd.Series, top_k: int = TOP_K
) -> pd.DataFrame:
    """Tabla para pantalla, en el orden del recomendador (`lista` indexada por seller_id).

    Marca los primeros `top_k` y deja los porcentajes de 0 a 100 para las columnas de Streamlit.
    """
    origen = lista.index.map(estado_vendedor)
    vista = pd.DataFrame(
        {
            "#": range(1, len(lista) + 1),
            "": ["⭐" if i < top_k else "" for i in range(len(lista))],
            "Vendedor": [str(s)[:8] for s in lista.index],
            "Desde": [
                ESTADOS.get(s, s) + (" (local)" if s == estado else "") if isinstance(s, str) else "—" for s in origen
            ],
        }
    )
    for columna, titulo in COLUMNAS.items():
        if columna in lista.columns:
            valores = lista[columna].to_numpy()
            vista[titulo] = valores * 100 if columna == "riesgo_predicho" else valores
    return vista


def resumen_top(lista: pd.DataFrame, top_k: int = TOP_K) -> dict:
    """Lo que se lee en las tarjetas: cuántos candidatos hay y cómo son los recomendados frente a todos.

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
    """Precision@K, Recall@K y retraso de los mostrados por criterio (resumen_evaluacion), en texto."""
    columnas = {"precision": f"Precision@{k}", "recall": f"Recall@{k}", "retraso": "Retraso de los mostrados"}
    vista = pd.DataFrame(
        {titulo: resumen[columna].map(format_pct) for columna, titulo in columnas.items() if columna in resumen.columns}
    )
    vista.index = [CRITERIOS.get(c, c) for c in resumen.index]
    vista.index.name = "Criterio"
    return vista


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


def texto_pct(valor: float | None) -> str:
    return "—" if valor is None else format_pct(valor)


def texto_reales(valor: float | None) -> str:
    if valor is None:
        return "—"
    return "R$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def texto_vendedores(n: int) -> str:
    return f"{format_int(n)} {'vendedor' if n == 1 else 'vendedores'}"

import pandas as pd
import pytest

from app.vendedores import (
    MIN_CANDIDATOS,
    categorias,
    combinaciones,
    comparacion_criterios,
    es_significativa,
    estados_de,
    nombre_categoria,
    nombre_estado,
    nivel_riesgo,
    puntaje,
    resumen_top,
    tabla_metricas,
    tabla_vendedores,
    texto_diferencia,
    texto_reales,
    texto_vendedores,
)


CORTES = {"alto": 0.20, "medio": 0.05}


@pytest.fixture
def lista():
    """Como la devuelve src.recomendador.recomendar: indexada por vendedor, del menor al mayor riesgo."""
    return pd.DataFrame(
        {
            "ventas_aqui": [11, 14, 48, 9],
            "precio_mediano": [100.0, 60.0, 30.0, 80.0],
            "riesgo_predicho": [0.03, 0.04, 0.05, 0.09],
            "pedidos_hist": [58, 103, 406, 61],
        },
        index=pd.Index(["vend-uno-0001", "vend-dos-0002", "vend-tres-003", "vend-cuat-004"], name="seller_id"),
    )


@pytest.fixture
def estado_vendedor():
    return pd.Series({"vend-uno-0001": "SP", "vend-dos-0002": "RJ", "vend-tres-003": "MG"})


def _ventas(spec):
    """spec: (categoría, estado, cantidad de vendedores distintos)."""
    filas = [
        {"product_category_name": cat, "customer_state": est, "seller_id": f"{cat}-{est}-{i}"}
        for cat, est, n in spec
        for i in range(n)
    ]
    return pd.DataFrame(filas)


def test_solo_se_ofrecen_combinaciones_con_mas_vendedores_que_los_recomendados():
    ventas = _ventas([("perfumaria", "SP", MIN_CANDIDATOS), ("perfumaria", "AC", MIN_CANDIDATOS - 1), ("telefonia", "RJ", 9)])
    combos = combinaciones(pd.concat([ventas, ventas]))  # pedidos repetidos del mismo vendedor no suman
    assert set(map(tuple, combos[["product_category_name", "customer_state"]].values)) == {
        ("perfumaria", "SP"),
        ("telefonia", "RJ"),
    }
    assert categorias(combos) == ["perfumaria", "telefonia"]


def test_los_estados_se_ordenan_por_nombre():
    combos = combinaciones(_ventas([("perfumaria", "SP", 5), ("perfumaria", "AC", 5), ("telefonia", "BA", 5)]))
    assert estados_de(combos, "perfumaria") == ["AC", "SP"]  # Acre antes que São Paulo
    assert nombre_estado("RJ") == "Rio de Janeiro (RJ)"
    assert nombre_estado("XX") == "XX (XX)"


def test_la_tabla_respeta_el_orden_del_recomendador(lista, estado_vendedor):
    """La pestaña no reordena: si el recomendador cambia de criterio, cambia la lista que llega y nada más."""
    al_reves = lista.iloc[::-1]
    tabla = tabla_vendedores(al_reves, "RJ", estado_vendedor, top_k=2)
    assert list(tabla["Vendedor"]) == ["vend-cuat-004", "vend-tres-003", "vend-dos-0002", "vend-uno-0001"]
    assert list(tabla[""]) == ["⭐", "⭐", "", ""]


def test_la_tabla_marca_los_primeros_y_muestra_nivel_y_puntaje(lista, estado_vendedor):
    """El riesgo se muestra como nivel y puntaje para ordenar, no como frecuencia futura de retrasos."""
    tabla = tabla_vendedores(lista, "RJ", estado_vendedor, CORTES, top_k=3)
    assert list(tabla["#"]) == [1, 2, 3, 4]
    assert list(tabla[""]) == ["⭐", "⭐", "⭐", ""]
    assert list(tabla["Puntaje"]) == [3, 4, 5, 9]
    assert list(tabla["Riesgo"]) == ["🟢 Bajo", "🟢 Bajo", "🟡 Medio", "🟡 Medio"]
    assert list(tabla["Origen"]) == ["São Paulo", "Rio de Janeiro (mismo estado)", "Minas Gerais", "—"]
    assert list(tabla["Ventas en este contexto"]) == [11, 14, 48, 9]
    assert not any("de cada" in c for c in tabla.columns)


def test_la_tabla_omite_las_columnas_que_el_recomendador_no_devuelve(lista, estado_vendedor):
    tabla = tabla_vendedores(lista.drop(columns=["pedidos_hist", "precio_mediano"]), "RJ", estado_vendedor)
    assert "Pedidos anteriores" not in tabla.columns and "Precio histórico de referencia" not in tabla.columns
    assert {"Puntaje", "Ventas en este contexto"} <= set(tabla.columns)
    assert "Riesgo" not in tabla.columns  # sin cortes no hay nivel


def test_el_resumen_compara_a_los_recomendados_con_todos(lista):
    r = resumen_top(lista, top_k=2)
    assert r["candidatos"] == 4 and r["top"] == 2
    assert r["riesgo_top"] == pytest.approx(0.035)
    assert r["riesgo_todos"] == pytest.approx(0.0525)
    assert r["precio_top"] == pytest.approx(80.0)
    assert r["precio_todos"] == pytest.approx(70.0)


def test_el_resumen_queda_vacio_si_faltan_datos(lista):
    r = resumen_top(lista.drop(columns=["riesgo_predicho"]))
    assert r["riesgo_top"] is None and r["riesgo_todos"] is None


def test_tabla_de_metricas_para_el_jurado():
    resumen = pd.DataFrame(
        {"precision": [0.712, 0.661], "recall": [0.283, 0.270], "retraso": [0.0257, 0.0314]},
        index=["clasificador", "azar"],
    )
    vista = tabla_metricas(resumen, k=3)
    assert list(vista.index) == ["Nuestro modelo", "Uno cualquiera (al azar)"]
    assert vista.loc["Nuestro modelo", "Precision@3"] == "71,2 %"
    assert vista.loc["Uno cualquiera (al azar)", "Tasa media de retraso de los mostrados"] == "3,1 %"


def test_la_comparacion_sigue_el_orden_de_los_criterios_y_no_redondea_a_pedidos():
    resumen = pd.DataFrame(
        {"retraso": [0.0314, 0.0258, 0.0280, 0.0275]}, index=["azar", "clasificador", "historial", "popularidad"]
    )
    c = comparacion_criterios(resumen)
    assert list(c["criterio"]) == [
        "Nuestro modelo", "El que menos se atrasó antes", "El que más vende", "Uno cualquiera (al azar)"
    ]
    assert list(c["etiqueta"]) == ["2,6 %", "2,8 %", "2,8 %", "3,1 %"]
    assert comparacion_criterios(resumen.drop(columns=["retraso"])).empty


def test_significancia_y_texto_de_la_diferencia():
    assert es_significativa((-0.0058, -0.0085, -0.0031))
    assert not es_significativa((-0.0023, -0.0059, 0.0012))
    assert texto_diferencia((-0.0058, -0.0085, -0.0031)) == "−0,58 puntos (IC 95 %: −0,85 a −0,31)"
    assert texto_diferencia((-0.0023, -0.0059, 0.0012)) == "−0,23 puntos (IC 95 %: −0,59 a +0,12)"


def test_formatos():
    # Español primero (la clave interna no cambia); si no hay, inglés de Kaggle; si no, la clave.
    assert nombre_categoria("cama_mesa_banho", {"cama_mesa_banho": "bed_bath_table"}) == "Cama, mesa y baño"
    assert nombre_categoria("otra_cosa", {"otra_cosa": "other_thing"}) == "other thing"
    assert nombre_categoria("sin_categoria", {}) == "sin categoria"
    assert texto_reales(1234.5) == "R$ 1.234,50"
    assert texto_reales(None) == "—"
    assert texto_vendedores(1) == "1 vendedor" and texto_vendedores(1500) == "1.500 vendedores"


def test_puntaje_y_nivel():
    assert puntaje(0.074) == 7
    assert puntaje(None) is None
    assert nivel_riesgo([0.01, 0.05, 0.25], CORTES) == ["🟢 Bajo", "🟡 Medio", "🔴 Alto"]

# Promesas de entrega por ruta

## Objetivo

Como segundo uso de la señal logística del proyecto, se evaluó una propuesta
para definir los días prometidos de entrega según la ruta de cada pedido.

La idea es que no todas las rutas se comportan igual: un envío desde São Paulo
hacia Río de Janeiro puede tener tiempos históricos distintos de uno dentro de
São Paulo. Por eso se analiza si una fecha prometida ajustada por ruta puede
reducir los incumplimientos de la fecha comunicada al cliente.

Este análisis es complementario. No modifica el clasificador de riesgo, la API
ni la aplicación de Streamlit.

## Regla evaluada

Cada pedido se representa con una sola ruta:

```text
estado del vendedor principal -> estado del cliente
```

El vendedor principal es el que aporta el ítem de mayor precio del pedido. Esta
decisión permite conservar una única ruta por pedido, incluso cuando interviene
más de un vendedor.

Para cada ruta se calcula el percentil 95 (P95) de los días reales entre la
compra y la entrega al cliente. El valor se redondea hacia arriba al día entero
siguiente y se usa como nueva promesa propuesta.

Si una ruta de test no apareció en train, se usa como respaldo el P95 global de
train, que fue de 32 días.

## Separación temporal y prevención de fuga

La tabla de rutas se ajusta solo con los pedidos de train. La validación se
mantiene separada y el test se utiliza una única vez para reportar el resultado.

| Cohorte | Pedidos | Uso |
| --- | ---: | --- |
| Train | 61.276 | Calcular el P95 por ruta y el P95 global. |
| Validación | 15.620 | Se conserva sin intervenir en la regla. |
| Test | 19.293 | Evaluar la propuesta. |

También existe una prueba automática que modifica artificialmente los días de
entrega de test y comprueba que la tabla de P95 no cambia. Así se verifica que
el resultado de test no entra en la definición de las promesas.

## Resultado en test

| Métrica | Promesa original de Olist | P95 por ruta |
| --- | ---: | ---: |
| Incumplimientos de la fecha prometida | 673 (3,49 %) | 242 (1,25 %) |
| Cambio neto de incumplimientos | — | -431 |
| Promesa media | 22,0 días | 27,5 días |

La regla por ruta cubre el 99,74 % de los pedidos de test con una ruta conocida
en train. Las 376 rutas observadas se complementan con el P95 global para las
rutas no vistas.

También se observan 528 incumplimientos originales que dejan de ocurrir frente
a la nueva fecha y 97 incumplimientos nuevos, porque algunas rutas reciben una
promesa más corta que la original. El cambio mediano de la promesa es de
**+6 días**.

## El trade-off: menos incumplimientos, pero promesas más largas

Este resultado no significa que las entregas sean físicamente más rápidas. La
entrega real media de test es de 8,7 días; lo que cambia es la fecha que se
comunica al cliente.

En el 72 % de los pedidos la promesa por ruta es más larga que la promesa
original. Por eso la reducción de incumplimientos debe interpretarse como una
mejor alineación entre la fecha prometida y el comportamiento histórico, no
como una mejora gratuita del servicio logístico.

Como referencia descriptiva, se comparó la regla por ruta con extender la
promesa original la misma cantidad para todos los pedidos. La promesa por ruta
aumenta en promedio unos 5,5 días, por lo que se muestran los dos enteros más
cercanos:

| Política evaluada en test | Incumplimientos de promesa |
| --- | ---: |
| Promesa original | 3,49 % |
| Promesa original + 5 días para todos | 1,32 % |
| P95 por ruta | 1,25 % |
| Promesa original + 6 días para todos | 1,10 % |

La regla por ruta mejora frente a sumar cinco días a todos, pero no muestra una
ventaja clara frente a sumar seis días. Por lo tanto, no se presenta como una
optimización sin costo: su valor potencial está en diferenciar rutas, aunque en
este test esa diferenciación no supera de forma concluyente a un corrimiento
uniforme de longitud comparable.

Esta comparación se usa para interpretar el resultado final; no se utiliza para
elegir o reentrenar el modelo de clasificación.

El mismo criterio se aplica al recomendador de vendedores: una menor tasa de
retraso frente a una promesa puede ir acompañada de una fecha prometida más
larga. Cumplir mejor una promesa no equivale necesariamente a entregar más
rápido.

## Ejemplos de rutas frecuentes

| Ruta | Pedidos en train | P95 observado | Días a prometer |
| --- | ---: | ---: | ---: |
| SP -> SP | 18.230 | 20,0 | 20 |
| SP -> RJ | 5.533 | 43,0 | 43 |
| SP -> MG | 4.914 | 26,0 | 26 |
| SP -> RS | 2.410 | 37,0 | 37 |

El caso SP -> RJ ilustra una limitación importante: su P95 de 43 días está
influido por entregas inusualmente largas observadas en train, incluyendo el
período excepcional de febrero y marzo de 2018. Antes de convertir esta tabla
en una política operativa sería necesario revisar si esos eventos históricos
siguen representando el comportamiento actual de la ruta.

## Limitaciones y próximos pasos

- 272 de las 376 rutas tienen menos de 30 pedidos en train. Aunque representan
  solo el 3,6 % de los pedidos de test, sus P95 son menos estables.
- Una versión productiva debería definir un mínimo de pedidos por ruta y usar
  el P95 global cuando una ruta no alcance ese soporte. Ese umbral debería
  elegirse con validación y una decisión de negocio, no ajustándolo después de
  observar el test.
- La promesa se basa en comportamiento histórico. Debe recalibrarse y
  monitorearse si cambian rutas, transportadoras o condiciones operativas.
- Esta propuesta no separa la parte atribuible al vendedor de la atribuible al
  transporte. Por eso es una herramienta para comunicar una expectativa más
  realista, no una explicación causal de cada demora.

## Cómo reproducirlo

Desde la raíz del repositorio, con los CSV originales en `data/raw`:

```powershell
.\.venv\Scripts\python.exe -m src.promesas_ruta
```

El comando genera localmente:

- `reports/promesas_por_ruta.csv`: tabla de rutas, P95 y días a prometer.
- `reports/evaluacion_promesas_ruta.json`: resumen de la evaluación temporal.

Los reportes no se versionan porque se generan a partir de los datos locales.

## Validación técnica

Se verificó el estilo de los archivos incorporados y se ejecutó la suite
completa del proyecto sobre la versión integrada de `developer`:

```text
134 passed, 1 skipped
```

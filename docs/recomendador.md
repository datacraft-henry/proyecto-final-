# Sistema de recomendación: vendedores según riesgo de retraso

Es el segundo producto de la misma señal de riesgo. El clasificador sirve a logística después de aprobar el pago (qué pedidos vigilar); el recomendador sirve al cliente antes de comprar: **en la categoría que busca y para su estado, le muestra los vendedores con menor riesgo de retraso**, según el mismo clasificador. Decisión 12 en `docs/decisiones.md`.

## El enfoque: recomendación contextual para clientes sin historial

La consigna pide recomendar según el historial del usuario, pero en Olist el 96,9 % de los clientes compra una sola vez (`customer_unique_id`): todos son clientes de **arranque en frío** (*cold start*). Sin historial por cliente no hay filtrado colaborativo posible. La técnica para ese caso es la **recomendación contextual**: se recomienda según el contexto de la compra.

| Pieza | En este sistema |
|---|---|
| Contexto | Categoría del producto y estado de destino del cliente |
| Lo que se recomienda | Vendedores que ya vendieron esa categoría a ese estado |
| Criterio (modelo) | Riesgo de retraso medio que el clasificador predice para los pedidos del vendedor |
| Baselines | Popularidad (los vendedores con más pedidos, el baseline clásico de arranque en frío) e historial (su tasa de retraso suavizada) |
| Referencia | Azar entre los mismos candidatos |

## Por qué recomendar por retraso

**El retraso es lo que más daña la reseña.** De las reseñas de pedidos que llegaron tarde, el 53,7 % es de 1 estrella, contra 6,6 % de los que llegaron a tiempo (nota media 2,27 contra 4,29; 6.406 y 89.681 reseñas). Recomendar vendedores que se retrasan menos apunta a ese KPI: la satisfacción del cliente.

**Sobre la recompra, la señal es débil.** En Olist casi nadie vuelve a comprar, llegue o no a tiempo. Tomamos el primer pedido de cada cliente, con seguimiento completo:

| Primer pedido | Clientes | Recompra a 6 meses | Ajustada por estado |
|---|---|---|---|
| A tiempo | 51.526 | 2,9 % | 2,9 % |
| Tarde | 3.668 | 2,5 % | 2,7 % |

Primeros pedidos comprados hasta el 28/02/2018. Quien recibe tarde vuelve algo menos (riesgo relativo 0,85), pero la diferencia es de 0,4 puntos con un IC 95 % de −1,0 a +0,1: no es concluyente, y se achica al ajustar por estado. A un año (primeros pedidos hasta el 31/08/2017) da 4,4 % contra 3,3 % (IC 95 % de −2,3 a +0,3 puntos; 747 clientes con primer pedido tarde).

- **Sí se puede decir:** el retraso daña la satisfacción, y los clientes con un primer pedido tardío además recompran algo menos. Medir si recomendar vendedores puntuales mejora la recompra es el objetivo de la prueba A/B.
- **No se puede decir:** que los clientes compran una sola vez porque Olist se demora. El 97 % de los que recibieron a tiempo tampoco volvió.

## Cómo recomienda

`src/recomendador.py`:

1. **Pedidos de un solo vendedor** (94.931 de los 96.203 con target), para atribuir el retraso sin ambigüedad.
2. **Riesgo de cada pedido:** `prob_retraso` del clasificador calibrado y congelado (`models/modelo_principal_calibrado.joblib`, `predict_risk`). No se reentrena nada.
3. **Historial hasta el inicio del test** (27/05/2018, el mismo corte del clasificador): por vendedor con al menos 10 pedidos, su riesgo medio predicho (`puntaje_vendedores`). Son 987 vendedores.
4. **Recomendación** (`recomendar`): de los vendedores que ya vendieron esa categoría a ese estado, los K de menor riesgo medio, con su precio mediano y cuántas veces vendieron ahí. Es la función que usa la pestaña de la app.
5. **Medición en el período de test** (27/05 al 29/08/2018), que ni el clasificador ni el historial vieron (`evaluar_recomendacion`).

El clasificador no usa la identidad del vendedor: su riesgo resume cómo son los pedidos de ese vendedor (plazo prometido, distancia al cliente, si vende fuera de su estado, peso, flete). En test, dentro de cada categoría y estado, los 3 recomendados **prometen unos 2 días más** que el resto de los candidatos, están **a la misma distancia** del cliente y **no son más caros**. Su orden se parece poco al de la tasa histórica: miden cosas distintas.

## Resultado: Precision@K y Recall@K en test

En cada combinación de categoría y estado del cliente se muestran K vendedores con cada criterio:

- **Candidatos:** los vendedores con historial que vendieron en esa combinación en test, con al menos 5 pedidos en test. Solo cuentan las combinaciones con al menos 2K candidatos.
- **Vendedor bueno:** en test se retrasó menos que la media del test (3,5 %).
- **Precision@K:** de los K mostrados, qué parte es buena. **Recall@K:** de los buenos disponibles, qué parte quedó entre los K.
- **Retraso de los mostrados:** la tasa de retraso en test de los K vendedores, promediada por vendedor. Es lo que vive el cliente, sin el umbral de "bueno".

Con K = 3 (211 combinaciones, mediana de 11 candidatos):

| Criterio | Precision@3 | Recall@3 | Retraso de los mostrados | Diferencia con el azar (IC 95 %) |
|---|---|---|---|---|
| **Clasificador (recomendador)** | **71,1 %** | **28,0 %** | **2,58 %** | **−0,56 pp (−0,84 a −0,29)** |
| Historial | 69,8 % | 28,4 % | 2,79 % | −0,35 pp (−0,64 a −0,06) |
| Popularidad | 69,4 % | 27,7 % | 2,75 % | −0,40 pp (−0,61 a −0,19) |
| Azar | 66,1 % | 27,0 % | 3,14 % | — |

Con K = 5 (128 combinaciones, mediana de 17): Precision@5 de 73,4 % con el clasificador, contra 68,9 % historial, 68,1 % popularidad y 66,6 % azar; retraso de los mostrados de 2,46 % contra 2,78 %, 2,93 % y 3,13 %. Todo se reproduce con `python -m src.recomendador`. Las cifras del clasificador son las del equipo de referencia del proyecto: en otra máquina pueden variar en el último decimal, porque la regresión logística se optimiza por iteraciones y cada procesador redondea distinto.

**Cómo leerlo:** el cliente que compra a los 3 recomendados vive un retraso de 2,6 % contra 3,1 % al azar, un riesgo relativo de 0,82. El clasificador es el mejor de los tres criterios, pero **no se distingue del historial**: la diferencia es de −0,21 pp con un IC 95 % de −0,58 a +0,14. Es un efecto real pero modesto, y es lo esperable: 7 de cada 10 retrasos ocurren en el transporte, no en el vendedor (ver "Límites").

**Con todo 2018 el historial también ordena.** Ordenando con 2017 y midiendo en 2018 (enero a agosto), el tercio de vendedores de menor riesgo histórico se retrasa 6,7 % y el de mayor 9,9 %, también dentro del mismo estado del cliente (6,7 % contra 9,8 %), y los de menor riesgo no son más caros (precio mediano de R$ 69,90 contra R$ 69,99; `tercios_despues_del_corte`). El clasificador no se compara en ese período porque ya vio esos meses al entrenar.

## Límites

- **Recomienda en parte a quien promete más días.** Los recomendados prometen unos 2 días más: se retrasan menos frente a su promesa, pero el cliente puede esperar más. Lo que se recomienda es cumplir lo prometido, no entregar más rápido.
- **El efecto es modesto y no supera al historial.** En test el recomendador baja el retraso de los mostrados en 0,6 puntos frente al azar, y su ventaja sobre el historial no es significativa.
- **El vendedor explica solo una parte del retraso.** De los pedidos que llegaron tarde, solo el 27,7 % tuvo un vendedor que entregó a la transportadora después de su plazo límite (`shipping_limit_date`). En los otros 7 de cada 10, el retraso ocurrió en el transporte. Cuando el vendedor despacha tarde, el pedido llega tarde el 21,1 % de las veces, contra 5,4 % cuando despacha a tiempo. La fecha de despacho se usa solo para este diagnóstico: es posterior a la compra y nunca entra al modelo (decisión 4).
- **El test es corto y de bajo retraso.** 95 días con 3,5 % de retraso: pocos eventos por vendedor, y la medición es ruidosa.
- **El umbral de "bueno" es nuestro.** Por eso también reportamos el retraso de los mostrados, que no depende de él.
- **Los candidatos no son el catálogo.** Son los que vendieron en test, no los que ofrecían el producto (ese dato no está). Saber quién vendió en test es una ventaja que la operación real no tendría.
- **Combinaciones no independientes.** Un vendedor grande aparece en varias; los IC por bootstrap de combinaciones son optimistas.
- **Promedio por vendedor, no por pedido.** Un vendedor con 5 pedidos pesa igual que uno con 500.
- **Cobertura parcial:** el 73 % de los pedidos de test tiene un vendedor con al menos 10 pedidos de historial. Los vendedores nuevos no se pueden recomendar.
- **Es observacional.** Los clientes no eligieron vendedor al azar. El efecto de recomendar se mediría con una prueba A/B.
- **Datos de 2017–2018.**

## Cómo correrlo

Desde la raíz del repo, con los 9 CSV en `data/raw/` y el modelo generado (`python -m src.umbral`):

```bash
python -m src.recomendador                 # todas las cifras de este documento
python -m pytest tests/test_recomendador.py
```

La app de `prototipos/recomendador_vendedores.py` muestra el baseline de historial y queda hasta que la pestaña de recomendación de la app principal la reemplace.

## Qué datos necesitaríamos de Olist

| Dato | Para qué |
|---|---|
| Transportadora y modalidad de envío de cada pedido | Separar el retraso del vendedor del retraso del transporte |
| Eventos de seguimiento con fecha (recolección, centros de distribución, intento de entrega) | Saber en qué tramo de la ruta se pierde el tiempo |
| Catálogo de ofertas: qué vendedores ofrecían cada producto, con precio y stock | Recomendar entre alternativas reales |
| Ubicación real del depósito del vendedor | Calcular la distancia desde donde sale el paquete |
| Tiempo de preparación declarado y capacidad del vendedor | Anticipar demoras de despacho en picos |
| Qué vendedores vio el cliente y cuál eligió | Medir el efecto de recomendar con una prueba A/B |
| Pedidos posteriores a 2018, con reclamos y cancelaciones | Validar que el orden sigue vigente e incluir pedidos que nunca llegaron |

El primer dato importa también para el modelo principal: si 7 de cada 10 retrasos ocurren en el transporte, la transportadora es la variable que más falta.

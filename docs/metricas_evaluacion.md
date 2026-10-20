# Métricas de evaluación y justificación del modelo

Qué medimos, por qué, cómo validamos y por qué el modelo principal es la regresión logística. Los números salen de `reports/comparacion_modelos.json` (`python -m src.training`) y `reports/umbral_calibracion_principal.json` (`python -m src.umbral`); el detalle de cada decisión está en [`decisiones.md`](decisiones.md) (puntos 9 y 10) y [`decisiones_modelos.md`](decisiones_modelos.md).

## El problema que mide la evaluación

Predecimos `pedido_retrasado` (entrega después del día prometido) al aprobar la compra. Los retrasos son pocos y su tasa cambia mucho con el tiempo:

| Cohorte | Período (aprobación) | Pedidos | Tasa de retraso |
|---|---|---|---|
| Train | ene 2017 a mar 2018 | 61.276 | 7,8 % |
| Validación | mar a may 2018 | 15.620 | 6,8 % |
| Test | fines de may a ago 2018 | 19.293 | 3,5 % |

Eso define las métricas: con clases tan desbalanceadas un modelo que dice "nunca se retrasa" acierta el 96,5 % del test y no sirve para nada, y con la tasa moviéndose así un umbral de probabilidad fijo dispara cantidades muy distintas de alertas según el mes.

## Métricas que usamos

| Métrica | Para qué la usamos | Por qué |
|---|---|---|
| **PR-AUC** (precision media) | Elegir el modelo | Resume precision y recall en todos los umbrales y se concentra en la clase rara. Su piso es la tasa de retraso (0,035 en test), así que se lee contra ese piso. |
| **ROC-AUC** | Control secundario | Mide si el modelo ordena bien los pedidos. No depende de la prevalencia, por eso sirve para comparar cohortes con tasas distintas. Detectó la caída de los árboles en test. |
| **Precision y recall en el nivel alto** | Leer el modelo en términos operativos | Logística revisa una cantidad fija de pedidos. Importa qué parte de esas alertas son retrasos reales (precision) y qué parte de los retrasos captan (recall). |
| **Lift por nivel** | Explicarlo al negocio | Cuántas veces más retrasos hay en un nivel que en el total. Es la forma más simple de decir "el nivel alto vale la pena". |
| **Calibración por deciles** | Saber si la probabilidad se puede leer como tal | Compara el riesgo predicho con la tasa observada en cada decil. |
| **Matriz de confusión** | Detalle de errores en el umbral elegido | Cuántos falsos positivos cuesta cada retraso detectado. |

**Qué no usamos para decidir:** la exactitud (accuracy), por el desbalance, y el F1, porque depende del umbral y el umbral lo fijamos aparte, por volumen de alertas. El F1 se reporta igual en los JSON.

## Plan de validación

1. **Corte por fecha, nunca al azar.** `temporal_three_way_split` (`src/validation.py`) separa train, validación y test en ese orden. Todo pedido de test es posterior a todo pedido de train, como pasaría en producción.
2. **Elección del modelo con validación cruzada temporal.** Sobre train + validación, 4 tramos consecutivos; cada tramo se predice con un modelo entrenado solo con lo anterior (`temporal_cv_folds` en `src/training.py`). Gana el mejor PR-AUC medio y, si un candidato más simple queda a menos de 0,01, gana el simple.
3. **Calibración y niveles en validación.** Con el modelo entrenado en train se ajusta la recalibración de Platt y se fijan los cortes de riesgo: alto = 5 % de pedidos con más riesgo, medio = 15 % siguiente, bajo = el resto (`src/calibration.py`).
4. **Test una sola vez.** Se mide recién con el modelo, la calibración y los cortes ya elegidos. Se informa para todos los candidatos como diagnóstico, pero no cambia la elección.
5. **Modelo final.** El mismo pipeline se reentrena con train + validación y se guarda con el calibrador y los cortes de validación en `models/modelo_principal_calibrado.joblib`.
6. **Sin fuga.** Solo entran variables conocidas al aprobar la compra. El historial del vendedor cuenta únicamente pedidos entregados antes de ese momento, y hay un test que lo verifica (`tests/test_feature_engineering.py`).

## Resultados

### Elección del modelo (validación cruzada temporal, PR-AUC)

| Modelo | Tramo 1 | Tramo 2 | Tramo 3 | Tramo 4 | Media | ROC-AUC media |
|---|---|---|---|---|---|---|
| Regresión logística, baseline (referencia) | 0,119 | 0,278 | 0,301 | 0,129 | 0,206 | 0,701 |
| **Regresión logística, selección (elegido)** | 0,118 | 0,280 | 0,310 | 0,139 | **0,212** | 0,711 |
| Árboles (HistGradientBoosting), selección | 0,111 | 0,274 | 0,326 | 0,148 | 0,215 | 0,710 |
| Árboles, todas las features | 0,110 | 0,259 | 0,313 | 0,153 | 0,209 | 0,706 |

Los árboles quedan a 0,003 de la logística, dentro de la tolerancia de 0,01, así que gana la logística por ser más simple. La diferencia entre tramos (de 0,12 a 0,31) es mayor que la diferencia entre modelos: es la deriva temporal, y por eso no damos por buena una mejora de menos de 0,01.

### Test (19.293 pedidos, 3,5 % de retrasos)

| Modelo | PR-AUC | ROC-AUC |
|---|---|---|
| Logística, baseline | 0,072 | 0,700 |
| **Logística, selección** | **0,073** | **0,707** |
| Árboles, selección | 0,051 | 0,579 |
| Árboles, todas las features | 0,046 | 0,557 |

El PR-AUC de 0,073 es el doble del piso (0,035). Es un modelo de priorización, no un detector: ordena mejor que el azar, pero la mayoría de las alertas no terminan en retraso.

### Niveles de riesgo del modelo final en test

Modelo reentrenado con train + validación, con los cortes fijados en validación para un 5 % de pedidos en alto:

| Nivel | % de pedidos | Tasa de retraso | % de los retrasos que capta | Lift |
|---|---|---|---|---|
| Alto | 11,2 % | 7,5 % | 24,1 % | 2,15 |
| Medio | 22,0 % | 5,6 % | 35,2 % | 1,60 |
| Bajo | 66,8 % | 2,1 % | 40,7 % | 0,61 |

Leído para el negocio: revisando 1 de cada 9 pedidos se llega a 1 de cada 4 retrasos, y en ese grupo los retrasos son 2,1 veces más frecuentes que en el total. Con alto y medio juntos (33,2 % de los pedidos) se capta el 59 % de los retrasos.

### Calibración

En validación la probabilidad calibrada promedia 6,8 %, igual a la tasa real. En test promedia 9,3 % contra 3,5 % real, porque la tasa de retraso cayó a la mitad en esos meses y el calibrador se ajustó con el modelo de train. Por eso en la app y en la demo mostramos el **nivel** de riesgo y no la probabilidad como dato principal: el orden se sostiene, el valor absoluto no.

### Sensibilidad al costo de los errores

Si un retraso no detectado cuesta r veces una falsa alerta, el umbral óptimo con probabilidades calibradas es 1 / (1 + r). En test, con el modelo final:

| r | Umbral | % de pedidos en alerta | Recall | Precision |
|---|---|---|---|---|
| 5 | 0,167 | 14,6 % | 29,7 % | 7,1 % |
| 10 | 0,091 | 35,0 % | 62,9 % | 6,3 % |
| 20 | 0,048 | 64,3 % | 90,9 % | 4,9 % |

Lo dejamos como análisis, no como regla: con la calibración inflada en los meses nuevos estos umbrales alertan más de lo previsto, y los niveles por volumen son más estables.

## Por qué la regresión logística

1. **Empata con el mejor en validación cruzada** (0,212 contra 0,215) y la regla de empate, fijada antes de mirar el test, ya la elegía.
2. **Aguanta la deriva.** Olist acortó los plazos prometidos (mediana de 25 días en train, 21 en test). Los árboles aprenden cortes fijos sobre el plazo y en test caen a ROC-AUC 0,58; la logística usa una relación lineal y se mantiene en 0,71.
3. **Es explicable.** Cada coeficiente dice hacia dónde mueve el riesgo una variable, algo que se puede mostrar en la app y defender en la demo.
4. **Variables:** las del baseline sin el mes de compra (empeora la validación porque aprende meses que no se repiten) más la distancia cliente-vendedor y los feriados de los 7 días siguientes, que mejoran la validación cruzada.

## Límites

- La señal es moderada: PR-AUC 0,073 y ROC-AUC 0,71 en test. Sirve para priorizar revisiones, no para prometer cuáles pedidos se van a retrasar.
- Los cortes se fijaron para un 5 % de pedidos en alto y en los meses nuevos el nivel alto marca cerca del 10 %. Hay que vigilarlo (ver más abajo).
- Solo hay pedidos entregados, así que el modelo no sabe nada de cancelaciones ni de pedidos que nunca llegaron.

## Cómo sigue en el Sprint 2

- **Reentreno con los builders de Mathias (desde el 19/10).** Cada builder nuevo entra como candidato a la misma validación cruzada temporal. Se queda solo si sube el PR-AUC medio en 0,01 o más y pasa el test anti-fuga. Después se vuelve a correr `python -m src.umbral` para fijar calibración y cortes, y el test se mira una sola vez.
- **Modelo congelado el 23/10.** El artefacto y los dos JSON de ese día son los que usan la app y la API, y los números de este documento se actualizan con ellos.
- **Monitoreo del nivel alto.** Por mes: qué porcentaje de pedidos cae en alto (esperado 5 %, hoy cerca de 10 %) y, cuando llega la entrega, la tasa de retraso y el lift del nivel alto. Si el volumen se aleja mucho del 5 % o el lift se acerca a 1, se recalibran los cortes con los meses más recientes. Los valores exactos de alarma se fijan en esa tarea.

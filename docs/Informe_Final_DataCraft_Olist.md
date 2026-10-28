# Informe final DataCraft Riesgo de retraso en Olist

Equipo DataCraft | Versión actualizada al 27 de octubre de 2026 | Rama developer, commit 2a842b4

## Resumen ejecutivo

Olist promete una fecha de entrega y el 6,8 % de los pedidos llega después. Esos pedidos se llevan reseñas de 1 estrella en el 54 % de los casos, contra 6,6 % de los que llegan a tiempo. Construimos un clasificador que, al aprobar el pago, asigna a cada pedido un nivel de riesgo alto, medio o bajo con la información disponible en ese momento.

En los meses de prueba, que el modelo no vio, el nivel alto reúne el 11,2 % de los pedidos y el 24,1 % de los retrasos: el doble que revisar al azar (lift 2,15; ROC-AUC 0,707; PR-AUC 0,073). El resultado se consulta en una app de Streamlit para operaciones y en una API en FastAPI y Docker para otros sistemas. Sirve para priorizar el seguimiento, no para predecir cada pedido, y necesita monitoreo y recalibración porque la operación cambia.

Con la misma señal construimos el sistema de recomendación. Antes de comprar, en la categoría que busca y para su estado, el cliente ve los 3 vendedores con menor riesgo de retraso según el clasificador. Como el 97 % de los clientes compra una sola vez, la recomendación es por el contexto de la compra y no por el historial del cliente. En los meses de prueba, los vendedores recomendados se retrasan 2,6 % contra 3,1 % al azar (Precision@3 de 71,1 % contra 66,1 %): un efecto real pero modesto, porque en 7 de cada 10 pedidos tarde el vendedor despachó a tiempo.

## 1 El problema y por qué importa

El caso de estudio es Olist, un marketplace que conecta a pequeños vendedores con grandes marketplaces. Analizamos 96.203 pedidos entregados con compra entre enero de 2017 y agosto de 2018, publicados por Olist en Kaggle. El 6,8 % llega tarde, con picos de 12,4 % y 19 % en algunos meses. El Nordeste es la zona más afectada, con tasas de hasta 21,5 %, contra 4,5 % en São Paulo, y Río de Janeiro concentra buena parte de los retrasos. Para clientes fuera de São Paulo, el retraso pasa de 3,5 % con un vendedor del mismo estado a 8,8 % con uno de otro estado.

El retraso se asocia con una peor experiencia del cliente: el 54 % de las reseñas de los pedidos que llegan tarde es de 1 estrella, contra el 6,6 % de los que llegan a tiempo. Esto motiva evaluar tanto la priorización del seguimiento como la adecuación de la promesa de entrega a cada ruta. El estudio es educativo y no corresponde a una contratación ni a una implementación operativa en Olist.

Nos preguntamos entonces si se puede predecir el riesgo de que un pedido llegue tarde desde el momento en que se aprueba el pago, usando solo la información disponible en ese instante.

## 2 Datos y período de estudio

Trabajamos con 96.203 pedidos entregados cuya compra ocurrió entre enero de 2017 y agosto de 2018, de los 99.441 que Olist publicó en Kaggle. Son 9 tablas CSV públicas: pedidos, ítems, clientes, vendedores, productos, pagos, reseñas, geolocalización y traducción de categorías. También las cargamos en BigQuery para consultarlas en SQL.

El período se eligió por dos razones:

- Se excluye 2016 porque tiene 267 pedidos entregados y meses casi vacíos: no representa la operación.

- Se corta en agosto de 2018 porque ahí termina, en la práctica, el dataset. Septiembre y octubre tienen solo 20 pedidos, y 19 están cancelados. Cortar al final de unos datos tiene un riesgo, la censura: un pedido que no llegó antes del 17 de octubre de 2018 no se puede etiquetar, y si los que faltan fueran justo los demorados, la tasa del final saldría más baja de lo real.

La revisión del cierre del dataset sugiere que la censura no explica por sí sola la tasa baja del final. Agosto tiene el 97,5 % de sus pedidos entregados, frente a junio (98,8 %) y julio (97,8 %). Los entregados de agosto llegaron con una mediana de 7 días; desde una compra del 31 de agosto quedan 47 días hasta el cierre, y el 99 % de las entregas observadas ocurre en 46 días o menos. Esto respalda el período elegido, pero no permite descartar el sesgo por pedidos sin entrega observada. La revisión se documenta en docs/decisiones.md, punto 8.

Para modelar quedan 96.189 pedidos: los 96.203 menos 14 entregados sin fecha de aprobación. La vista pedidos_analitica de BigQuery da el mismo número, lo que valida en SQL el cálculo hecho en Python.

## 3 Definición del target

Un pedido está retrasado (pedido_retrasado = 1) si se entrega al menos un día calendario después de la fecha estimada. Con esta regla, el 6,79 % de los pedidos del período llega tarde. Solo se etiquetan los pedidos entregados con fecha real y estimada.

Comparamos por día y no por hora porque la fecha estimada viene siempre a las 00:00. Por hora, un pedido entregado el mismo día prometido contaría como tarde, y la tasa subiría a 8,11 % frente a 6,77 % por día (con los 96.470 pedidos entregados, antes de fijar el período).

La definición vive en un solo lugar, src/target.py (build_target), que usan el modelo, los notebooks y el script del tablero; la vista de BigQuery la replica en SQL. Los pedidos no entregados se excluyen porque su retraso no se observa, lo que limita las conclusiones a pedidos con entrega observada.

## 4 Hallazgos del análisis exploratorio

El retraso se concentra en ciertos meses, en el Nordeste y en Río de Janeiro, y depende del plazo prometido; todo eso se conoce al aprobar el pago. Los gráficos están en notebooks/02_eda_graficos.ipynb y siempre muestran la tasa de retraso, no la cantidad, para comparar grupos de distinto tamaño.

| Hallazgo | Cifra | Implicación para el modelo |
| --- | --- | --- |
| El retraso cambia mucho en el tiempo | 12,4 % en Black Friday 2017 (con 63 % más volumen), 19,0 % en marzo de 2018, 1,2 % en junio de 2018 | Validación temporal, nunca un split al azar; el mes de compra no entra |
| La geografía manda | Alagoas 21,5 % y Maranhão 17,6 %, contra 4,5 % en São Paulo; Río tiene el 12,8 % de los pedidos y el 22,9 % de los retrasos | Estado del cliente y distancia cliente-vendedor como variables |
| El vendedor de otro estado pesa solo fuera de SP | Fuera de SP: 3,5 % con vendedor local contra 8,8 % con vendedor de otro estado; en SP no hay diferencia | Interacción que la regresión logística no capta sin un término explícito |
| El plazo prometido es un margen de seguridad | Con 10 días o menos falla 1 de cada 10 veces; desde 21–25 días, más plazo es menos retraso | promised_days entra al modelo; su relación no es lineal |

Dos hallazgos más: el flete sobre el precio no separa (6,5 % a 7,0 % en todos los quintiles), y los pedidos con dos o más vendedores se retrasan 1,0 % contra 6,9 %, una rareza que queda como hipótesis abierta. Son asociaciones, no causas: ningún gráfico prueba que alargar el plazo reduzca el retraso.

## 5 Metodología y validación temporal

El modelo predice en el momento en que se aprueba el pago y solo usa información disponible en ese instante. La aprobación llega con una mediana de 21 minutos después de la compra, y predecir al comprar rinde casi igual (ROC-AUC 0,702 contra 0,701 en el baseline).

Sin fuga de datos. Quedan fuera las fechas de despacho y entrega, el estado final del pedido, las reseñas y cualquier duración calculada con esas fechas. Hay tests que fallan si alguna de esas columnas entra al modelo, y la API rechaza con error 422 cualquier campo que no conoce.

Variables. Plazo prometido, distancia cliente-vendedor, feriados nacionales de los 7 días siguientes, día y hora de compra, horas hasta la aprobación, cantidad de ítems y de vendedores, precio, flete, flete sobre precio, peso, volumen, cuotas, estado del cliente, tipo de pago y si algún vendedor está en otro estado. Los pedidos con varios ítems o vendedores se agregan con reglas explícitas (src/features.py).

Validación temporal en tres cohortes:

| Cohorte | Período | Pedidos | Tasa de retraso | Uso |
| --- | --- | --- | --- | --- |
| Train | ene 2017 – mar 2018 | 61.276 | 7,8 % | Entrenar |
| Validación | mar – may 2018 | 15.620 | 6,8 % | Calibrar y fijar los cortes de riesgo |
| Test | fines de may – ago 2018 | 19.293 | 3,5 % | Evaluar una sola vez |

El modelo se elige con validación cruzada temporal de 4 tramos sobre train + validación: cada tramo se predice con un modelo entrenado solo con lo anterior. Las transformaciones que aprenden de los datos se ajustan solo con los datos de entrenamiento.

## 6 Modelos comparados y elección

Elegimos una regresión logística con selección de variables por la validación cruzada temporal y la regla de desempate por simplicidad, fijada antes de mirar el test. Los árboles alcanzan PR-AUC CV de 0,215 y la logística 0,212, dentro de la tolerancia de 0,01. La evaluación posterior en test confirma que la logística conserva mejor su discriminación (ROC-AUC 0,707 frente a 0,579 de los árboles con selección); el test no determina la elección.

| Modelo | PR-AUC CV | ROC-AUC CV | PR-AUC test | ROC-AUC test |
| --- | --- | --- | --- | --- |
| Logística, baseline | 0,206 | 0,701 | 0,072 | 0,700 |
| Logística, selección (elegida) | 0,212 | 0,711 | 0,073 | 0,707 |
| Árboles (HistGradientBoosting), selección | 0,215 | 0,710 | 0,051 | 0,579 |
| Árboles, todas las variables | 0,209 | 0,706 | 0,046 | 0,557 |

- Por qué PR-AUC: con pocos retrasos, la ROC-AUC se ve optimista. El piso de la PR-AUC es la tasa de retraso, 0,035 en test, así que 0,073 es el doble que el azar.

- Regla de empate: si dos modelos quedan a menos de 0,01 de PR-AUC, gana el más simple. Se fijó antes de mirar el test.

- Por qué cayeron los árboles: Olist acortó los plazos prometidos (mediana de 25 días en train, 21 en test). Los árboles aprenden cortes fijos sobre el plazo que dejan de servir; la logística usa una relación lineal y se sostiene. Sin el plazo, los árboles caen a ROC-AUC 0,45.

- Variables elegidas: las del baseline sin el mes de compra, más la distancia y los feriados próximos. El mes empeora todos los modelos porque aprende meses que no se repiten. Región, categoría e historial del vendedor no mejoraron la validación.

- Feriados: suben la PR-AUC de la validación cruzada de 0,209 a 0,212, con signo negativo y sin efecto en test, donde no hay feriados. Es una decisión débil y se revisa con el monitoreo.

## 7 Calibración y niveles de riesgo

Las probabilidades del modelo se recalibran y se convierten en tres niveles mediante cortes elegidos por volumen en validación y luego congelados. En test, el nivel alto reúne el 11,2 % de los pedidos y el 24,1 % de los retrasos, con una tasa de retraso 2,15 veces la media.

| Nivel | % de pedidos | Tasa de retraso | % de los retrasos | Lift |
| --- | --- | --- | --- | --- |
| Alto | 11,2 % | 7,5 % | 24,1 % | 2,15 |
| Medio | 22,0 % | 5,6 % | 35,2 % | 1,60 |
| Bajo | 66,8 % | 2,1 % | 40,7 % | 0,61 |

Por qué recalibrar. Solo el 6,8 % de los pedidos se atrasa. Con tan pocos casos, el modelo corría el riesgo de ignorarlos, así que lo entrenamos con pesos de clase balanceados. Eso le enseña a ordenar bien, pero le hace creer que la prevalencia es mucho más alta: el baseline predecía un riesgo medio de 48,6 % cuando la tasa real de validación era 6,8 %. Ordenar bien es discriminación; que el número coincida con lo que pasa es calibración, y son propiedades distintas, como en una regla de predicción clínica. La calibración de Platt, ajustada en validación, es una transformación monótona: no cambia el orden de los pedidos, así que arregla la calibración sin tocar la discriminación, y lleva el riesgo medio a 6,8 %.

Por qué niveles por volumen. Los cortes se fijan en validación para que el 5 % de pedidos con mayor puntaje quede en alto y el 15 % siguiente en medio. Son objetivos de volumen de esa cohorte, no cuotas garantizadas para meses futuros: al aplicar cortes congelados, el volumen cambia si cambia la distribución de los puntajes. Logística debe planificar con el volumen efectivamente observado y monitorear su evolución.

Modelo final. Después de fijar calibrador y cortes, el mismo pipeline se reentrena con train + validación (76.896 pedidos) para aprender de los meses más recientes. En test ordena igual que el entrenado solo con train (ROC-AUC 0,707).

Dos advertencias. En test el riesgo medio calibrado es 9,3 % frente a una tasa real de 3,5 %: las probabilidades sobreestiman el retraso en la nueva cohorte. El modelo final fue reentrenado y conserva el calibrador ajustado con el modelo anterior, y también cambió la prevalencia. La discriminación se sostiene, pero el valor absoluto y el volumen por nivel deben revisarse. El nivel alto, diseñado para 5 %, marca 11,2 %. El monitoreo mensual implementado en src/monitoreo.py muestra 2,0 % en junio, 13,9 % en julio y 18,5 % en agosto de 2018.

Monitoreo implementado y ampliación propuesta. src/monitoreo.py calcula por mes, sobre test, el porcentaje en alto, la tasa de retraso del nivel y del mes, el lift y la proporción de retrasos captados. El volumen de alertas mide carga operativa y cambios de mezcla; por sí solo no demuestra descalibración. Proponemos agregar la razón O/E: número de retrasos observados dividido por la suma de probabilidades predichas, usando pedidos de una misma cohorte con seguimiento suficiente. O/E cercano a 1 indica coincidencia agregada, sin garantizar calibración en todos los grupos.

La propuesta operativa es revisar semanalmente el volumen de alertas y mensualmente la calibración y el lift, una vez observadas las entregas. Se propone evaluar una recalibración mensual con los últimos 2 o 3 meses maduros, validando su efecto antes de adoptarla. Si cae el lift de manera sostenida, corresponde revisar o reentrenar el modelo. El reporte semanal, O/E, sus intervalos y la recalibración programada son ampliaciones pendientes; no son funciones automatizadas del monitoreo actual.

## 8 Producto y entregables

El modelo calibrado se guarda en un solo archivo con el modelo, el calibrador y los cortes. La app local y la API usan predict_risk y existen pruebas de consistencia. La versión web utiliza resultados históricos precalculados con las mismas funciones: permite explorar la demostración sin cargar los CSV originales ni ejecutar nuevas inferencias en la nube.

| Pieza | Quién la usa | Para qué | Dónde | Estado |
| --- | --- | --- | --- | --- |
| App de Streamlit | Equipo de operaciones | Explorar pedidos históricos, niveles de riesgo, explicaciones y vendedores recomendados. La web usa resultados precalculados; el modo local usa el modelo. | app/ | Implementada; versión web publicada |
| API /predict | Otros sistemas de Olist | Pedir el riesgo de hasta 1.000 pedidos por llamada | api/, docs/api.md | Hecha, en Docker |
| Tablero de Power BI | Gerencia | Analizar entregas e indicadores históricos. El ETL genera un modelo estrella con target único; sus tablas incluyen datos posteriores a la aprobación y no son entrada del clasificador. | dashboard/Olist_Dashboard (1).pbix; scripts/unificacion/ | Archivo incorporado y actualizado el 27/10/2026 |

App publicada. https://soyhenry-datacraft.streamlit.app/ permite explorar resultados históricos. Los archivos de app/datos_web/ se generan con el modelo congelado. El modo local conserva el cálculo con los datos y el artefacto; la web no constituye un servicio de predicción de pedidos nuevos.

Power BI. El entregable está en dashboard/Olist_Dashboard (1).pbix y fue actualizado el 27 de octubre. El proceso de unificación produce tablas de hechos de pedidos, ítems y pagos, dimensiones y una tabla plana para EDA. fact_orders conserva los 99.441 pedidos y solo 96.203 tienen target válido. El dashboard es un entregable descriptivo independiente del modelo predictivo; para el detalle de visuales e indicadores se consulta el archivo PBIX.

Seguimiento de experimentos. python -m src.training --mlflow registra una corrida padre y una hija por candidato en el experimento datacraft-retraso-pedidos. Guarda hiperparámetros, métricas por tramo de validación, métricas finales y artefactos. Permite comparar los cuatro candidatos y conservar trazabilidad sin cambiar el modelo ni las cifras; las corridas locales se excluyen de Git. Implementación: src/experiment_tracking.py.

Calidad y documentación. La suite de pytest verifica, entre otros aspectos, target, prevención de fuga, partición temporal, calibración, consistencia de app y API, recomendador, unificación y promesas por ruta. GitHub Actions ejecuta pruebas en push y pull requests a developer y main, con Python 3.10 y 3.13. El estado de cada ejecución debe consultarse en Actions; no se confunde la existencia de tests con una nueva ejecución validada. El repositorio incluye docs/manual_usuario.md, docs/api.md y videos de demostración de la app y la API. El video de la API es una simulación de integración, no un despliegue en Olist.

La API. Está hecha en FastAPI, con validación de Pydantic: un campo que no conoce, como la fecha de entrega, se rechaza con error 422, así que la fuga de datos se frena en la entrada. También se rechazan un estado que no sea una de las 27 siglas de Brasil y un tipo de pago que no exista en Olist. Sin modelo cargado responde 503. GET /health informa con qué se entrenó el modelo y sus cortes, y /docs es la documentación interactiva. Además, POST /predict/pedido recibe solo el order_id y arma las variables con el mismo pipeline del entrenamiento, así que nadie tiene que calcularlas a mano y no hay una segunda implementación que pueda dar distinto; con los 3 pedidos de la demo da exactamente las mismas probabilidades. Su predicción solo es honesta con pedidos del período de test (desde el 27 de mayo de 2018), porque los anteriores el modelo final ya los vio al entrenar. Corre en Docker con scikit-learn fijo en 1.7.2, y el modelo se monta como volumen porque los archivos del modelo no se versionan.

Comprobado de punta a punta. Tres pedidos reales del período de test dan el mismo resultado en la app y en la API: uno de Alagoas marcado alto (54 %) que llegó 7 semanas tarde, uno de São Paulo marcado bajo (2 %) que llegó una semana antes, y uno de Río marcado alto (30 %) que llegó a tiempo, una falsa alarma. Los genera scripts/demo_pedidos.py.

Datos para el equipo. La base está cargada en BigQuery y la vista pedidos_analitica replica el target y las variables del baseline para 96.189 pedidos. Sirve para consulta y validación histórica; el modelo no lee de ella. La vista filtra pedidos entregados de 2017–2018, exige fecha de entrega real y todavía no incorpora distancia ni feriados. No puede utilizarse directamente para puntuar pedidos recién aprobados: se necesita una vista de inferencia independiente, sin esos filtros y sin exigir target.

Un día con el modelo. En el período de test (27 de mayo a 29 de agosto de 2018, 95 días), el nivel alto marca unos 23 pedidos al día. Entre ellos se atrasan en promedio 1,7; revisando 23 pedidos al azar se encontrarían 0,8. La app ya muestra una acción por nivel, y en la práctica significa:

| Nivel | En la app | Qué hace el equipo |
| --- | --- | --- |
| Alto | Revisar hoy | Confirmar que el vendedor despachó; si el envío va tarde, avisar al cliente antes de que reclame |
| Medio | Seguir de cerca | Revisar el seguimiento si se acerca la fecha prometida |
| Bajo | Seguimiento habitual | Nada extra: el seguimiento de siempre |

Como en 7 de cada 10 pedidos tarde el vendedor despachó a tiempo y el problema estuvo después del despacho (sección 9), el aviso a tiempo es probablemente la acción que más protege la reseña. Para medirlo proponemos una prueba A/B de unos 2 meses sobre los pedidos de riesgo alto y medio: la mitad con aviso preventivo y la otra mitad sin aviso, comparando reseñas de 1 estrella y reclamos (diseño en la sección 11). No estimamos un retorno en reales porque no tenemos el costo de una reseña mala ni datos de recompra: eso es lo que mediría el piloto.

## 9 Sistema de recomendación

La consigna pide un sistema de recomendación basado en el historial del cliente, pero en Olist el 96,9 % de los clientes compra una sola vez. El historial individual disponible es insuficiente para sostener una recomendación colaborativa general. Por eso la recomendación es contextual: por categoría buscada y estado de destino, se muestran los 3 vendedores con menor riesgo medio según el clasificador. Se puntúan sus pedidos anteriores al test con el modelo congelado, sin reentrenarlo. La tasa histórica de retraso y la popularidad son baselines.

Por qué recomendar por retraso. El retraso es lo que más daña la reseña: el 53,7 % de las reseñas de pedidos tardíos es de 1 estrella, contra 6,6 % a tiempo. Sobre la recompra la señal es débil: de los clientes cuyo primer pedido llegó tarde, el 2,5 % vuelve a comprar en 6 meses, contra 2,9 % de los que lo recibieron a tiempo, una diferencia no concluyente (IC 95 % de −1,0 a +0,1 puntos). El 97 % de los que recibieron a tiempo tampoco volvió, así que el retraso no explica por qué casi nadie repite; medir si recomendar vendedores puntuales mejora la recompra es lo que haría la prueba A/B.

Resultado. Se mide en el mismo período de prueba del clasificador, simulando la pantalla del cliente en cada combinación de categoría y estado (211 combinaciones con al menos 6 vendedores para elegir):

| Criterio | Precision@3 | Recall@3 | Retraso de los recomendados |
| --- | --- | --- | --- |
| Clasificador (recomendador) | 71,1 % | 28,0 % | 2,58 % |
| Historial del vendedor | 69,8 % | 28,4 % | 2,79 % |
| Popularidad | 69,4 % | 27,7 % | 2,75 % |
| Azar | 66,1 % | 27,0 % | 3,14 % |

Un vendedor es "bueno" si en la prueba se retrasó menos que la media (3,5 %); Precision@3 es la parte de los 3 recomendados que es buena y Recall@3 la parte de los buenos disponibles que quedó entre los 3. Quien compra a los recomendados vive un retraso de 2,6 % contra 3,1 % al azar (−0,56 puntos, IC 95 % de −0,84 a −0,29), un riesgo relativo de 0,82. Los tres criterios superan al azar; el clasificador tiene el menor retraso, pero no se distingue ni del historial ni de la popularidad (−0,21 puntos, IC 95 % de −0,58 a +0,14). Los recomendados no son más caros, están a la misma distancia del cliente y prometen unos 2 días más: el sistema recomienda a quien cumple lo que promete, no a quien entrega más rápido.

Límites. El vendedor explica solo una parte del retraso: de los pedidos que llegaron tarde, solo el 27,7 % tuvo un vendedor que despachó después de su plazo límite. En los demás el vendedor despachó a tiempo y el retraso se produjo después, pero sin datos de la transportadora no se puede decir dónde (transporte, dirección, cliente ausente). Los candidatos son los vendedores que vendieron, no el catálogo de los que ofrecían el producto, que no está en los datos. El 73 % de los pedidos de prueba tiene un vendedor con historial suficiente para recomendarlo. Y es observacional: que los recomendados se retrasen menos no prueba que recomendarlos reduzca el retraso; eso lo mediría una prueba A/B.

El código y las cifras están en src/recomendador.py (python -m src.recomendador), el detalle en docs/recomendador.md y la decisión en docs/decisiones.md (punto 12). La app de Streamlit lo muestra en la pestaña "Vendedores recomendados", como una ayuda para revisar opciones y no como precio o stock actual.

## 10 Limitaciones

El modelo sirve para priorizar, no para predecir con certeza, y tiene estos límites:

- Discriminación modesta: ROC-AUC 0,707 y PR-AUC 0,073. Con lo que se sabe al aprobar el pago no se predice mucho más.

- Deriva: la prevalencia cae a la mitad en test y Olist acortó los plazos. Las probabilidades se inflan y el nivel alto marca más del doble de lo previsto.

- Solo pedidos con entrega observada. Los pedidos cancelados, no entregados o sin fechas suficientes no se etiquetan como puntuales. El desempeño y la tasa de retraso reportados describen las entregas observadas; no permiten cuantificar pérdidas ni el riesgo completo de pedidos que nunca llegaron. Se requiere información del desenlace y de los motivos de cancelación para ampliar el análisis.

- Asociaciones, no causas: ningún hallazgo prueba que alargar un plazo o cambiar de vendedor reduzca el retraso.

- Datos históricos de Brasil (2017–2018), sin clima, tránsito ni capacidad operativa.

- La API lee pedidos históricos: por order_id arma las variables desde los CSV de 2017–2018, pero todavía no lee pedidos nuevos desde una base.

- Prototipo educativo, no validado para uso en producción.

## 11 Próximos pasos

El paso que más importa es probar el modelo en la operación; los demás lo mantienen vigente y lo conectan a los datos de Olist.

- Validar en operación con una prueba A/B: medir si vigilar los pedidos de riesgo alto reduce reclamos o reseñas de 1 estrella.

- Evaluar una recalibración mensual con los últimos 2 o 3 meses de pedidos entregados y con seguimiento suficiente. En test el riesgo medio es 9,3 % frente a 3,5 % observado y el volumen alto supera el 5 % de diseño. El seguimiento mensual del nivel alto ya está implementado; el reporte semanal, O/E y el proceso automático de recalibración quedan como próximos desarrollos.

- Llevar el pipeline de datos a BigQuery. Primero, cargar en un dataset separado el modelo estrella generado por scripts/unificacion/ y conectar Power BI. Segundo, construir una vista de inferencia para pedidos aprobados: debe admitir pedidos aún no entregados, no exigir fecha real de entrega ni target y usar solo datos disponibles al aprobar el pago. Debe incorporar distancia, feriados y las mismas reglas de agregación, imputación y nombres del pipeline. Tercero, adaptar /predict/pedido y comprobar paridad de variables y predicciones con Python. La vista histórica pedidos_analitica se conserva para evaluación; sus filtros de entrega y período no sirven para pedidos nuevos.

- Pedir a Olist la transportadora y el seguimiento de cada pedido: en 7 de cada 10 pedidos tarde el vendedor despachó a tiempo, y sin esos datos no se puede saber dónde se perdió el tiempo; es lo que más le falta al modelo. También sumar señales del EDA: tiempo histórico de la ruta entre estados, fin de mes y la interacción región × vendedor de otro estado.

Diseño propuesto de pruebas A/B. Los siguientes cálculos son escenarios de planificación, no experimentos realizados ni impactos comprobados. Usamos asignación 1:1, contraste bilateral de dos proporciones independientes, significancia de 5 %, potencia de 80 % y aproximación normal sin corrección de continuidad. Los efectos objetivo son supuestos de diseño. Las duraciones son pisos de captación derivados de unos 203 pedidos diarios del test; falta sumar el tiempo de entrega y observación de reseñas, y ajustar por elegibilidad, falta de reseñas y repetición de vendedores. La fórmula y los resultados reproducibles se incluyen en el anexo.

| Prueba | Qué recibe el grupo con cambio | Métrica principal | Efecto que se busca detectar | Pedidos por grupo | Duración |
| --- | --- | --- | --- | --- | --- |
| Aviso de logística | Aviso preventivo a los pedidos de riesgo alto y medio | % de reseñas de 1 estrella | De 7,0 % a 4,9 % (−30 %) | 1.991 observaciones con reseña | Al menos 60 días de captación; más seguimiento |
| Recomendador | Ve primero los 3 vendedores recomendados | % de pedidos tarde | De 3,1 % a 2,6 % | 17.385 pedidos | Al menos 172 días a 203 pedidos/día; más si la elegibilidad es parcial |
| Promesa por ruta | La fecha según la regla por ruta | % de pedidos tarde | De 3,5 % a 1,3 % | 759 pedidos | Al menos 8 días de captación; más seguimiento y prueba de conversión |

- Con solo el nivel alto, detectar una caída de reseñas de 1 estrella de 8,5 % a 6,0 % requiere aproximadamente 1.688 observaciones por grupo. A unos 23 pedidos diarios, la captación mínima sería de unos 147 días, antes de considerar pedidos sin reseña y seguimiento. Incluir nivel medio amplía el volumen elegible, pero requiere comprobar la tasa de referencia de ese grupo combinado.

- La promesa necesita una métrica de resguardo: la conversión. Una caída relativa de 5 % equivale a pasar de 2,0 % a 1,9 %, o de 3,0 % a 2,85 %. Con los mismos supuestos estadísticos, requiere aproximadamente 300.137 o 198.101 visitas por grupo, respectivamente. Las visitas y la conversión no están en el dataset; Olist debe aportarlas. La regla solo se adoptaría si cumple el criterio previamente acordado para retrasos y conversión.

- La recompra se mediría como resultado secundario. Su tasa baja y la ventana de seguimiento de seis meses implican un piloto más largo o con mayor volumen; su tamaño debe calcularse con un efecto objetivo y población elegible definidos.

Tres usos con distinto grado de desarrollo. La app permite demostrar cómo logística podría priorizar pedidos después de la aprobación; la pestaña de vendedores muestra una recomendación contextual con datos históricos antes de una compra hipotética. Ambos son prototipos educativos, sin integración operativa en Olist. La promesa P95 por ruta es un análisis complementario implementado en src/promesas_ruta.py: baja los incumplimientos de 3,49 % a 1,25 %, con promesa media de 27,5 días frente a 22,0. No usa la probabilidad del clasificador ni modifica la app o la API; aplica percentiles históricos por ruta. No supera de forma concluyente a sumar 6 días a todas las promesas (1,10 %). Antes de adoptarla se requiere validar también conversión.

## 12 Equipo y aportes

El proyecto lo desarrolló el equipo DataCraft, con una parte del pipeline por integrante.

| Integrante | Rol | Aportes principales |
| --- | --- | --- |
| Agustina Baez | Data Scientist · Scrum Master | Proceso del equipo, revisión y aprobación de los pull requests |
| Cristian Atrio | Data Scientist | Entorno y repositorio, baseline, features, validación temporal, modelo principal y modelo final reentrenado, feriados en el modelo, métricas, monitoreo del nivel alto, app de Streamlit con su modo web y seguimiento de experimentos en MLflow |
| Alexis Sircelj | Data Scientist | Base de datos, esquema estrella para Power BI y EDA con gráficos |
| Mathias Chaparro | Data Scientist | Exploración del target, control de fuga, auditoría de datos y variables del Sprint 2 (carga del vendedor y feriados) |
| María Camila Uribe Noreña | Data Scientist | Período de estudio, base en BigQuery, calibración, niveles de riesgo, API, Docker y sistema de recomendación |

## 13 Organización del equipo y retrospectiva

Trabajamos con dos sprints: del 29 de septiembre al 14 de octubre y del 15 al 28 de octubre de 2026. Usamos Trello y un planner para distribuir tareas y coordinar reuniones. Agustina desempeñó el rol de Scrum Master, organizó el backlog y acompañó el seguimiento y la integración mediante pull requests.

La retrospectiva destaca la distribución de responsabilidades, la comunicación y el apoyo entre integrantes. Las mejoras acordadas son mover las tarjetas completadas a Hecho, fijar fechas y horarios de entrega para tareas dependientes y reservar tiempo para revisar, integrar y documentar antes de cada demo. Las evidencias de cronogramas, tablero y ceremonias están en docs/retrospectiva.md. El período correcto del segundo sprint es del 15 al 28 de octubre, según la aclaración del documento.

## 14 Entregables y reproducción

El código y las decisiones de este informe corresponden a developer, commit 2a842b4, revisado al 27 de octubre de 2026. Los enlaces fijados al commit conservan esa versión aunque la rama cambie. Los CSV originales, modelos entrenados y reportes generados localmente no se incluyen en Git; para reproducir las métricas se requieren los nueve CSV de Kaggle y ejecutar el pipeline. Los tests utilizan datos sintéticos y no sustituyen una reproducción completa con los datos originales.

- [Repositorio y guía de instalación](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/README.md)

- [Aplicación web](https://soyhenry-datacraft.streamlit.app/)

- [Dashboard Power BI](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/dashboard/Olist_Dashboard%20%281%29.pbix)

- [Manual de usuario](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/docs/manual_usuario.md)

- [Documentación de API](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/docs/api.md)

- [Video de la app](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/docs/demo/demo_app.mp4)

- [Video de la API y alcance de la simulación](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/docs/demo/README.md)

- [Métricas y validación](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/docs/metricas_evaluacion.md)

- [Decisiones de modelado](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/docs/decisiones_modelos.md)

- [Recomendador](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/docs/recomendador.md)

- [Promesas por ruta](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/docs/promesas_ruta.md)

- [ETL y esquema estrella](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/scripts/unificacion/README.md)

- [SQL histórico](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/sql/02_tabla_analitica.sql)

- [Retrospectiva y evidencias Scrum](https://github.com/datacraft-henry/proyecto-final-/blob/2a842b4/docs/retrospectiva.md)

- [Pruebas automáticas](https://github.com/datacraft-henry/proyecto-final-/actions)

- [Fuente de datos](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

Secuencia principal de reproducción desde la raíz del repositorio, con el entorno preparado según README.md y los CSV en data/raw/:

```bash
python -m pip install -r requirements.txt
```

```bash
python -m pytest -q
```

```bash
python -m src.training --mlflow
```

```bash
python -m src.umbral
```

```bash
python -m src.monitoreo
```

```bash
python -m src.recomendador
```

```bash
python -m src.promesas_ruta
```

```bash
python scripts/unificacion/unificar_olist.py data/raw data/processed/olist_limpio
```

```bash
streamlit run app/streamlit_app.py
```

```bash
uvicorn api.main:app --reload
```

```bash
python -m mlflow ui
```

Para consultar el artefacto congelado de la entrega deben conservarse también su versión y sus reportes. Volver a entrenar reproduce el procedimiento; pueden existir diferencias numéricas menores por entorno. Las instrucciones de Docker, configuración de API y generación de datos web se encuentran en el README y docs/api.md.

## 15 Anexo de planificación estadística

Para cada prueba de dos proporciones, p0 es la tasa de referencia, p1 la tasa objetivo y p̄ = (p0 + p1) / 2. Con z = 1,959964 para α bilateral de 0,05 y zβ = 0,841621 para potencia de 0,80, el tamaño por grupo se aproxima por n = techo{[z × √(2p̄(1−p̄)) + zβ × √(p0(1−p0)+p1(1−p1))]² / (p0−p1)²}. La igualdad de tamaños, independencia y disponibilidad del resultado son supuestos de cálculo. No se aplican corrección de continuidad, ajustes por pérdidas, multiplicidad o agrupamiento.

Con los porcentajes redondeados de los escenarios, se obtienen: aviso alto y medio, 0,070 frente a 0,049, n = 1.991 por grupo; recomendador, 0,031 frente a 0,026, n = 17.385; promesa por ruta, 0,035 frente a 0,013, n = 759; aviso solo alto, 0,085 frente a 0,060, n = 1.688. Las tasas objetivo no son efectos causales demostrados. El 7,0 % de reseñas de referencia del grupo alto y medio debe confirmarse en la población del piloto.

Las duraciones mínimas usan techo(2n / volumen diario elegible): aproximadamente 60 días para el aviso alto y medio si los 203 × 0,332 pedidos diarios produjeran reseña; 172 días para el recomendador si todos los 203 pedidos fueran elegibles; y 8 días para medir incumplimientos con la promesa. El 73 % con historial suficiente no equivale a cobertura completa del catálogo ni asegura la elegibilidad del piloto. Las duraciones reales aumentan por cobertura parcial, pedidos sin reseña, maduración de resultados y la medición de conversión.

La razón O/E se propone como ampliación. El informe reporta la descalibración agregada verificable en la documentación del modelo: probabilidad media de 9,3 % frente a retraso observado de 3,5 % en test. El módulo actual no genera una tabla mensual O/E ni intervalos de confianza. Antes de publicarlos, se debe incorporar un cálculo reproducible con cohorte, seguimiento, denominador y método de intervalos definidos. El monitoreo sobre validación con el modelo final es optimista porque ese modelo ya se entrenó con esos pedidos.

# Decisiones del equipo

Registro corto de lo que acordamos y por qué. Cada decisión nueva se agrega abajo con fecha. Si algo cambia, se tacha y se agrega la versión nueva en lugar de borrarla.

| Estado | Significado |
|---|---|
| Acordada | El equipo la tomó y se aplica |
| Pendiente | Hay una propuesta, falta confirmar |

## 1. Alcance: el centro del proyecto es el clasificador de retrasos

- **Estado:** Pendiente de confirmar con el mentor.
- **Decisión:** el entregable principal es un clasificador binario que estima si un pedido va a llegar después del día prometido. El análisis comercial (ventas, categorías, vendedores) queda como anexo descriptivo, no como un segundo modelo.
- **Por qué:** es el problema con target claro, datos suficientes y una acción concreta (priorizar seguimiento logístico). Dos modelos en paralelo dividen el tiempo del equipo.

## 2. Target: `pedido_retrasado` por día calendario

- **Estado:** Acordada.
- **Decisión:** 1 si `order_delivered_customer_date` cae en un día posterior a `order_estimated_delivery_date`. Solo pedidos con estado `delivered` y ambas fechas presentes.
- **Por qué:** la fecha estimada viene siempre a las 00:00. Comparando con la hora, un pedido entregado el mismo día prometido cuenta como tarde. Con los datos de Kaggle: 8,11 % comparando timestamps contra 6,77 % por día calendario (96.470 pedidos entregados). Usamos 6,77 %.
- **Implementación:** una sola función reutilizable en `src/`. Nadie recalcula el target en un notebook.
- **Actualización (2026-10-05):** con el período de estudio de la decisión 8, la población queda en 96.203 pedidos y la tasa en 6,79 %. En el código la columna se llama `pedido_retrasado` (antes `is_late`).

## 3. Momento de predicción: al aprobar la compra

- **Estado:** Acordada (alineada al README). La predicción al momento de la compra queda como alternativa.
- **Decisión:** el modelo predice cuando Olist aprueba el pago (`order_approved_at`). Se usan solo pedidos con fecha de aprobación, el orden temporal se toma por esa fecha y se suma la variable `approval_hours` (horas entre compra y aprobación).
- **Alternativa:** `--anchor compra` predice al momento de la compra, sin usar nada de la aprobación. Es más estricta y permite avisar antes.
- **Por qué no importa mucho para el baseline:** con los datos de Kaggle las dos variantes rinden casi igual (test desde fines de mayo 2018):

| Momento | Pedidos test | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| Aprobación (por defecto) | 19.293 | 0,701 | 0,072 | 0,77 | 0,061 | 0,113 |
| Compra | 19.363 | 0,702 | 0,071 | 0,77 | 0,061 | 0,113 |

  La aprobación suele llegar a los minutos (mediana 21 min), así que casi no agrega información. Si un modelo posterior gana claramente con la aprobación, se mantiene; si no, conviene la compra porque avisa antes.
- **Implementación:** `src/features.py` (parámetro `anchor`). Hay un test anti-fuga por cada variante.

## 4. Sin fuga de datos

- **Estado:** Acordada.
- **Decisión:** el modelo solo usa información disponible en el momento de predicción. Quedan afuera las fechas de despacho y entrega, el estado final del pedido, las reseñas y cualquier duración calculada con esas fechas. Con la variante "compra" también queda afuera `order_approved_at`.

## 5. Validación temporal

- **Estado:** Acordada.
- **Decisión:** se entrena con pedidos más viejos y se evalúa con los más nuevos, ordenados por la fecha del momento de predicción. Nunca split aleatorio. Las transformaciones se ajustan solo con train.
- **Métricas:** matriz de confusión, precision, recall, F1, ROC-AUC y PR-AUC. El umbral se elige con validación, no con test.
- **Pendiente:** el baseline actual divide solo en train y test y reporta las métricas con umbral fijo 0,5 directo en test. Falta separar un conjunto de validación (entre train y test, también por fecha) para elegir el umbral ahí y dejar el test solo para la evaluación final.
- **Actualización (2026-10-06):** `temporal_three_way_split` en `src/validation.py` separa train, validación y test por fecha. Se usa en `src/umbral.py` (decisión 9). Queda pendiente la validación cruzada temporal con varios cortes.

## 6. Flujo de trabajo con Git

- **Estado:** Acordada.
- `main`: solo entregas.
- `developer`: integración. Todo entra por pull request aprobado por Agustina.
- `feature/E#-descripcion`: una rama por tarea, creada desde `developer` actualizado. PR chicos y frecuentes.
- Commits con prefijo: `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
- La lógica reutilizable va en `src/`; los notebooks solo la usan para explorar y mostrar resultados. Cada notebook tiene un solo dueño para evitar conflictos.

## 7. Datos y archivos fuera de Git

- **Estado:** Acordada.
- Los CSV de Kaggle se descargan a `data/raw/` y no se suben. El `.gitignore` ignora todo `data/` (salvo los `.gitkeep`), y además `*.csv`, `*.zip`, `*.pdf`, `*.docx` y `*.xlsx` en cualquier carpeta, para que nadie suba datos o documentos personales por error.
- Antes de cada commit: revisar `git status` y no usar `git add .` sin mirar.

## 8. Período de estudio: compras de enero 2017 a agosto 2018

- **Estado:** Acordada (5 oct, chat del equipo).
- **Decisión:** entran solo los pedidos comprados desde el 1 de enero de 2017 hasta el 31 de agosto de 2018. Con el target de la decisión 2 quedan **96.203 pedidos y 6,79 % de retrasos**.
- **Por qué se excluye 2016:** son 267 pedidos entregados, con meses casi vacíos. No representan la operación.
- **Por qué se corta en agosto 2018 (sesgo de censura):** los datos terminan el 17 de octubre de 2018. Un pedido que todavía no llegó a esa fecha no se puede etiquetar, y si los que faltan son justamente los demorados, la tasa del final parece más baja de lo que es.
  - Septiembre y octubre de 2018 tienen 20 pedidos y ninguno se entregó: se excluyen.
  - A una compra del 31 de agosto le quedan 47 días hasta el corte, y el 99 % de los pedidos se entrega en 46 días o menos. Solo el 0,8 % tarda más.
  - Julio y agosto de 2018 tienen la misma proporción de pedidos entregados que los meses anteriores (97–98 %).
  - Conclusión: el sesgo de censura en julio y agosto es menor a un punto porcentual. La tasa baja de esos meses (3,4 % y 6,2 %) es real, no un efecto del corte.
- **Implementación:** `PERIOD_START` y `PERIOD_END` en `src/target.py`. `build_target` filtra el período por defecto; con `only_study_period=False` se ven todos los pedidos, solo para comparar.

## 9. Calibración y niveles de riesgo

- **Estado:** Propuesta (tarea 3). Se ratifica en el daily.
- **Problema:** el baseline entrena con `class_weight="balanced"`. Ordena bien los pedidos (ROC-AUC 0,73 en validación), pero sus probabilidades están infladas: predice 48,6 % de riesgo medio cuando la tasa real de validación es 6,8 %. Con el umbral fijo de 0,5 alerta sobre casi la mitad de los pedidos.
- **Decisión:**
  1. **Tres cohortes por fecha:** train (ene 2017 a mar 2018), validación (mar a may 2018) y test (fines de may a ago 2018). Calibración, cortes y umbral se eligen en validación; el test se usa una sola vez.
  2. **Recalibración de Platt** ajustada en validación. En validación el riesgo medio queda en 6,8 % (igual a la tasa real). En test predice 7,1 % frente a 3,5 % real, porque la prevalencia cae a la mitad: ninguna calibración fija aguanta un cambio así.
  3. **Niveles de riesgo por volumen de pedidos, no por probabilidad:** alto = 5 % de pedidos con más riesgo, medio = 15 % siguiente, bajo = el resto. Los cortes se fijan en validación y se aplican sin cambios. Así el volumen de alertas es estable aunque cambie la prevalencia.
  4. **Umbral por costos** (1 / (1 + r), con r = costo de un falso negativo / costo de un falso positivo) solo como análisis de sensibilidad con r = 5, 10 y 20, no como regla única.
- **Resultado en test:**

| Nivel | % de pedidos | Tasa de retraso | % de los retrasos | Lift |
|---|---|---|---|---|
| Alto | 6,2 % | 8,7 % | 15,5 % | 2,49 |
| Medio | 15,4 % | 5,0 % | 22,0 % | 1,43 |
| Bajo | 78,4 % | 2,8 % | 62,6 % | 0,80 |

- **Aplicado al modelo principal (decisión 10), en test:** riesgo medio calibrado 9,3 % frente a 3,5 % real (en validación 6,8 % = tasa real). Es el modelo final: reentrenado con train + validación, con el calibrador y los cortes fijados en validación (actualización 2026-10-08; reentrenado el 2026-10-20 con los feriados próximos).

| Nivel | % de pedidos | Tasa de retraso | % de los retrasos | Lift |
|---|---|---|---|---|
| Alto | 11,2 % | 7,5 % | 24,1 % | 2,15 |
| Medio | 22,0 % | 5,6 % | 35,2 % | 1,60 |
| Bajo | 66,8 % | 2,1 % | 40,7 % | 0,61 |

  En test el nivel alto cubre 11,2 % de pedidos y no 5 %: con los cortes fijos de validación, el modelo principal sube los puntajes de los meses nuevos. El volumen de alertas se mueve menos que con un umbral fijo, pero hay que vigilarlo en producción.
- **Por qué:** con la prevalencia mensual entre 1,2 % y 19 % (EDA), un umbral fijo dispara cantidades muy distintas de alertas según el mes. Logística necesita saber cuántos pedidos va a revisar.
- **Implementación:** `src/calibration.py` y `src/umbral.py` `python -m src.umbral` aplica la calibración al modelo principal, escribe `reports/umbral_calibracion_principal.json` y guarda el artefacto calibrado (modelo, features, calibrador y cortes) en `models/modelo_principal_calibrado.joblib`, que es el que servirá la API. `python -m src.umbral --modelo baseline` reproduce la tabla del baseline. `predict_risk(artefacto, pedidos)` devuelve `prob_retraso` y `nivel_riesgo`. Pruebas en `tests/test_calibration.py`.

## 10. Modelo principal: regresión logística con selección de features

- **Estado:** Propuesta (parte 2). Se ratifica en el daily.
- **Decisión:** el modelo principal es una regresión logística balanceada con las variables del baseline sin `purchase_month`, más la distancia cliente-vendedor (`max_distance_km`) y, desde el Sprint 2, los feriados nacionales de los 7 días siguientes a la aprobación (`n_national_holidays_next_7d`). Se elige con validación cruzada temporal de 4 tramos sobre train + validación, por PR-AUC medio, y ante un empate (menos de 0,01) gana el modelo más simple.
- **Resultado en test:** ROC-AUC 0,707 y PR-AUC 0,073, contra 0,700 y 0,072 del baseline. Los feriados suben el PR-AUC de la validación cruzada de 0,209 a 0,212 y no cambian el test.
- **Modelo final (2026-10-08):** se reentrena con train + validación en `src/umbral.py` y se guarda calibrado para la API. En test ordena igual que el de train (ROC-AUC 0,707). Detalle en `docs/decisiones_modelos.md`.
- **Por qué no árboles:** empatan en la validación cruzada pero en test caen a ROC-AUC 0,58, porque Olist acortó los plazos prometidos y los árboles no extrapolan ese cambio.
- **Implementación:** `src/feature_engineering.py` (features de ruta, producto e historial del vendedor, extensible con `FEATURE_BUILDERS`) y `src/training.py` (`python -m src.training`). Detalle y tabla completa en `docs/decisiones_modelos.md`.

## 11. Script de unificación para el dashboard

- **Estado:** Acordada (con la aprobación del PR por Agustina).
- **Decisión:** el modelo estrella y la tabla plana para Power BI y EDA se generan con `scripts/unificacion/unificar_olist.py`. `scripts/` es para scripts reproducibles que no son lógica del modelo; `src/` queda para código con tests. Los CSV de salida no se suben (punto 7).
- **Por qué:** el dashboard necesita relaciones entre tablas sin duplicar filas ni importes, y el script lo valida. Cada integrante regenera las tablas desde los CSV de Kaggle.
- **Alineación con el target (2026-10-20):** el script ya no calcula el target por su cuenta. `src/unificacion.py` (`agregar_target`) llama a `build_target` de `src/target.py` y une el resultado a `fact_orders` con un left join por `order_id`. Se conservan los 99.441 pedidos; 96.203 quedan etiquetados (tasa 6,79 %, igual que el modelo) y 3.238 en vacío (no entregados, sin fecha o fuera del período del punto 8). La columna pasa de `entrega_tardia` a `pedido_retrasado` (acordado con Agustina). Tests en `tests/test_unificacion.py`.
- **Límite:** `fact_orders` incluye datos posteriores a la aprobación (fechas de entrega, reseñas). Por el punto 4 no es entrada del modelo, que usa la tabla de `src/features.py`.
- **Implementación:** `python scripts/unificacion/unificar_olist.py data/raw data/processed/olist_limpio`. Detalle en `scripts/unificacion/README.md`.

## 12. Sistema de recomendación: vendedores ordenados por el clasificador

- **Estado:** Acordada en el grupo el 25/10/2026 (Camila, Cristian y Alexis). Se ratifica con la aprobación del PR.
- **Decisión:** el recomendador de vendedores deja de ser un prototipo y pasa a ser el sistema de recomendación del proyecto. Ordena a los vendedores por el riesgo medio que el clasificador calibrado (punto 9 y 10) predice para sus pedidos, sin reentrenarlo. La tasa histórica suavizada y la popularidad quedan como baselines.
- **Por qué contextual:** el 96,9 % de los clientes compra una sola vez, así que no hay historial por cliente para un filtrado colaborativo. Se recomienda según la categoría que busca el cliente y su estado de destino.
- **Evaluación:** en el período de test del clasificador (27/05 al 29/08/2018), con Precision@K, Recall@K y el retraso de los vendedores mostrados. Con K = 3, el retraso de los recomendados es 2,57 %, contra 2,79 % historial, 2,75 % popularidad y 3,14 % azar. La ventaja sobre el historial no es significativa y se dice así.
- **Implementación:** `src/recomendador.py` (`preparar_clasificador`, `puntaje_vendedores`, `recomendar`, `evaluar_recomendacion`), `python -m src.recomendador`, tests en `tests/test_recomendador.py`. Detalle en `docs/recomendador.md`. La pestaña en la app de Streamlit la suma Cristian con `recomendar`.

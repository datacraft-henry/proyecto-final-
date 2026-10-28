# Aporte de Mathias – Sprint 2: variables de contexto y evaluación

## Objetivo

Incorporar variables que ayuden a anticipar el riesgo de demora de un pedido sin usar información que todavía no existe al momento de hacer la predicción.

En este proyecto, el momento de predicción es la aprobación del pago.

## Variables incorporadas

### 1. Carga reciente del vendedor

Se incorporaron tres variables:

- `seller_recent_approved_orders_7d`
- `seller_recent_approved_orders_14d`
- `seller_recent_approved_orders_30d`

Cada una representa la cantidad de pedidos aprobados para el vendedor principal durante los 7, 14 o 30 días anteriores al momento de predicción.

La ventana usada es:

```text
[prediction_time - ventana, prediction_time)
```

Esto significa que:

- Incluye solamente pedidos aprobados antes del pedido actual.
- Excluye el pedido que se está evaluando.
- Excluye pedidos aprobados exactamente en el mismo instante.
- No usa entregas reales, retrasos, reviews ni el target de otros pedidos.

La idea es aproximar si el vendedor puede estar atravesando una mayor carga operativa.

### 2. Feriados nacionales de Brasil

Se incorporaron dos variables:

- `is_national_holiday`: vale `1` si la aprobación ocurrió en un feriado nacional de Brasil y `0` en caso contrario.
- `n_national_holidays_next_7d`: cuenta los feriados nacionales entre el día posterior a la aprobación y los siete días siguientes.

El calendario se conoce con anticipación, por lo que esta segunda variable también puede conocerse al aprobarse el pago.

Se usan únicamente feriados nacionales. No se incluyen feriados estatales, municipales ni Carnaval, porque no forman parte de una definición nacional uniforme dentro de la fuente utilizada.

Para calcularlos se agregó la dependencia `holidays` a `requirements.txt`.

## Prevención de fuga de información

Las variables usan solamente información disponible al aprobarse el pago:

- Fecha de aprobación del pedido.
- Identidad del vendedor principal.
- Aprobaciones anteriores del vendedor.
- Calendario nacional de feriados.

No usan:

- Fecha real de entrega.
- Fecha estimada de entrega.
- Estado final del pedido.
- Reviews.
- Target `pedido_retrasado`.

Se agregaron pruebas automáticas que verifican que modificar artificialmente las fechas de entrega no cambia estas variables.

## Evaluación de aporte al modelo

Las nuevas variables se evaluaron de manera aislada con Regresión Logística. Se mantuvo el mismo esquema de validación cruzada temporal del proyecto:

- Cuatro folds temporales.
- Cada fold entrena solo con pedidos anteriores al período que evalúa.
- Métrica principal: PR-AUC, apropiada porque los pedidos demorados son una clase minoritaria.
- El conjunto de test no se usó para elegir variables, para conservarlo como evaluación final imparcial.

La referencia es la selección actual del modelo logístico: variables base, sin `purchase_month`, más `max_distance_km`.

| Alternativa | PR-AUC medio | Cambio vs. referencia |
|---|---:|---:|
| Referencia | 0.209256 | 0.000000 |
| Carga 7 días | 0.207199 | -0.002057 |
| Carga 14 días | 0.208603 | -0.000653 |
| Carga 30 días | 0.208270 | -0.000986 |
| Feriado el mismo día | 0.208890 | -0.000366 |
| Feriados en los próximos 7 días | 0.211510 | +0.002254 |

## Conclusión

Las variables de carga reciente y el feriado del mismo día no mejoran el PR-AUC frente a la referencia.

La única variable que muestra una mejora es `n_national_holidays_next_7d`. La ganancia es pequeña, pero se observa en los cuatro folds temporales. Por ese motivo queda propuesta como candidata para el reentrenamiento final.

No se modificó `SELECTED_EXTRA_FEATURES` ni el modelo productivo. La decisión de incorporar esa variable al reentrenamiento final queda a cargo de Cristian, responsable del modelado.

## Validación técnica

Se verificó:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Resultado final:

```text
93 passed, 1 skipped
```

También se ejecutó `flake8` sobre los archivos modificados, ignorando los errores de longitud de línea ya existentes en el proyecto.
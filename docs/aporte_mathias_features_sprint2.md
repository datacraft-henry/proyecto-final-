# Aporte de Mathias – Sprint 2: features de contexto

## Objetivo

Incorporar variables que ayuden a anticipar el riesgo de demora de un pedido sin utilizar información que todavía no existiría al momento de hacer la predicción.

El momento de predicción del proyecto es la aprobación del pago.

## Variables incorporadas

### 1. Carga reciente del vendedor

Se creó la variable:

`seller_recent_approved_orders_30d`

Representa la cantidad de pedidos aprobados para el vendedor principal durante los 30 días anteriores al momento de predicción.

La ventana utilizada es:

`[prediction_time - 30 días, prediction_time)`

Esto significa que:

- Incluye solamente pedidos aprobados antes del pedido actual.
- Excluye el pedido que se está evaluando.
- Excluye pedidos aprobados exactamente en el mismo instante.
- No usa fechas de entrega, retrasos ni reseñas de otros pedidos.

La intención es aproximar si un vendedor podría estar atravesando un período de mayor carga operativa.

### 2. Feriado nacional de Brasil

Se creó la variable:

`is_national_holiday`

Vale `1` si la aprobación del pedido ocurrió en un feriado nacional de Brasil y `0` en caso contrario.

Se usan únicamente feriados nacionales. No se incorporan feriados estatales o municipales porque esa información no está disponible de forma uniforme para todos los pedidos.

Para calcularlos se agregó la dependencia `holidays` al archivo `requirements.txt`.

## Prevención de fuga de información

Las dos variables usan únicamente datos disponibles al aprobarse el pago:

- Fecha de aprobación del pedido.
- Identidad del vendedor principal.
- Historial de aprobaciones anteriores del vendedor.
- Calendario de feriados nacionales.

No usan:

- Fecha real de entrega.
- Fecha de entrega estimada.
- Estado final del pedido.
- Reviews.
- Target `pedido_retrasado`.

Se agregaron pruebas automáticas para verificar que modificar artificialmente las fechas de entrega no cambia ninguna de estas variables.

## Validación técnica

Se ejecutaron los tests de ingeniería de variables:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_feature_engineering.py -q
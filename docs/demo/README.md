# Demo de la API

Video demostrativo que muestra cómo podría utilizarse la API de DataCraft cuando se aprueba el pago de un pedido.

## Recorrido de la demostración

1. El cliente realiza una compra y se aprueba el pago.
2. El sistema dispone de los datos conocidos del pedido, como destino, plazo prometido y flete.
3. Una integración consulta la API y recibe el nivel de riesgo.
4. Logística prioriza la revisión del pedido.
5. El equipo consulta al vendedor y, si corresponde, informa al cliente y registra el seguimiento.

## Alcance de la simulación

El video es una simulación animada de integración con el pedido ficticio DC-1042. No es una grabación de una integración desplegada en Olist ni una nueva corrida del modelo.

La API calcula el riesgo. Los contactos con el vendedor, los mensajes al cliente y el registro del seguimiento representan acciones del equipo de logística, no automatizaciones implementadas por la API.

Un nivel alto permite priorizar la revisión, pero no asegura que el pedido se retrasará.

## Detalles del video

- Duración: 45 segundos.
- Resolución: 3840 × 2160 (4K), a 60 cuadros por segundo.
- Música instrumental de fondo.
- Video: [demo_api.mp4](demo_api.mp4).
- Documentación técnica: [API](../api.md).
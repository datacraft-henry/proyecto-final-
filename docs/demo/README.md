# Demo de la API

Video breve que muestra el funcionamiento de la API de DataCraft.

La demostración simula el ingreso de un pedido nuevo mediante el endpoint
`POST /predict`. La API recibe las variables disponibles al aprobarse el pago
y devuelve una probabilidad de retraso junto con un nivel de riesgo: alto,
medio o bajo.

El nivel permite priorizar qué pedidos conviene revisar primero. La probabilidad
se interpreta principalmente como un orden de riesgo, no como una certeza de
que el pedido se retrasará.

- Duración: 38 segundos.
- Video: [demo_api.mp4](demo_api.mp4).
- Documentación técnica: [API](../api.md).
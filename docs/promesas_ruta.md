# Promesas de entrega por ruta

## Objetivo

Como segundo uso de la señal logística del proyecto, se analizó una propuesta para definir los días prometidos de entrega según la ruta del pedido.

La idea es simple: no todas las rutas tardan lo mismo. Un envío desde São Paulo hacia Río de Janeiro no tiene el mismo comportamiento histórico que un envío dentro de São Paulo. Por eso se evaluó si una promesa ajustada por ruta podría reducir los incumplimientos de fecha.

Este análisis no modifica el modelo de clasificación, la API ni la aplicación de Streamlit. Es una propuesta complementaria para el equipo que define las fechas prometidas.

## Cómo se define una ruta

Cada pedido queda representado por una sola ruta:

```text
estado del vendedor principal -> estado del cliente
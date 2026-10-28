# Manual de usuario

Cómo usar el riesgo de retraso por pedido. La **app** es para el equipo de operaciones y la **API** es para otros sistemas. Las dos usan el mismo modelo y la misma función (`predict_risk`), así que dan siempre el mismo resultado.

La instalación completa está en el [README](../README.md#cómo-instalar-y-correr-el-proyecto). Para este manual alcanza con tener los 9 CSV de Kaggle en `data/raw/` y el modelo generado (`python -m src.umbral`).

## 1. Qué responde el modelo

En el momento en que se aprueba el pago, el modelo estima qué tan probable es que el pedido llegue **después del día prometido**. Solo usa lo que se sabe en ese momento: plazo prometido, distancia entre el vendedor y el cliente, estado, precio, flete, peso, forma de pago y feriados de los 7 días siguientes.

Cada pedido recibe un **nivel**:

| Nivel | Qué es | Qué hacer |
|---|---|---|
| 🔴 Alto | Los pedidos más riesgosos (pensado para el 5 %; en el período de test marca el 11 %) | **Revisar hoy:** confirmar que el vendedor despachó y, si el envío va tarde, avisar al cliente antes de que reclame |
| 🟡 Medio | Los siguientes en riesgo (pensado para el 15 %; en test marca el 22 %) | **Seguir de cerca:** revisar el seguimiento si se acerca la fecha prometida |
| 🟢 Bajo | El resto | **Seguimiento habitual** |

**Lo que hay que saber antes de usarlo:**

- **Lo importante es el nivel, no el número.** El puntaje sirve para ordenar. En los meses nuevos sale más alto que la tasa real de retraso: el promedio del modelo en test es 9,3 % y la tasa real fue 3,5 %.
- **Prioriza, no adivina.** En test, el nivel alto tuvo 7,5 % de retraso contra 3,5 % del total: más del doble que revisar al azar. Pero 9 de cada 10 pedidos en alto llegan a tiempo.
- **Los factores no son causas.** El detalle de un pedido muestra qué variables subieron o bajaron su riesgo según el modelo. No quiere decir que esa sea la razón del retraso.

## 2. La app (operaciones)

Para abrirla, desde la raíz del repo:

```bash
streamlit run app/streamlit_app.py
```

Se abre en el navegador (por defecto, http://localhost:8501). La primera carga tarda porque arma los pedidos desde los CSV.

La app muestra los pedidos del **período de test** (27/05 al 29/08/2018), que el modelo no vio al entrenar.

### Leer la pantalla, de arriba hacia abajo

1. **La frase de arriba** dice cuántos pedidos en riesgo alto hay para revisar en el período elegido.
2. **Las tres tarjetas** muestran cuántos pedidos hay en cada nivel, qué parte del total son y qué hacer con ellos.
3. **"Dónde están los retrasos"** compara dos barras: la parte de los pedidos que es riesgo alto y la parte de los retrasos que cayó en riesgo alto. Si la segunda es más grande, el nivel alto está concentrando retrasos. Al lado aparece cuántas veces más retrasos se encuentran que revisando al azar.
4. **"Pedidos a revisar"** es la lista, ordenada del más riesgoso al menos riesgoso.
5. **El inspector**, a la derecha de la lista, muestra el detalle del pedido elegido.

### Filtrar y buscar

- **Pedidos aprobados entre:** elige el rango de fechas.
- **Buscar un pedido:** pega el `order_id` y presiona Enter o el botón **Buscar**. **✕ Limpiar** borra la búsqueda y deja el período y el nivel como estaban.
- **Nivel de riesgo:** los botones sobre la lista filtran por Todos, Alto, Medio o Bajo. Arranca en Alto.
- **Mostrar si se retrasó:** agrega si el pedido de verdad se retrasó. Es un dato histórico para revisar el modelo: en un pedido nuevo todavía no se sabe.

Si una búsqueda no encuentra nada, la app dice si el pedido existe pero queda fuera de los filtros (con un botón para verlo sin filtros) o si el código no existe.

### Inspeccionar un pedido

Marca la casilla al inicio de una fila. Sin selección, el inspector muestra el pedido más riesgoso de la lista. Ahí aparecen:

- El código del pedido, su nivel y su puntaje de 0 a 100.
- El plazo prometido, la distancia y el estado del cliente.
- **Por qué tiene este nivel:** las variables que más suben su riesgo frente al pedido promedio.
- **Lo que juega a favor:** las que lo bajan.
- **Acción sugerida** según el nivel.
- **📋 Copiar ID del pedido**, para buscarlo en otro sistema o en la API.

### Descargar la lista

**Descargar lista (CSV)** baja la lista tal como está filtrada, en el archivo `pedidos_a_revisar.csv`. Sirve para repartir la revisión del día.

El botón 🌙 / ☀️ de la esquina cambia entre modo oscuro y claro. "Cómo leer esta pantalla", al final, resume estas reglas.

## 3. La API (otros sistemas)

Para levantarla:

```bash
uvicorn api.main:app --reload
```

O en Docker (no hace falta instalar Python):

```bash
docker build -t datacraft-api .
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" -v "$(pwd)/data/raw:/app/data/raw" datacraft-api
```

En Windows PowerShell, cambiar `$(pwd)` por `${PWD}`. El volumen de `data/raw` solo hace falta para predecir por `order_id`.

La forma más fácil de probarla es la documentación interactiva: **http://127.0.0.1:8000/docs**. Cada endpoint tiene un botón **Try it out**, se pega el JSON y se presiona **Execute**.

| Endpoint | Para qué |
|---|---|
| `GET /health` | Confirma que el modelo está cargado, con qué se entrenó y sus cortes de nivel |
| `POST /predict` | Riesgo de hasta 1.000 pedidos, enviando sus variables ya calculadas |
| `POST /predict/pedido` | Riesgo a partir del `order_id`: la API arma las variables desde los CSV |

### Predecir por `order_id`

Es lo más simple para probar:

```json
{"order_ids": ["ae213a9f84c777fb7a31e8c0f09fd30c", "2b5d5c1847d069c5efeb82f4056c15d8"]}
```

La respuesta trae, por pedido, `prob_retraso` y `nivel_riesgo`. Los códigos que no existen vuelven en `no_encontrados`. **La primera llamada tarda unos 30 segundos** porque arma todos los pedidos; las siguientes son inmediatas.

Solo es una prueba honesta con pedidos del período de test (aprobados desde el 27/05/2018). Los anteriores los vio el modelo al entrenar.

### Errores frecuentes

| Respuesta | Qué pasó |
|---|---|
| `422` | El JSON trae un campo que no existe (por ejemplo, la fecha de entrega), un valor imposible, un estado de Brasil que no existe o un tipo de pago desconocido. La respuesta dice cuál campo |
| `503` | No hay modelo cargado (falta correr `python -m src.umbral` o montar `models/`), o no están los CSV para `/predict/pedido` |
| Puerto 8000 ocupado | Ya hay otra API corriendo: `docker ps` para verla y `docker stop <id>` para cerrarla |

El detalle de cada campo, un ejemplo completo con `curl` y los 3 pedidos reales de la demo están en [`docs/api.md`](api.md).

## 4. Vendedores recomendados

En la app, la pestaña **Vendedores recomendados** responde a qué vendedor conviene comprarle: se elige la categoría del producto y el estado del cliente, y se ven en tarjetas los 3 vendedores que ya vendieron ahí con menor puntaje de riesgo, con su precio histórico de referencia (no es el precio actual). Categoría y destino solo filtran candidatos: el orden sale del historial completo de cada vendedor. Es una ayuda para revisar opciones; no reemplaza confirmar producto, stock, precio y plazo. La primera vez que se abre tarda cerca de un minuto.

- **Se retrasan menos frente a lo que prometen, no entregan más rápido:** en parte recomienda a quien promete más días.
- Debajo, "Qué encontramos al evaluar las recomendaciones" compara con elegir por historial, por popularidad o al azar. Supera al azar, pero frente al historial y a la popularidad la diferencia no es significativa. Precision, recall e intervalos están en "Evaluación técnica — para el jurado".

Cómo recomienda y sus límites: [`docs/recomendador.md`](recomendador.md).

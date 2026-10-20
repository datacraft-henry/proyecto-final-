# Unificación de datasets Olist (modelo estrella + tabla plana para EDA)

Script: `unificar_olist.py`

Toma los 9 CSV originales de Olist y genera tablas limpias, listas para **Power BI** (modelo estrella) y para el **EDA** (tabla plana a nivel ítem).

> **Importante:** este script es para dashboard y análisis descriptivo. **No es la tabla de entrenamiento del modelo.** Sus tablas incluyen datos posteriores a la aprobación de la compra (fechas de entrega, reseñas), que `docs/decisiones.md` (punto 4) prohíbe como entrada del clasificador. Para modelar se usa la tabla analítica de `src/features.py`.

## Requisitos previos

1. **Los 9 CSV de Olist dentro de `data/raw/`.** Los datos **no vienen con el repo**: son archivos grandes y su uso está sujeto a la licencia de Kaggle, por eso Git los ignora (`docs/decisiones.md`, punto 7). Cada persona los descarga por su cuenta desde [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) y los descomprime en `data/raw/`, con los nombres originales. Mismo paso que el punto 3 del README principal.
2. **Dependencias instaladas:** `python -m pip install -r requirements.txt`.
3. **Ejecutar siempre desde la raíz del repo**, porque las rutas del comando de abajo son relativas a ella.

Si falta el paso 1, el script falla con `FileNotFoundError: ... data\raw\olist_orders_dataset.csv`. Significa que los CSV no están en `data/raw/`.

## Uso

Desde la raíz del repo:

```bash
python scripts/unificacion/unificar_olist.py data/raw data/processed/olist_limpio
```

| Argumento | Qué es | Por defecto |
|---|---|---|
| 1.º | Carpeta con los 9 CSV originales | `.` (carpeta actual) |
| 2.º | Carpeta de salida (se crea si no existe) | `olist_limpio` |

La salida conviene escribirla en `data/processed/`, que Git ignora. Los CSV generados **no se suben al repo** (`docs/decisiones.md`, punto 7).

Requiere `pandas` y `numpy` (ya están en `requirements.txt`).

## Qué entra

Los 9 archivos de Olist:

`olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_order_payments_dataset.csv`, `olist_order_reviews_dataset.csv`, `olist_customers_dataset.csv`, `olist_products_dataset.csv`, `olist_sellers_dataset.csv`, `olist_geolocation_dataset.csv`, `product_category_name_translation.csv`.

## Qué sale (9 CSV)

| Archivo | Tipo | Grano (una fila por...) | Contenido |
|---|---|---|---|
| `fact_orders` | Hechos | Pedido | Pedidos con cliente, ítems, pagos, reseña y duraciones calculadas |
| `fact_order_items` | Hechos | Ítem de pedido | Ítems con datos del pedido y `monto_item` (precio + flete) |
| `fact_order_payments` | Hechos | Pago | Tabla de pagos tal como viene, sin agregar |
| `dim_customers` | Dimensión | `customer_id` | Clientes |
| `dim_sellers` | Dimensión | Vendedor | Vendedores |
| `dim_products` | Dimensión | Producto | Productos con categoría traducida y volumen |
| `dim_geolocation` | Dimensión | Prefijo de código postal | Latitud, longitud, ciudad y estado |
| `dim_calendario` | Dimensión | Día | Año, mes, trimestre, día de la semana |
| `olist_items_flat_EDA` | Tabla plana | Ítem de pedido | Ítems unidos a producto, vendedor y datos del pedido, para EDA |

Relaciones para Power BI: `fact_orders.order_id` ↔ `fact_order_items.order_id` y `fact_order_payments.order_id`; `customer_id`, `product_id` y `seller_id` hacia sus dimensiones; `zip_prefix` hacia `dim_geolocation`; `fecha_compra` hacia `dim_calendario.fecha`.

## Qué filtra y qué transforma

**Lo que se descarta:**
- En la geolocalización, las filas con coordenadas fuera de Brasil (latitud entre -34 y 6, longitud entre -74 y -34).

**Lo que se agrega para que no se dupliquen filas:**
- **Geolocalización:** de muchas filas por código postal a una sola, usando la **mediana** de latitud y longitud y la ciudad y el estado más frecuentes.
- **Pagos:** a nivel pedido se suman (`pago_total`), se cuentan los métodos (`n_metodos_pago`), se toma el máximo de cuotas (`cuotas_max`) y se define `tipo_pago_principal` como el método de **mayor monto** dentro del pedido.
- **Reseñas:** un pedido puede tener varias. Se conserva la **última respondida** y se guarda cuántas había (`n_reviews`).
- **Ítems:** a nivel pedido se calculan `n_items`, `n_vendedores`, `total_price` y `total_freight`.

**Limpieza de productos:**
- Se corrigen los nombres de columna con error de ortografía del original (`lenght` → `length`).
- Se agregan a mano las traducciones de dos categorías que la tabla oficial no incluye (`pc_gamer` y `portateis_cozinha_e_preparadores_de_alimentos`).
- Categorías sin dato quedan como `desconocida` (original) y `unknown` (inglés).
- Se calcula `product_volume_cm3` (largo × alto × ancho).

**Lo que NO se filtra:**
- `fact_orders` conserva **todos los pedidos**, entregados o no. Para analizar entregas hay que filtrar por `pedido_retrasado` no nula.

## Columnas calculadas en `fact_orders`

| Columna | Definición |
|---|---|
| `dias_aprobacion` | Compra → aprobación, en días (con decimales) |
| `dias_a_transportista` | Compra → entrega al transportista |
| `dias_entrega_total` | Compra → entrega al cliente |
| `dias_entrega_estimada` | Compra → fecha estimada |
| `dias_retraso` | Entrega real − fecha estimada, en días con decimales. Negativo si llegó antes. Es continua, útil para ver la distribución |
| `pedido_retrasado` | **Target.** Ver abajo |
| `fecha_compra` | Fecha de compra sin hora (clave hacia `dim_calendario`) |
| `tiene_items`, `tiene_review` | Indicadores booleanos |

### Definición de `pedido_retrasado`

No se calcula en el script: se toma de `build_target` (`src/target.py`) a través de `agregar_target` (`src/unificacion.py`), así hay una sola definición en todo el proyecto (`docs/decisiones.md`, puntos 2 y 8).

- `1.0` si el **día** de entrega real es posterior al **día** estimado.
- `0.0` si llegó el día estimado o antes.
- **Vacío (NaN)** si el pedido no está `delivered`, le falta alguna fecha o se compró fuera del período de estudio (enero 2017 a agosto 2018). Estos pedidos no se etiquetan como puntuales.

**Verificación esperada:** `fact_orders["pedido_retrasado"].mean()` debería dar **0,0679** (los NaN se excluyen del promedio), con 96.203 pedidos no nulos.

## Validaciones automáticas

El script se detiene con error si falla alguna de estas comprobaciones:

- La tabla plana tiene exactamente las mismas filas que `order_items` (ningún merge duplicó filas).
- `fact_orders` tiene un solo registro por `order_id`.
- Los identificadores de producto, cliente y vendedor son únicos en sus dimensiones.
- La suma de precios coincide con el original, en `fact_order_items` y en `fact_orders`.
- La suma de pagos coincide con el original.

Al terminar imprime el tamaño de cada tabla y `OK: todas las validaciones pasaron`.

## Limitaciones conocidas

- `dim_geolocation` solo tiene los códigos postales con al menos una coordenada válida. Algunos clientes o vendedores pueden quedar sin ubicación al cruzar.
- `fact_orders` incluye datos posteriores a la aprobación (ver la advertencia del inicio). No usarla como entrada del modelo.
- La definición del target está **repetida** acá y en `src/target.py`. Si el equipo cambia la regla, hay que actualizar ambos lugares.
- El script no tiene tests automáticos.
- Los CSV de salida son una "foto" de los datos originales. Si cambian los de `data/raw/`, hay que volver a correr el script.

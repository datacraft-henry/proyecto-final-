# Base de datos en BigQuery (E1)

Los 9 CSV de Olist cargados en un dataset `olist` de BigQuery, con el esquema
documentado y una vista `pedidos_analitica` que replica en SQL el target y las
variables de `src/` para que cualquiera del equipo consulte sin correr Python.

| Archivo | Para qué |
|---|---|
| `01_esquema.sql` | Tablas con tipos (sin constraints: el sandbox no las acepta) |
| `02_tabla_analitica.sql` | Vista `pedidos_analitica`: una fila por pedido entregado, `pedido_retrasado` + variables del baseline |
| `03_validaciones.sql` | Conteos, llaves únicas, huérfanos y tasa de retraso esperada (6,77 %) |
| `cargar_bigquery.py` | Script idempotente que crea el dataset, carga los CSV y corre las validaciones |

## Cómo cargarla (una vez por persona o una vez para el equipo)

1. Proyecto en Google Cloud con BigQuery habilitado (el sandbox gratis alcanza: el dataset pesa ~130 MB).
2. Dependencias extra (no están en `requirements.txt` hasta que el equipo lo acuerde):
   ```bash
   pip install google-cloud-bigquery db-dtypes
   ```
3. Autenticación local:
   ```bash
   gcloud auth application-default login
   ```
4. CSV de Kaggle descomprimidos en `data/raw/`.
5. Cargar y validar:
   ```bash
   python -m sql.cargar_bigquery --proyecto MI-PROYECTO --validar
   ```
   Repetirlo no duplica filas (`WRITE_TRUNCATE`).

## Cómo consultarla desde Python

```python
from google.cloud import bigquery
df = bigquery.Client(project="MI-PROYECTO").query(
    "SELECT * FROM `MI-PROYECTO.olist.pedidos_analitica`"
).to_dataframe()
```

## Criterio de aceptación (tarjeta E1 en Trello)

- [x] `python -m sql.cargar_bigquery --proyecto X --validar` termina en `VALIDACIÓN: OK`.
- [x] `pedidos_analitica` tiene la misma cantidad de filas y la misma tasa de `pedido_retrasado` que `build_features(...)` de `src/` con anchor `aprobacion`.
- [x] Los compañeros con acceso al proyecto pueden consultar la vista desde la consola de BigQuery.

### Evidencia de la validación (26/10/2026)

Base cargada el 06/10/2026 en el proyecto `olist-datacraft`, dataset `olist`. Las consultas de `03_validaciones.sql` se volvieron a correr el 26/10/2026 sobre esa base:

| Prueba | Esperado | Obtenido | Resultado |
|---|---|---|---|
| Filas por tabla (9 tablas) | Cifras de Kaggle, versión 2 | Las 9 coinciden (orders 99.441, order_items 112.650, customers 99.441, sellers 3.095, products 32.951, order_payments 103.886, order_reviews 99.224, geolocation 1.000.163, traducción 71) | OK |
| Llaves primarias únicas (5) | Sin duplicados | Sin duplicados | OK |
| Huérfanos de llaves foráneas (5) | 0 | 0 en las 5 | OK |
| Entregados sin filtro de período | 96.470 · 6,77 % tarde | 96.470 · 6,77 % | OK |
| Entregados en el período de estudio | Igual a `build_target()` | 96.203 · 6,79 % | OK |
| Vista `pedidos_analitica` | Igual a `build_features(..., anchor="aprobacion")` con target: 96.189 · 6,79 % | 96.189 · 6,79 % | OK |

El equipo tiene acceso de lectura al proyecto con sus cuentas de Google. En el sandbox las tablas vencen a los 60 días: esta carga vence el 05/12/2026, y para seguir usándola hay que recargar con `cargar_bigquery.py` o activar facturación.

## Decisiones

- **BigQuery y no Cloud SQL:** gratis, sin servidor que administrar, y el equipo consulta desde cualquier lado. El costo es que no aplica llaves foráneas: por eso `03_validaciones.sql` revisa huérfanos y unicidad después de cada carga.
- **Período de estudio en la vista:** compras de 2017-01-01 a 2018-09-01 (decisión 8), igual que `build_target()`. Las tablas base conservan todos los pedidos; el filtro está solo en la vista.
- **Códigos postales como STRING** (igual que `src/data_loading.py`) para no perder ceros a la izquierda.
- **`review_id` no es PK:** en el CSV de Kaggle se repite en varios pedidos.
- **La vista no reemplaza a `src/features.py`:** el modelo sigue entrenándose desde el código Python, que es la fuente de verdad. La vista es para explorar y para verificar que ambas dan lo mismo.

## Relaciones entre tablas

| Tabla | Llave | Referencia |
|---|---|---|
| customers | customer_id | |
| sellers | seller_id | |
| products | product_id | |
| orders | order_id | customer_id → customers |
| order_items | (order_id, order_item_id) | order_id → orders, product_id → products, seller_id → sellers |
| order_payments | (order_id, payment_sequential) | order_id → orders |
| order_reviews | review_id (no única en Kaggle) | order_id → orders |

## Caducidad en el sandbox

En el sandbox gratuito de BigQuery las tablas caducan a los 60 días y las particiones por fecha se borran de inmediato (por eso `orders` no va particionada). Recargar con `cargar_bigquery.py` antes de que venza, o activar facturación para quitar la caducidad.

## Cifras verificadas (carga del 5 oct 2026)

- 9 tablas con los conteos de Kaggle; llaves únicas y sin huérfanos.
- Entregados sin filtro de período: 96.470, 6,77 % tarde (coincide con `docs/decisiones.md`).
- `pedidos_analitica` (período 2017-01 a 2018-08, con fecha de aprobación): 96.189 filas, 6,79 % tarde.

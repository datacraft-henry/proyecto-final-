# Olist Delivery Risk

Proyecto de ciencia de datos orientado a estimar el riesgo de retraso en las entregas de pedidos de Olist, utilizando información comercial y logística disponible al momento de aprobar la compra.

**Estado:** Sprint 2 (cierre y demo el 28 de octubre de 2026). Listos: base de datos, EDA, target único, modelo principal calibrado con niveles de riesgo, monitoreo del nivel alto, app de Streamlit, API, Docker y pedidos de ejemplo para la demo. En curso: tablero de Power BI, informe final y presentación.

## Índice

1. [Problema de negocio](#problema-de-negocio)
2. [Objetivo](#objetivo)
3. [Alcance](#alcance)
4. [Datos](#datos)
5. [Definición de la predicción](#definición-de-la-predicción)
6. [Variables evaluadas](#variables-evaluadas)
7. [Metodología](#metodología)
8. [Evaluación](#evaluación)
9. [Resultados](#resultados)
10. [Dónde está cada entregable](#dónde-está-cada-entregable)
11. [Estructura del repositorio](#estructura-del-repositorio)
12. [Herramientas](#herramientas)
13. [Cómo instalar y correr el proyecto](#cómo-instalar-y-correr-el-proyecto)
14. [Cómo trabajamos con Git](#cómo-trabajamos-con-git)
15. [Limitaciones](#limitaciones)
16. [Próximos pasos](#próximos-pasos)
17. [Autoría](#autoría)

## Problema de negocio

Los retrasos en las entregas pueden afectar la experiencia del cliente. Identificar anticipadamente los pedidos con mayor riesgo de demora permitiría orientar el seguimiento logístico y la comunicación preventiva.

Este proyecto responde la siguiente pregunta:

> ¿Podemos predecir si un pedido llegará después del día prometido utilizando únicamente información disponible al aprobar la compra?

## Objetivo

Desarrollar y evaluar un modelo de clasificación que estime el riesgo de entrega tardía de un pedido.

La predicción busca apoyar la priorización del seguimiento logístico. Su impacto sobre los retrasos y la satisfacción del cliente requeriría una validación posterior en una operación real.

## Alcance

El proyecto cubre:

- Explorar la calidad de los datos y las relaciones entre las tablas.
- Construir una tabla analítica con una fila por pedido.
- Analizar factores asociados con entregas tardías.
- Entrenar y comparar modelos de clasificación.
- Evaluar la detección de retrasos y la generación de falsas alertas.
- Desarrollar una app en Streamlit y una API de predicción.

La recomendación de acciones para aumentar ventas y la predicción de cancelaciones quedan fuera del alcance.

## Datos

Usamos el conjunto público [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), con pedidos históricos de 2016 a 2018.

| Archivo | Contenido |
|---|---|
| `olist_orders_dataset.csv` | Pedidos, estados y fechas |
| `olist_order_items_dataset.csv` | Artículos, vendedores, precios y fletes |
| `olist_customers_dataset.csv` | Identificadores y ubicación de clientes |
| `olist_order_payments_dataset.csv` | Información de pagos |
| `olist_order_reviews_dataset.csv` | Valoraciones y comentarios |
| `olist_products_dataset.csv` | Categorías y características de productos |
| `olist_sellers_dataset.csv` | Información de vendedores |
| `olist_geolocation_dataset.csv` | Coordenadas asociadas a códigos postales |
| `product_category_name_translation.csv` | Traducción de categorías al inglés |

Los datos originales se descargan de Kaggle y se guardan localmente en `data/raw/`. No se incluyen en este repositorio. Su uso está sujeto a las condiciones y la licencia indicadas en la fuente.

## Definición de la predicción

**Unidad de análisis:** un pedido.

**Momento de predicción:** aprobación de la compra.

**Variable objetivo:** `pedido_retrasado`.

| Valor | Definición |
|---|---|
| `1` | La entrega ocurrió después del día estimado |
| `0` | La entrega ocurrió el día estimado o antes |

La comparación se hace por día calendario, sin considerar la hora (decisión 2).

Usamos los pedidos entregados con fechas válidas de entrega real y estimada, comprados entre enero de 2017 y agosto de 2018 (decisión 8): 96.203 pedidos, 6,79 % con retraso. Los pedidos sin información suficiente no se etiquetan como puntuales.

Esta selección limita las conclusiones a pedidos con entrega observada.

## Variables evaluadas

Evaluamos características como:

- Fecha de compra y de aprobación.
- Ubicación del cliente y de los vendedores.
- Categorías y cantidad de artículos.
- Precio y valor del flete.
- Peso y dimensiones de los productos.
- Tipo de pago y número de cuotas.
- Tiempo prometido para la entrega.
- Distancia aproximada entre origen y destino.

Cada variable se revisó para confirmar que se conoce al aprobar el pago. Los pedidos con varios artículos o vendedores se agregan con reglas explícitas (`src/features.py`). El modelo final usa las variables del baseline sin el mes de compra, más la distancia cliente-vendedor y los feriados nacionales de los 7 días siguientes (`docs/decisiones_modelos.md`).

La fecha real de entrega se usa para construir la variable objetivo, pero no como entrada del modelo. Tampoco se usan reseñas ni otros datos conocidos después del momento de predicción (decisión 4); hay tests que fallan si alguno entra.

## Metodología

1. **Exploración y calidad:** revisamos tipos de datos, faltantes, duplicados y coherencia de fechas.
2. **Integración:** unimos las tablas controlando sus relaciones para evitar multiplicar pedidos o importes.
3. **Preparación:** construimos la variable objetivo y las características predictoras.
4. **División temporal:** separamos períodos de entrenamiento, validación y prueba.
5. **Modelado:** comparamos una regresión logística de referencia con una logística con selección de variables y modelos de árboles, con validación cruzada temporal de 4 tramos.
6. **Evaluación:** analizamos métricas, umbrales de alerta y tipos de error.
7. **Comunicación:** documentamos resultados, limitaciones y funcionamiento de la demo.

Las transformaciones que aprenden de los datos se ajustan solo con el conjunto de entrenamiento. El conjunto de prueba se usa una sola vez, para la evaluación final.

## Evaluación

Usamos:

- **Recall:** proporción de retrasos reales que el modelo detecta.
- **Precision:** proporción de alertas que corresponden a retrasos reales.
- **F1-score:** medida que combina precision y recall.
- **PR-AUC:** resumen del comportamiento de precision y recall al variar el umbral.
- **ROC-AUC:** capacidad de discriminación entre clases.
- **Matriz de confusión:** detalle de aciertos y errores.

El modelo se elige por PR-AUC, porque con 6,8 % de retrasos la ROC-AUC sobrestima el desempeño. Las probabilidades se recalibran con Platt en validación, y los niveles de riesgo se definen por volumen de pedidos, no por un umbral fijo (decisión 9). Detalle en `docs/metricas_evaluacion.md`.

## Resultados

Modelo final: regresión logística, reentrenada con train + validación y calibrada. Resultados en test (fines de mayo a agosto de 2018, 19.293 pedidos, 3,5 % de retraso):

- **ROC-AUC 0,707** y **PR-AUC 0,073**, el doble del piso de 0,035 (la tasa de retraso).
- Los árboles empataron en validación cruzada pero cayeron a ROC-AUC 0,58 en test, porque Olist acortó los plazos prometidos.

| Nivel | % de pedidos | Tasa de retraso | % de los retrasos que capta | Lift |
|---|---|---|---|---|
| Alto | 11,2 % | 7,5 % | 24,1 % | 2,15 |
| Medio | 22,0 % | 5,6 % | 35,2 % | 1,60 |
| Bajo | 66,8 % | 2,1 % | 40,7 % | 0,61 |

Revisando 1 de cada 9 pedidos se llega a 1 de cada 4 retrasos. Es un modelo para **priorizar** el seguimiento, no para predecir con certeza qué pedido se va a retrasar: lo confiable es el orden y el nivel, no la probabilidad exacta. Fuente: `docs/metricas_evaluacion.md` y `docs/decisiones_modelos.md`.

## Dónde está cada entregable

| Entregable | Dónde |
|---|---|
| Decisiones del equipo | [`docs/decisiones.md`](docs/decisiones.md) |
| Exploración del target y control de fuga | [`notebooks/01_exploracion_y_target.ipynb`](notebooks/01_exploracion_y_target.ipynb) · [`docs/aporte_mathias_exploracion.md`](docs/aporte_mathias_exploracion.md) |
| EDA con gráficos | [`notebooks/02_eda_graficos.ipynb`](notebooks/02_eda_graficos.ipynb) · [`reports/figuras/`](reports/figuras/) · [`docs/aporte_alexis_eda_graficos.md`](docs/aporte_alexis_eda_graficos.md) |
| Base de datos en BigQuery | [`sql/`](sql/) |
| Target único y período de estudio | [`src/target.py`](src/target.py) |
| Baseline | [`src/baseline.py`](src/baseline.py) · [`docs/decisiones_baseline.md`](docs/decisiones_baseline.md) |
| Features, comparación de modelos y modelo principal | [`src/feature_engineering.py`](src/feature_engineering.py) · [`src/training.py`](src/training.py) · [`docs/decisiones_modelos.md`](docs/decisiones_modelos.md) |
| Features del sprint 2 (carga del vendedor y feriados) | [`src/feature_ablation.py`](src/feature_ablation.py) · [`docs/aporte_mathias_features_sprint2.md`](docs/aporte_mathias_features_sprint2.md) |
| Calibración y niveles de riesgo | [`src/calibration.py`](src/calibration.py) · [`src/umbral.py`](src/umbral.py) · [`src/monitoreo.py`](src/monitoreo.py) |
| Métricas de evaluación y justificación del modelo | [`docs/metricas_evaluacion.md`](docs/metricas_evaluacion.md) |
| App de riesgo por pedido | [`app/streamlit_app.py`](app/streamlit_app.py) (pantalla) · [`app/riesgo.py`](app/riesgo.py) (lógica) · [`app/vista.py`](app/vista.py) (filtros y formato) · [`app/estilos.py`](app/estilos.py) (CSS) |
| API de predicción y Docker | [`api/main.py`](api/main.py) · [`Dockerfile`](Dockerfile) · [`docs/api.md`](docs/api.md) |
| Pedidos reales para la demo de la API | [`scripts/demo_pedidos.py`](scripts/demo_pedidos.py) · [`docs/api.md`](docs/api.md#pedidos-reales-para-la-demo) |
| Unificación y esquema estrella para Power BI | [`scripts/unificacion/`](scripts/unificacion/) · [`src/unificacion.py`](src/unificacion.py) |
| Tests (anti-fuga, target, calibración, app y API) | [`tests/`](tests/) |

## Estructura del repositorio

| Ruta | Propósito |
|---|---|
| `data/raw/` | Datos originales locales, excluidos de Git |
| `data/processed/` | Datos preparados locales, excluidos de Git |
| `notebooks/` | Exploración, análisis y experimentos (sin lógica reutilizable) |
| `src/` | Código reutilizable: carga, target, variables, modelos, calibración |
| `tests/` | Tests automáticos (`pytest`) |
| `sql/` | Esquema y carga de la base en BigQuery |
| `scripts/` | Scripts reproducibles que no son lógica del modelo (unificación para Power BI y pedidos de la demo) |
| `app/` | App de riesgo por pedido en Streamlit |
| `api/` | API de predicción (FastAPI) |
| `docs/` | Decisiones del equipo, métricas y aportes |
| `reports/` | Gráficos y resultados |
| `models/` | Modelos entrenados, locales y excluidos de Git |
| `Dockerfile` | Imagen de la API |
| `requirements.txt` | Dependencias del proyecto |

Las carpetas vacías se mantienen con un archivo `.gitkeep`.

## Herramientas

- Python, pandas y NumPy.
- Scikit-learn y joblib.
- Matplotlib y Seaborn.
- Streamlit (app) y FastAPI con Uvicorn (API).
- Docker.
- Google BigQuery y Power BI.
- pytest.
- Jupyter y Visual Studio Code.
- Git y GitHub.

## Cómo instalar y correr el proyecto

Requisitos: Python 3.10 o superior y Git.

```bash
# 1. Clonar el repo y pararse en developer
git clone https://github.com/datacraft-henry/proyecto-final-.git
cd proyecto-final-
git checkout developer

# 2. Crear un entorno virtual e instalar dependencias
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements.txt

# 3. Descargar los datos
#    Bajar el zip de https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
#    y descomprimir los 9 CSV dentro de data/raw/ (Git los ignora).

# 4. Verificar que todo funciona
python -m pytest

# 5. Entrenar y comparar modelos (escribe reports/comparacion_modelos.json)
python -m src.training

# 6. Calibrar y guardar el modelo final (models/modelo_principal_calibrado.joblib)
python -m src.umbral

# 6b. Monitorear por mes el volumen y la tasa del nivel alto (reports/monitoreo_nivel_alto.json)
python -m src.monitoreo

# 7. Abrir la app de riesgo por pedido
streamlit run app/streamlit_app.py

# 8. Levantar la API (documentación en http://127.0.0.1:8000/docs)
uvicorn api.main:app --reload

# 9. O correr la API en Docker (ver docs/api.md)
docker build -t datacraft-api .
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" datacraft-api

# 10. Sacar 3 pedidos reales del test para probar la API (reports/demo_pedidos.json)
python -m scripts.demo_pedidos
```

Los notebooks se abren con Jupyter desde la raíz del repo (`jupyter notebook`) o desde Visual Studio Code.

## Cómo trabajamos con Git

- `main`: solo entregas.
- `developer`: integración. Todo entra por pull request aprobado por Agustina.
- `feature/E#-descripcion`: una rama por tarea, creada desde `developer` actualizado.
- Commits chicos con prefijo: `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
- Los datos y los documentos personales nunca se suben (ver `.gitignore`).

El detalle y las decisiones del equipo están en [`docs/decisiones.md`](docs/decisiones.md).

## Limitaciones

- Los datos son históricos y corresponden al contexto brasileño.
- No se dispone de información completa sobre clima, tránsito o capacidad operativa.
- Entrenar con pedidos entregados introduce una selección que debe considerarse al interpretar los resultados.
- Las asociaciones observadas no demuestran relaciones causales.
- El desempeño con estos datos no garantiza resultados equivalentes en una operación actual.
- El proyecto es un prototipo educativo y no está validado para uso en producción.

## Próximos pasos

- [x] Configurar el entorno local y las dependencias.
- [x] Incorporar las reglas de exclusión de datos en `.gitignore`.
- [x] Explorar la tabla de pedidos (`notebooks/01_exploracion_y_target.ipynb`).
- [x] Construir la tabla analítica (`src/features.py`, una fila por pedido).
- [x] Baseline entrenado y evaluado (regresión logística con validación temporal, `src/baseline.py`).
- [x] Desarrollar el análisis exploratorio (`notebooks/02_eda_graficos.ipynb`).
- [x] Entrenar y evaluar modelos (`src/training.py`, ver `docs/decisiones_modelos.md`).
- [x] Calibrar el modelo principal y definir niveles de riesgo (`src/umbral.py`).
- [x] Construir la demo en Streamlit y la API
- [x] Sumar las features del EDA de Mathias (`src/feature_ablation.py`, ver `docs/decisiones_modelos.md`, "Sprint 2").
- [x] Usar el target único en el script de unificación (`src/unificacion.py`).
- [x] Empaquetar la API en Docker (`Dockerfile`, ver `docs/api.md`).
- [ ] Construir el tablero de Power BI.
- [x] Monitorear el volumen del nivel de riesgo alto (`src/monitoreo.py`).
- [ ] Escribir el informe final y preparar la defensa.

## Autoría

Proyecto final de Data Science de Henry, desarrollado por el equipo **DataCraft**:

| Integrante | Rol | Aportes principales |
|---|---|---|
| Agustina Baez | Data Scientist · Scrum Master | Proceso del equipo, revisión y aprobación de los pull requests |
| Cristian Atrio | Data Scientist | Entorno y repositorio, baseline, features y modelo principal |
| Alexis Sircelj | Data Scientist | Base de datos, esquema estrella para Power BI y EDA con gráficos |
| Mathias Chaparro | Data Scientist | Exploración del target, control de fuga y auditoría de datos |
| María Camila Uribe Noreña | Data Scientist | Período de estudio, base en BigQuery, calibración y niveles de riesgo |

Proyecto educativo y de prototipado.

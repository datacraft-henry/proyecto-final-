# Olist Delivery Risk

Proyecto de ciencia de datos orientado a estimar el riesgo de retraso en las entregas de pedidos de Olist, utilizando información comercial y logística disponible al momento de aprobar la compra.

**Estado:** cierre del Sprint 1 (DEMO 1, 14 de octubre de 2026). Base de datos, EDA, target, modelo principal calibrado y niveles de riesgo terminados. La app en Streamlit, la API y el tablero de Power BI son del Sprint 2.

## Índice

1. [Problema de negocio](#problema-de-negocio)
2. [Objetivo](#objetivo)
3. [Alcance](#alcance)
4. [Datos](#datos)
5. [Definición de la predicción](#definición-de-la-predicción)
6. [Variables candidatas](#variables-candidatas)
7. [Metodología](#metodología)
8. [Evaluación](#evaluación)
9. [Dónde está cada entregable](#dónde-está-cada-entregable)
10. [Estructura prevista](#estructura-prevista)
11. [Herramientas previstas](#herramientas-previstas)
12. [Cómo instalar y correr el proyecto](#cómo-instalar-y-correr-el-proyecto)
13. [Cómo trabajamos con Git](#cómo-trabajamos-con-git)
14. [Limitaciones](#limitaciones)
15. [Próximos pasos](#próximos-pasos)
16. [Autoría](#autoría)

## Problema de negocio

Los retrasos en las entregas pueden afectar la experiencia del cliente. Identificar anticipadamente los pedidos con mayor riesgo de demora permitiría orientar el seguimiento logístico y la comunicación preventiva.

Este proyecto explora la siguiente pregunta:

> ¿Podemos predecir si un pedido llegará después del día prometido utilizando únicamente información disponible al aprobar la compra?

## Objetivo

Desarrollar y evaluar un modelo de clasificación que estime el riesgo de entrega tardía de un pedido.

La predicción busca apoyar la priorización del seguimiento logístico. Su impacto sobre los retrasos y la satisfacción del cliente requeriría una validación posterior en una operación real.

## Alcance

La primera versión se enfocará en:

- Explorar la calidad de los datos y las relaciones entre las tablas.
- Construir una tabla analítica con una fila por pedido.
- Analizar factores asociados con entregas tardías.
- Entrenar y comparar modelos de clasificación.
- Evaluar la detección de retrasos y la generación de falsas alertas.
- Desarrollar una demo en Streamlit.

La recomendación de acciones para aumentar ventas y la predicción de cancelaciones quedan fuera del alcance inicial.

## Datos

Se utilizará el conjunto público [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), con pedidos históricos de 2016 a 2018.

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

Los datos originales se descargarán desde la fuente y se almacenarán localmente en `data/raw/`. No se incluirán en este repositorio. Su uso está sujeto a las condiciones y la licencia indicadas en la fuente.

## Definición de la predicción

**Unidad de análisis:** un pedido.

**Momento de predicción:** aprobación de la compra.

**Variable objetivo:** `pedido_retrasado`.

| Valor | Definición |
|---|---|
| `1` | La entrega ocurrió después del día estimado |
| `0` | La entrega ocurrió el día estimado o antes |

La comparación se realizará por día calendario, sin considerar la hora.

Para la primera versión se utilizarán pedidos con estado entregado y fechas válidas de entrega real y estimada. Los pedidos sin información suficiente no se etiquetarán automáticamente como puntuales.

Esta selección limita las conclusiones a pedidos con entrega observada.

## Variables candidatas

Se evaluarán características como:

- Fecha de compra y de aprobación.
- Ubicación del cliente y de los vendedores.
- Categorías y cantidad de artículos.
- Precio y valor del flete.
- Peso y dimensiones de los productos.
- Tipo de pago y número de cuotas.
- Tiempo prometido para la entrega.
- Distancia aproximada entre origen y destino.

La disponibilidad de cada variable en el momento de predicción se revisará antes de incluirla. Los pedidos con múltiples artículos o vendedores requerirán reglas de agregación explícitas.

La fecha real de entrega se utilizará para construir la variable objetivo, pero no como entrada del modelo. Tampoco se utilizarán reseñas posteriores del mismo pedido ni otros datos conocidos después del momento de predicción.

## Metodología

1. **Exploración y calidad:** revisar tipos de datos, faltantes, duplicados y coherencia de fechas.
2. **Integración:** unir las tablas controlando sus relaciones para evitar multiplicar pedidos o importes.
3. **Preparación:** construir la variable objetivo y las características predictoras.
4. **División temporal:** separar periodos de entrenamiento, validación y prueba.
5. **Modelado:** comparar una referencia sencilla con regresión logística y modelos de árboles.
6. **Evaluación:** analizar métricas, umbrales de alerta y tipos de error.
7. **Comunicación:** documentar resultados, limitaciones y funcionamiento de la demo.

Las transformaciones que aprendan de los datos se ajustarán únicamente con el conjunto de entrenamiento. El conjunto de prueba se reservará para la evaluación final.

## Evaluación

Se utilizarán:

- **Recall:** proporción de retrasos reales que el modelo detecta.
- **Precision:** proporción de alertas que corresponden a retrasos reales.
- **F1-score:** medida que combina precision y recall.
- **PR-AUC:** resumen del comportamiento de precision y recall al variar el umbral.
- **ROC-AUC:** capacidad de discriminación entre clases.
- **Matriz de confusión:** detalle de aciertos y errores.

El umbral de alerta se seleccionará con los datos de validación, considerando el equilibrio entre detectar retrasos y generar falsas alertas.

También se revisará la calibración antes de interpretar los valores estimados como probabilidades de riesgo.

## Dónde está cada entregable

| Entregable | Dónde |
|---|---|
| Decisiones del equipo (11) | [`docs/decisiones.md`](docs/decisiones.md) |
| Exploración del target y control de fuga | [`notebooks/01_exploracion_y_target.ipynb`](notebooks/01_exploracion_y_target.ipynb) · [`docs/aporte_mathias_exploracion.md`](docs/aporte_mathias_exploracion.md) |
| EDA con gráficos | [`notebooks/02_eda_graficos.ipynb`](notebooks/02_eda_graficos.ipynb) · [`reports/figuras/`](reports/figuras/) · [`docs/aporte_alexis_eda_graficos.md`](docs/aporte_alexis_eda_graficos.md) |
| Base de datos en BigQuery | [`sql/`](sql/) |
| Target y período de estudio | [`src/target.py`](src/target.py) |
| Baseline | [`src/baseline.py`](src/baseline.py) · [`docs/decisiones_baseline.md`](docs/decisiones_baseline.md) |
| Features, comparación de modelos y modelo principal | [`src/feature_engineering.py`](src/feature_engineering.py) · [`src/training.py`](src/training.py) · [`docs/decisiones_modelos.md`](docs/decisiones_modelos.md) |
| Calibración y niveles de riesgo | [`src/calibration.py`](src/calibration.py) · [`src/umbral.py`](src/umbral.py) |

| App de riesgo por pedido | [`app/streamlit_app.py`](app/streamlit_app.py) (pantalla) · [`app/riesgo.py`](app/riesgo.py) (lógica) · [`app/vista.py`](app/vista.py) (filtros y formato) · [`app/estilos.py`](app/estilos.py) (CSS) |

| Métricas de evaluación, plan de validación y justificación del modelo | [`docs/metricas_evaluacion.md`](docs/metricas_evaluacion.md) |

| Esquema estrella para Power BI | [`scripts/unificacion/`](scripts/unificacion/) |
| Tests (anti-fuga, target, calibración) | [`tests/`](tests/) |

## Estructura prevista

| Ruta | Propósito |
|---|---|
| `data/raw/` | Datos originales locales, excluidos de Git |
| `data/processed/` | Datos preparados locales, excluidos de Git |
| `notebooks/` | Exploración, análisis y experimentos (sin lógica reutilizable) |
| `src/` | Código reutilizable: carga, target, variables, modelos |
| `tests/` | Tests automáticos (`pytest`) |
| `sql/` | Consultas y modelo de datos |
| `app/` | Demo en Streamlit |
| `docs/` | Decisiones del equipo (`docs/decisiones.md`) |
| `reports/` | Gráficos y resultados |
| `requirements.txt` | Dependencias del proyecto |
| `.gitignore` | Reglas de exclusión de archivos |
| `README.md` | Documentación principal |

Las carpetas vacías se mantienen con un archivo `.gitkeep`.

## Herramientas previstas

- Python.
- Pandas y NumPy.
- Matplotlib y Seaborn.
- Scikit-learn.
- Jupyter en Visual Studio Code.
- Streamlit.
- Git y GitHub.

## Cómo instalar y correr el proyecto

Requisitos: Python 3.10 o superior y Git.

```bash
# 1. Clonar el repo y pararse en developer
git clone https://github.com/CamilaUribeNorena/proyecto-final-.git
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

# 7. Abrir la app de riesgo por pedido
streamlit run app/streamlit_app.py
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
- [ ] Sumar las features del EDA de Mathias (ruta, carga del vendedor, feriados).
- [ ] Usar el target único en el script de unificación.
- [ ] Construir el tablero de Power BI.
- [ ] Monitorear el volumen del nivel de riesgo alto.
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

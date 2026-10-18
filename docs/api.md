# API de predicción

Devuelve, para uno o varios pedidos, la **probabilidad calibrada de retraso** y el **nivel de riesgo** (alto, medio, bajo). Usa el mismo modelo y la misma función que la app de Streamlit (`models/modelo_principal_calibrado.joblib` y `src.umbral.predict_risk`), así que las dos dan siempre el mismo resultado.

## Cómo correrla

Desde la raíz del repo, con los CSV en `data/raw/`:

```bash
pip install -r requirements.txt
python -m src.umbral                 # genera el artefacto calibrado (una sola vez)
uvicorn api.main:app --reload        # http://127.0.0.1:8000/docs
```

Para usar otro artefacto: `MODELO_PATH=ruta/al/modelo.joblib uvicorn api.main:app`.

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| `GET` | `/health` | Si hay modelo cargado, con qué se entrenó, sus variables y los cortes de riesgo |
| `POST` | `/predict` | Probabilidad y nivel de riesgo de hasta 1.000 pedidos por llamada |
| `GET` | `/docs` | Documentación interactiva (Swagger) para probar la API desde el navegador |

## Ejemplo

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "pedidos": [{
    "order_id": "demo-1",
    "purchase_dayofweek": 2, "purchase_hour": 14,
    "promised_days": 9, "approval_hours": 0.3,
    "n_items": 1, "n_sellers": 1,
    "total_price": 120.0, "total_freight": 18.0, "freight_ratio": 0.15,
    "total_weight_g": 800, "total_volume_cm3": 4000, "max_installments": 3,
    "max_distance_km": 1850,
    "customer_state": "MA", "main_payment_type": "credit_card", "seller_other_state": "True"
  }]
}'
```

Respuesta (números ilustrativos, dependen del artefacto entrenado):

```json
{"predicciones": [{"order_id": "demo-1", "prob_retraso": 0.1234, "nivel_riesgo": "alto"}]}
```

## Reglas de diseño

- **Solo variables conocidas al aprobar el pago (decisión 4).** El esquema rechaza con error 422 cualquier campo que no conozca, como las fechas de despacho o entrega o la reseña. La fuga de datos se frena en la entrada, no solo en el entrenamiento.
- **Los faltantes se aceptan.** Cualquier variable puede venir vacía; el pipeline la imputa con lo aprendido en train (mediana o `unknown`).
- **Sin modelo, error claro.** Si no existe el artefacto, `/health` responde `sin_modelo` y `/predict` devuelve 503 con la instrucción para generarlo.
- **La probabilidad se lee como orden, no como tasa exacta.** En los meses nuevos la calibración sobrestima (riesgo medio 8,3 % contra 3,5 % real en test, decisión 9). Lo confiable es el nivel de riesgo.
- **Las variables se calculan antes de llamar a la API.** La API recibe la fila del pedido ya armada, como la produce `src/feature_engineering.py`. Calcularlas desde los datos crudos queda para una versión posterior.

## Tests

`tests/test_api.py`, con un artefacto sintético que no depende de `models/`: el esquema cubre todas las variables del modelo, las columnas con fuga se rechazan, `/predict` da lo mismo que `predict_risk`, los faltantes se imputan, los valores imposibles se rechazan y sin modelo devuelve 503.

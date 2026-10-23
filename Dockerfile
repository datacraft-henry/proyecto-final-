# Imagen de la API de riesgo de retraso (api/main.py).
# El modelo NO va dentro de la imagen: los .joblib no se versionan (decisión 7),
# así que se monta la carpeta models/ al correr el contenedor. Ver docs/api.md.
FROM python:3.12-slim

WORKDIR /app

# Solo lo que la API necesita para cargar el artefacto y predecir.
# scikit-learn fijo en 1.8.0, la versión con la que se entrena el modelo: con otra, el .joblib puede dar resultados inválidos.
RUN pip install --no-cache-dir \
    "pandas>=2.0" "numpy>=1.24" "scikit-learn==1.8.0" "joblib>=1.3" \
    "fastapi>=0.110" "uvicorn>=0.27" "holidays>=0.40"

COPY src/ src/
COPY api/ api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

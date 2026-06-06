# Amazon Product Reviews Classifier with PySpark

Clasificador binario de sentimiento para reseñas de productos de Amazon usando PySpark MLlib. El proyecto toma texto de reseñas y puntuaciones numéricas, construye etiquetas positivas/negativas y entrena una regresión logística con características TF-IDF.

## Objetivo

Predecir si una reseña expresa una valoración positiva o negativa a partir del texto:

- Positiva: `rating >= 4`
- Negativa: `rating <= 2`
- Las reseñas neutrales se excluyen del entrenamiento

## Dataset

El notebook está preparado para el dataset público Datafiniti Amazon Consumer Reviews of Amazon Products. Por tamaño y licencia, el CSV no se incluye en el repositorio.

Ruta usada por defecto en Databricks:

```text
/FileStore/tables/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
```

## Estructura

```text
.
├── Amazon Product Reviews Classifier with PySpark.ipynb
├── src/
│   └── amazon_reviews_classifier.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Pipeline

1. Carga del CSV con Spark.
2. Selección de columnas de texto y rating.
3. Creación de etiqueta binaria.
4. Normalización básica del texto.
5. Tokenización y eliminación de stop words.
6. Vectorización con HashingTF e IDF.
7. Entrenamiento con regresión logística.
8. Evaluación con AUC-ROC.

## Uso en Databricks

Importa el notebook y ejecuta las celdas en orden. También puedes ejecutar el script:

```bash
spark-submit src/amazon_reviews_classifier.py \
  --input /FileStore/tables/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
```

Para guardar predicciones:

```bash
spark-submit src/amazon_reviews_classifier.py \
  --input /FileStore/tables/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv \
  --predictions-output /FileStore/tables/amazon_review_predictions
```

## Requisitos

```bash
pip install -r requirements.txt
```

En Databricks normalmente PySpark ya está disponible en el runtime del cluster.

## Resultado esperado

El modelo base suele alcanzar un AUC razonable para una primera aproximación con TF-IDF y regresión logística. El rendimiento final depende de la versión del dataset, limpieza previa y partición train/test.


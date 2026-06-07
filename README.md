# Amazon Product Reviews Classifier with PySpark

Binary sentiment classifier for Amazon product reviews using PySpark MLlib. The project converts review ratings into positive and negative labels, preprocesses review text, builds TF-IDF features and trains a logistic regression classifier.

## Objective

Predict whether a review expresses positive or negative sentiment from its text:

- Positive: `rating >= 4`
- Negative: `rating <= 2`
- Neutral reviews are excluded from training

## Dataset

The notebook is prepared for the public Datafiniti Amazon Consumer Reviews of Amazon Products dataset. The CSV is not included in the repository because of size and licensing constraints.

Default Databricks path:

```text
/FileStore/tables/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
```

## Structure

```text
.
|-- Amazon Product Reviews Classifier with PySpark.ipynb
|-- src/
|   `-- amazon_reviews_classifier.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Pipeline

1. Load the CSV with Spark.
2. Select the review text and rating columns.
3. Create a binary target label.
4. Apply basic text normalization.
5. Tokenize text and remove stop words.
6. Build features with HashingTF and IDF.
7. Train a logistic regression classifier.
8. Evaluate with AUC-ROC.

## Databricks Usage

Import the notebook and run the cells in order. The script can also be executed with `spark-submit`:

```bash
spark-submit src/amazon_reviews_classifier.py \
  --input /FileStore/tables/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
```

To save predictions:

```bash
spark-submit src/amazon_reviews_classifier.py \
  --input /FileStore/tables/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv \
  --predictions-output /FileStore/tables/amazon_review_predictions
```

## Requirements

```bash
pip install -r requirements.txt
```

PySpark is usually already available in Databricks runtimes.

## Expected Result

The baseline model should reach a reasonable AUC for a first TF-IDF plus logistic regression approach. Final performance depends on the dataset version, cleaning strategy and train/test split.

## Security

This repository does not require credentials or tokens. Local data and generated outputs are kept out of Git through `.gitignore`.

"""Train a PySpark sentiment classifier for Amazon product reviews."""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover, Tokenizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, when


DEFAULT_INPUT = "/FileStore/tables/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a binary sentiment classifier with PySpark.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV path in DBFS or local filesystem.")
    parser.add_argument("--text-column", default="reviews.text", help="Source column containing review text.")
    parser.add_argument("--rating-column", default="reviews.rating", help="Source column containing numeric ratings.")
    parser.add_argument("--predictions-output", default=None, help="Optional path to write model predictions as CSV.")
    return parser


def load_reviews(spark: SparkSession, path: str, text_column: str, rating_column: str):
    data = spark.read.csv(path, header=True, inferSchema=True, multiLine=True, escape='"')
    selected = data.select(
        col(f"`{text_column}`").alias("review_text"),
        col(f"`{rating_column}`").cast("double").alias("rating"),
    ).dropna(subset=["review_text", "rating"])

    labeled = selected.withColumn(
        "label",
        when(col("rating") >= 4, 1.0).when(col("rating") <= 2, 0.0),
    ).dropna(subset=["label"])

    return labeled.withColumn("clean_text", lower(col("review_text"))).withColumn(
        "clean_text", regexp_replace(col("clean_text"), r"[^a-zA-Z\s]", " ")
    )


def build_pipeline() -> Pipeline:
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    model = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)
    return Pipeline(stages=[tokenizer, remover, hashing_tf, idf, model])


def main() -> None:
    args = build_parser().parse_args()
    spark = SparkSession.builder.appName("AmazonReviewsSentimentClassifier").getOrCreate()

    reviews = load_reviews(spark, args.input, args.text_column, args.rating_column)
    train, test = reviews.randomSplit([0.8, 0.2], seed=42)

    pipeline_model = build_pipeline().fit(train)
    predictions = pipeline_model.transform(test)

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    auc = evaluator.evaluate(predictions)
    print(f"Test AUC: {auc:.4f}")

    predictions.select("review_text", "rating", "label", "prediction", "probability").show(
        10, truncate=80
    )

    if args.predictions_output:
        predictions.select("review_text", "rating", "label", "prediction", "probability").write.mode(
            "overwrite"
        ).csv(args.predictions_output, header=True)

    spark.stop()


if __name__ == "__main__":
    main()

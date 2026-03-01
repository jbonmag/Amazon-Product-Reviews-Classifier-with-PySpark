#  Amazon Product Reviews Classifier with PySpark

This project demonstrates how to build a machine learning pipeline using **PySpark** on **Databricks** to classify product reviews from Amazon as **positive** or **negative** based on the review text.

---

##  Dataset

We use the [Datafiniti Amazon Consumer Reviews of Amazon Products (May19)](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products) dataset. You can upload the CSV file to Databricks DBFS via `/FileStore/tables/`.

---

##  Objective

Train a binary classifier using PySpark MLlib to predict sentiment (positive/negative) from the review text.

---

##  Technologies

- Apache Spark (PySpark)
- Databricks Notebook
- Spark MLlib
- Logistic Regression
- NLP: Tokenization, Stopword Removal, TF-IDF

---

##  Pipeline Steps

1. **Load Data** from CSV
2. **Select & Rename Columns** (`reviews.text`, `reviews.rating`)
3. **Label Creation**:
   - Rating >= 4 → Positive (1)
   - Rating <= 2 → Negative (0)
4. **Text Preprocessing**:
   - Lowercase
   - Remove punctuation
5. **ML Pipeline**:
   - Tokenizer → StopWordsRemover → HashingTF → IDF → Logistic Regression
6. **Train/Test Split**
7. **Evaluation with AUC**

---

##  Results

- Achieved an AUC of **~0.89**, indicating solid performance for a basic logistic regression classifier on text data.

---

##  How to Use

Paste this into a Databricks Python notebook:

```python
file_path = "/FileStore/tables/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
# See full code in notebook or accompanying script
```

---

##  Sample Output

| texto                              | rating | etiqueta | prediction | probability        |
|------------------------------------|--------|----------|------------|--------------------|
| "Great product, works perfectly."  |   5    |    1     |    1.0     | [0.12, 0.88]       |
| "Terrible, waste of money."        |   1    |    0     |    0.0     | [0.96, 0.04]       |

---

##  Folder Structure

```
📁 project/
├── Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
├── amazon_reviews_pipeline.ipynb
└── README.md
```

---

`@jubonet-`

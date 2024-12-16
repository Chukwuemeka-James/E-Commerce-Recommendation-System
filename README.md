# E-Commerce-Recommendation-System-Machine-Learning-Product-Recommendation-system

This project demonstrates the use of various data analysis and recommendation system techniques on a marketing dataset. The dataset consists of product details, reviews, ratings, and other attributes, and is used to develop a recommendation system using both content-based and collaborative filtering methods.

## Installation

To run this project, you'll need Python 3.x and the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `spacy`

### Steps

### 1. Data Loading and Data Processing and Cleaning

Missing values in columns like `Product Rating`, `Product Reviews Count`, `Product Category`, etc., are filled with appropriate defaults (e.g., `0` for ratings, empty string for text fields).

```python
train_data['Product Rating'] = train_data['Product Rating'].fillna(0)
train_data['Product Reviews Count'] = train_data['Product Reviews Count'].fillna(0)
train_data['Product Category'] = train_data['Product Category'].fillna('')
train_data['Product Brand'] = train_data['Product Brand'].fillna('')
train_data['Product Description'] = train_data['Product Description'].fillna('')
```

### 2. Data Transformation

The dataset's column names are mapped to more intuitive names, and some columns are further transformed to numeric types as needed. We also extract tags from product descriptions using SpaCy's NLP tools.

```python
train_data.rename(columns=column_name_mapping, inplace=True)
```

### 3. Recommendation System

Two types of recommendation systems are implemented:

- **Content-Based Recommendations:** Uses item attributes like `Tags`, `Description`, `Category`, etc., to recommend similar products using cosine similarity based on TF-IDF vectors.

```python
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
```

- **Collaborative Filtering Recommendations:** Recommends products based on user preferences. It calculates cosine similarities between users' ratings to suggest items rated highly by similar users.

```python
user_similarity = cosine_similarity(user_item_matrix)
```

### 4. Hybrid Recommendation System

A hybrid recommendation system that combines both content-based and collaborative filtering methods is also included to provide better results by merging both approaches.
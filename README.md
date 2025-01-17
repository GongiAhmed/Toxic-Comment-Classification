# Toxic Comment Classification

This project explores various approaches to classify toxic comments using natural language processing techniques.  The goal is to identify comments that are toxic, severe toxic, obscene, threatening, insulting, or express identity hate.

## Dataset

The project utilizes the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle ([link to dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)).

## Approaches

Three different approaches were implemented and compared:

1. **TF-IDF with Logistic Regression:**  Term Frequency-Inverse Document Frequency (TF-IDF) was used to vectorize the text data, followed by a Logistic Regression classifier for each toxicity label.

2. **Count Vectorizer with Naive Bayes:**  A Count Vectorizer was used for text vectorization, and a Multinomial Naive Bayes classifier was trained for each label.

3. **Word Embeddings (GloVe) with LSTM:** Pre-trained GloVe word embeddings were used to represent words, and a Long Short-Term Memory (LSTM) neural network was trained for multi-label classification.


## Results

| Approach                           | Accuracy | F1 Score | Precision | Recall |
|------------------------------------|----------|----------|-----------|--------|
| TF-IDF + Logistic Regression       | 97.38\%  | 0.43    |  0.78      | 0.29   |
| Count Vectorizer + Naive Bayes      | 97\%     | 0.46    |   0.49     | 0.49   |
| Word Embeddings (GloVe) + LSTM | 97.87\%  | 0.41     | 0.48 | 0.31    |

<p align="center">
  <img src="https://github.com/GongiAhmed/Toxic-Comment-Classification/blob/main/Toxic%20Comment%20Classification/images/results.png" />
</p>


## Visualizations

The notebook includes visualizations of the data, including:

* Correlation Heatmap of Toxicity Labels
* Distribution of Comment Lengths
* WordCloud of Toxic Comments
* Comparison of Classification Approaches

## Usage

1. Clone the repository: `git clone https://github.com/yourusername/your-repo-name.git`
2. Install dependencies (listed in requirements.txt)
3. Run the `Toxic_Comment_Classification.ipynb` notebook. 


## Further Improvements

* Experiment with different hyperparameters for the models.
* Explore other text preprocessing techniques (e.g., stemming, lemmatization).
* Implement more advanced deep learning models (e.g., BERT, RoBERTa).
* Address class imbalance issues for better performance on less frequent labels.

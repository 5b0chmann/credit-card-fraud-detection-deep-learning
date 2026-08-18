# Credit Card Fraud Detection with Machine Learning and Deep Learning

This project investigates different machine learning and deep learning approaches for detecting fraudulent credit card transactions in a highly imbalanced dataset.

The objective was not only to maximize predictive performance, but also to understand which evaluation metrics are most meaningful for fraud detection and whether deep learning actually outperforms classical machine learning models on tabular data.

## Executive Summary

Models evaluated:

- Logistic Regression
- Random Forest
- Deep Neural Network (DNN)
- Autoencoder

Key findings:

- Random Forest achieved the best overall balance between Precision and Recall

- Deep Neural Networks achieved strong ROC-AUC scores but did not outperform Random Forest on tabular data

- ROC-AUC alone is insufficient for evaluating highly imbalanced fraud datasets

- Autoencoder-based anomaly detection performed significantly worse than supervised approaches

## Business Problem

Credit card fraud detection is a highly imbalanced classification problem.

Only a tiny fraction of transactions are fraudulent, making traditional evaluation metrics such as Accuracy misleading.

The challenge is to identify fraudulent transactions while minimizing false alarms.

## Project Workflow

1. Exploratory Data Analysis
2. Data Quality Assessment
3. Baseline Model (Logistic Regression)
4. Random Forest Classifier
5. Deep Neural Network
6. Autoencoder Anomaly Detection
7. Model Comparison and Evaluation

## Results
| Model | Precision | Recall | F1 | ROC-AUC |
|--------|----------|---------|---------|---------|
| Logistic Regression | 0.66 | 0.79 | 0.72 | 0.9657 |
| Random Forest | 0.96 | 0.73 | 0.83 | 0.9663 |
| DNN | 0.89 | 0.77 | 0.82 | 0.9642 |
| Autoencoder | 0.15 | 0.58 | 0.24 | 0.9145 |

### Best Model

Random Forest achieved the best overall performance with
a Precision of 0.96, an F1-Score of 0.83 and a ROC-AUC of 0.9663.

## Development Environment

**Local Development:**
  - Exploratory Data Analysis (EDA)
  - Logistic Regression
  - Random Forest

**Google Colab:**
  - Deep Neural Network (TensorFlow/Keras)
  - Autoencoder (TensorFlow/Keras)

The deep learning models were developed and evaluated in Google Colab to take advantage of a cloud-based machine learning environment.

## Lessons Learned

This project demonstrated that:

- Deep Learning does not automatically outperform classical machine learning methods
- Evaluation metrics must be chosen carefully for imbalanced datasets
- Model interpretability and false positive rates are critical in fraud detection
- Random Forest remains a strong benchmark for structured tabular data

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Google Colab
- GitHub

# Credit Card Fraud Detection with Machine Learning and Deep Learning

This project investigates different machine learning and deep learning approaches for detecting fraudulent credit card transactions in a highly imbalanced dataset.

The objective was not only to maximize predictive performance, but also to understand which evaluation metrics are most meaningful for fraud detection and whether deep learning actually outperforms classical machine learning models on tabular data.

## Executive Summary

Models evaluated:
- Logistic Regression
- Random Forest
- Deep Neural Network (DNN)
- Autoencoder

Additional analysis:
- Threshold optimization demonstrated that model performance depends not only on model selection but also on the chosen decision threshold
- Hyperparameter optimization showed that model performance can be further improved through systematic parameter tuning
- Combining threshold optimization and hyperparameter tuning highlighted the importance of business-driven model configuration

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
8. Threshold Optimization
9. Hyperparameter Optimization
10. Final Model Evaluation

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

### Threshold Optimization

After identifying Random Forest as the best-performing model, an additional threshold optimization study was conducted.

The default threshold of 0.5 achieved the highest precision (0.96) with only 3 false positive predictions.

A threshold of 0.3 achieved the highest F1-Score (0.84) while detecting more fraudulent transactions.

This analysis demonstrated that threshold selection can be as important as model selection itself and should be aligned with the business objective.

#### Hyperparameter Optimization

The Random Forest classifier was further optimized using RandomizedSearchCV with F1-Score as the optimization objective.

Compared to the baseline Random Forest model:

- Precision improved from 0.9583 to 0.9589
- Recall improved from 0.7263 to 0.7368
- F1-Score improved from 0.8263 to 0.8333

The optimization increased fraud detection performance while maintaining an extremely low number of false positive predictions.

The experiment also demonstrated that improving one metric does not necessarily improve all metrics, as ROC-AUC slightly decreased while F1-Score increased.

#### Final Model Evaluation

The final experiment combined:

- Random Forest Hyperparameter Optimization
- Optimized Decision Threshold (0.3)

Results:

- Precision: 0.9125
- Recall: 0.7684
- F1-Score: 0.8343
- ROC-AUC: 0.9338

The final model detected 73 fraudulent transactions while generating only 7 false positive predictions.

An important project finding was that threshold optimization had a greater impact on business-relevant performance metrics than hyperparameter optimization alone.

This demonstrates that practical fraud detection performance depends not only on algorithm selection, but also on threshold strategy and model configuration.

## Lessons Learned

This project demonstrated that:

- Deep Learning does not automatically outperform classical machine learning methods
- Evaluation metrics must be chosen carefully for imbalanced datasets
- Model interpretability and false positive rates are critical in fraud detection
- Random Forest remains a strong benchmark for structured tabular data
- Decision threshold selection can be as important as model selection itself
- Hyperparameter optimization can improve model performance, but threshold selection may have an even greater impact on business outcomes
- Business-driven model configuration is often as important as algorithm selection itself

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

# Advanced Fraud Detection Using Ensemble Learning, Feature Engineering, and Explainable AI

## Overview

This project presents a comprehensive machine learning pipeline for binary classification using advanced feature engineering, ensemble learning, and explainable artificial intelligence techniques. The objective is to improve predictive performance beyond traditional machine learning approaches while maintaining model transparency and interpretability.

The project begins with exploratory data analysis and a baseline Logistic Regression model. Multiple feature engineering techniques are then applied to enrich the dataset, followed by the development of high-performance LightGBM and CatBoost models. Finally, an ensemble model combines the strengths of both algorithms to achieve improved classification performance.

To ensure interpretability, Feature Importance, Permutation Importance, and SHAP (SHapley Additive Explanations) analyses are incorporated into the workflow.

---

## Objectives

The primary objectives of this project are:

* Develop a robust binary classification system.
* Compare baseline and advanced machine learning models.
* Investigate the impact of feature engineering on performance.
* Improve predictive accuracy through ensemble learning.
* Explain model predictions using modern Explainable AI techniques.
* Generate production-ready predictions for deployment or competition submission.

---

## Dataset Description

The dataset consists of anonymized numerical features used for binary classification.

### Training Dataset

The training dataset contains:

| Column  | Description                  |
| ------- | ---------------------------- |
| ID_code | Unique sample identifier     |
| target  | Binary target variable       |
| var_*   | Anonymous numerical features |

### Test Dataset

The test dataset contains:

| Column  | Description                  |
| ------- | ---------------------------- |
| ID_code | Unique sample identifier     |
| var_*   | Anonymous numerical features |

### Classification Task

The goal is to predict the probability that a sample belongs to the positive class (`target = 1`).

Because the dataset is imbalanced, ROC-AUC is selected as the primary evaluation metric.

---

# Methodology

The project is divided into multiple phases.

---

## Phase 0: Exploratory Data Analysis

Exploratory Data Analysis (EDA) is performed to understand the characteristics of the dataset.

### Analysis Performed

#### Missing Value Analysis

The total number of missing values is calculated for both training and testing datasets.

```python
train.isnull().sum().sum()
test.isnull().sum().sum()
```

#### Class Distribution Analysis

The target distribution is examined to determine the degree of class imbalance.

```python
y.value_counts()
```

#### Statistical Summary

Basic statistics are generated for all numerical features.

```python
X_raw.describe()
```

#### Visualization

The following visualizations are created:

* Target class distribution
* Feature distribution comparison between classes
* Histograms of selected variables

These visualizations help identify feature behavior and potential class separability.

---

## Phase 1: Baseline Model Development

A Logistic Regression model is used as a benchmark.

### Data Preparation

The dataset is divided into training and validation sets using stratified sampling to preserve class proportions.

```python
train_test_split(
    X_raw,
    y,
    test_size=0.2,
    stratify=y
)
```

### Feature Scaling

Standardization is applied using StandardScaler.

```python
scaler.fit_transform()
```

### Logistic Regression Training

```python
LogisticRegression(max_iter=1000)
```

### Evaluation

Performance is measured using ROC-AUC score.

The baseline establishes a reference point for evaluating future improvements.

---

# Feature Engineering

Feature engineering is the most important component of this project and significantly improves model performance.

---

## Count Encoding Features

Count encoding captures how frequently feature values appear in the dataset.

### Process

1. Combine train and test datasets.
2. Round feature values to two decimal places.
3. Compute occurrence frequency of each value.
4. Replace values with their frequencies.

Example:

| Original Value | Frequency |
| -------------- | --------- |
| 1.25           | 340       |
| -0.87          | 45        |
| 0.01           | 982       |

Generated Features:

```text
cnt_var_0
cnt_var_1
cnt_var_2
...
```

### Motivation

Rare feature values often contain stronger predictive information than common values.

---

## Row-Based Statistical Features

Several statistical features are generated for every sample.

### Generated Features

| Feature    | Description                    |
| ---------- | ------------------------------ |
| row_mean   | Average feature value          |
| row_std    | Standard deviation             |
| row_sum    | Sum of features                |
| row_max    | Maximum value                  |
| row_min    | Minimum value                  |
| row_median | Median value                   |
| row_range  | Difference between max and min |
| row_skew   | Skewness                       |
| pos_ratio  | Ratio of positive values       |
| neg_ratio  | Ratio of negative values       |
| abs_sum    | Sum of absolute values         |

### Motivation

These features capture global sample-level behavior that individual variables may fail to represent.

---

## Count-Based Aggregate Features

Aggregate statistics are calculated from count-encoded features.

Generated Features:

```text
cnt_mean
cnt_std
cnt_max
cnt_min
cnt_sum
```

These summarize the rarity characteristics of each sample.

---

## Inverse Features

Additional nonlinear transformations are generated.

For every feature:

```python
inverse_feature = 1 - feature
```

Generated Features:

```text
inv_var_0
inv_var_1
inv_var_2
...
```

### Motivation

Inverse transformations may expose relationships not immediately visible in the original feature space.

---

## Final Feature Space

After feature engineering, the dataset expands substantially and contains:

* Original Features
* Count Encoded Features
* Statistical Features
* Aggregate Count Features
* Inverse Features

This enriched feature space provides significantly more predictive information.

---

# Advanced Model Development

Two gradient boosting algorithms are trained using stratified cross-validation.

---

## LightGBM

LightGBM is a gradient boosting framework designed for efficiency and high predictive performance.

### Hyperparameters

```python
n_estimators = 2500
learning_rate = 0.03
max_depth = 5
num_leaves = 32
subsample = 0.8
colsample_bytree = 0.15
```

### Training Strategy

* Stratified 5-Fold Cross Validation
* Out-of-Fold Prediction Generation
* Test Prediction Averaging

### Advantages

* Fast training
* High scalability
* Strong predictive accuracy

---

## CatBoost

CatBoost is another gradient boosting algorithm designed to reduce overfitting and improve generalization.

### Hyperparameters

```python
iterations = 2500
learning_rate = 0.05
depth = 4
l2_leaf_reg = 3
```

### Additional Features

* GPU acceleration
* Early stopping
* Automatic regularization
* Robust handling of complex feature interactions

### Training Strategy

* Stratified 5-Fold Cross Validation
* Out-of-Fold Predictions
* Best Model Selection

---

# Ensemble Learning

Predictions from LightGBM and CatBoost are combined using weighted averaging.

## Ensemble Formula

The weight assigned to each model is proportional to its ROC-AUC score.

Final Prediction:

```text
(LightGBM_AUC × LightGBM_Prediction +
 CatBoost_AUC × CatBoost_Prediction)

/

(LightGBM_AUC + CatBoost_AUC)
```

### Benefits

* Reduced variance
* Better generalization
* Increased robustness
* Improved overall ROC-AUC

---

# Model Evaluation

The following models are evaluated:

1. Logistic Regression
2. LightGBM
3. CatBoost
4. Weighted Ensemble

Evaluation Metric:

```text
ROC-AUC Score
```

Example Results:

| Model               | ROC-AUC |
| ------------------- | ------- |
| Logistic Regression | 0.85    |
| LightGBM            | 0.90    |
| CatBoost            | 0.91    |
| Ensemble            | 0.92    |

The ensemble consistently achieves the highest performance.

---

# Explainable Artificial Intelligence

Understanding model decisions is critical.

Three explainability techniques are implemented.

---

## Feature Importance

CatBoost feature importance scores are extracted.

```python
get_feature_importance()
```

The top features are visualized using bar charts.

---

## Permutation Importance

Permutation Importance measures the decrease in model performance when feature values are randomly shuffled.

Benefits:

* Model agnostic
* Reliable importance estimation
* Captures true predictive contribution

---

## SHAP Analysis

SHAP provides both global and local interpretability.

### SHAP Summary Plot

Displays:

* Feature importance
* Impact direction
* Feature value distribution

### SHAP Bar Plot

Ranks features according to overall influence.

### SHAP Dependence Plot

Shows relationships between feature values and prediction outcomes.

---

# Visualizations Generated

The pipeline automatically creates multiple visualizations.

## Exploratory Analysis

* Target Distribution Plot
* Feature Distribution Histograms

## Model Evaluation

* ROC-AUC Comparison Chart

## Explainability

* Feature Importance Plot
* Permutation Importance Plot
* SHAP Summary Plot
* SHAP Feature Ranking Plot
* SHAP Dependence Plot

---

# Output

The final predictions are saved as:

```text
submission.csv
```

Format:

| ID_code  | target |
| -------- | ------ |
| sample_1 | 0.032  |
| sample_2 | 0.871  |
| sample_3 | 0.115  |

This file can be directly uploaded for competition evaluation.

---

# Installation

Clone the repository:

```bash
git clone Advanced-Fraud-Detection-using-Ensemble-Learning--Feature-Engineering-and-Explainable-AI.git

cd Advanced-Fraud-Detection-using-Ensemble-Learning--Feature-Engineering-and-Explainable-AI
```

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
pip install lightgbm
pip install catboost
pip install shap
```
---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* LightGBM
* CatBoost
* SHAP

---

# Key Findings

* Count-encoded rarity features are highly predictive.
* Statistical row-level features improve class separability.
* Gradient boosting significantly outperforms linear models.
* Weighted ensemble learning provides the best performance.
* SHAP analysis confirms that feature rarity strongly influences predictions.
* Explainability techniques make model decisions transparent and trustworthy.

---

# Future Improvements

Potential extensions include:

* Hyperparameter optimization using Optuna
* Feature selection using Boruta
* Model stacking
* Neural network integration
* Automated machine learning pipelines
* Real-time deployment using FastAPI or Flask
* Cloud deployment using AWS or Azure

---

# Author

Iman Fatima

Computer Engineering Student

Interests:

* Machine Learning
* Artificial Intelligence
* Embedded Systems
* FPGA Design

---


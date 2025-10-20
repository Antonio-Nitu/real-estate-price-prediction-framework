# A Methodological Approach to Predicting Housing Prices using Machine Learning

## Abstract

This repository presents a comprehensive machine learning framework for predicting residential housing prices. The project undertakes a systematic investigation using the Ames Housing dataset, sourced from the Kaggle "House Prices: Advanced Regression Techniques" competition. The core of this work lies in a robust, modular pipeline that handles data preprocessing, comparative model evaluation, and hyperparameter optimization. A comparative study of Linear Regression, Decision Tree, and Random Forest models is conducted, with the final optimized Random Forest model achieving a **Root Mean Squared Logarithmic Error (RMSLE) of 0.14862** on the hold-out test set, demonstrating strong predictive performance. The project's architecture emphasizes reproducibility, maintainability, and a deep analysis of the underlying data.

-----

## 1\. Methodology

The project follows a structured machine learning workflow, from data ingestion to model deployment and analysis. The entire process is encapsulated in a modular codebase designed for clarity and reusability.

### 1.1. Data Source and Characteristics

The study utilizes the **Ames Housing dataset**, which contains 79 explanatory variables describing various aspects of residential homes in Ames, Iowa. The dataset is split into:

  * **Training Set:** 1460 instances with features and the target variable, `SalePrice`.
  * **Testing Set:** 1459 instances with features, used to evaluate the final model's generalization performance.

### 1.2. Preprocessing Pipeline

A critical component of this project is the `scikit-learn` preprocessing pipeline, which ensures that all transformations are applied consistently and prevents data leakage. The pipeline is constructed using a `ColumnTransformer` to apply type-specific operations.

  * **Numerical Features:**

    1.  **Imputation:** Missing numerical values are imputed using the **median** of their respective columns. The median is chosen over the mean for its robustness to outliers, which are common in real estate data (e.g., unusually large lot areas or sale prices).
    2.  **Scaling:** Features are standardized using `StandardScaler`, which removes the mean and scales to unit variance. This is crucial for models like Linear Regression and helps gradient-based optimizers converge faster.

  * **Categorical Features:**

    1.  **Imputation:** Missing categorical values are imputed with the constant string `'None'`. In this dataset, `NaN` often carries explicit meaning (e.g., no garage, no pool), making this a more accurate strategy than using the mode.
    2.  **Encoding:** Features are transformed using `OneHotEncoder`. This converts categorical variables into a numerical format without imposing an arbitrary ordinal relationship between categories.

### 1.3. Modeling and Evaluation

A comparative study was conducted to evaluate the performance of three distinct regression algorithms.

1.  **Linear Regression:** Serves as a baseline model to establish a performance benchmark.
2.  **Decision Tree Regressor:** A non-linear model capable of capturing complex interactions between features.
3.  **Random Forest Regressor:** An ensemble method that builds multiple decision trees and aggregates their results. It is robust to overfitting and generally provides higher accuracy.

The models are evaluated on a validation set (an 80/20 split of the original training data) using the following metrics:

  * **Root Mean Squared Error (RMSE):** Measures the standard deviation of the prediction errors.
  * **Mean Absolute Error (MAE):** Measures the average magnitude of the errors.
  * **R-squared ($R^2$):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

### 1.4. Hyperparameter Optimization

The most promising model, the **Random Forest Regressor**, was further optimized. `GridSearchCV` was employed to perform an exhaustive search over a predefined grid of hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`) to identify the optimal configuration that minimizes the cross-validated error.

-----

## 2\. Results and Analysis

### 2.1. Model Performance Comparison

The Random Forest Regressor demonstrated superior performance across all metrics, confirming the effectiveness of ensemble methods on this type of structured dataset.

| Model               | RMSE (on validation set) | MAE (on validation set) | R² (on validation set) |
| ------------------- | ------------------------ | ----------------------- | ---------------------- |
| Linear Regression   | 65385.31                 | 21125.45                | 0.44                   |
| Decision Tree       | 42463.80                 | 27542.52                | 0.76                   |
| **Random Forest** | **29179.84** | **17611.60** | **0.88** |

### 2.2. Feature Importance Analysis

An analysis of the final tuned Random Forest model revealed the key drivers of house prices. As expected, `OverallQual` (overall material and finish quality) was the most influential feature. Other significant predictors included `GrLivArea` (above-grade living area square footage) and the total square footage of the basement. This analysis provides actionable insights into the housing market and confirms the model's alignment with real-world valuation principles.

-----

## 3\. Code Architecture

The project's codebase is modular, promoting the principles of separation of concerns.

  * `main.py`: The main entry point that orchestrates the entire pipeline from data loading to prediction generation.
  * `data_preprocessing.py`: Contains all functions related to data loading, cleaning, and the construction of the `scikit-learn` preprocessing pipeline.
  * `model_training.py`: Houses the logic for defining, training, evaluating, and tuning the machine learning models.
  * `visualizations.py`: Includes functions for generating plots, such as the feature importance chart and model performance visualizations.

-----

## 4\. Replicating the Experiment

To replicate this experiment, follow the steps below.

**1. Clone the Repository:**

```bash
git clone https://github.com/Antonio-Nitu/real-estate-price-prediction-framework.git
cd real-estate-price-prediction-framework
```

**2. Set Up the Environment:**

```bash
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**4. Run the Pipeline:**

```bash
python main.py
```

The script will execute the full pipeline, generating the model comparison CSV, the final prediction file, and all associated visualizations in the `plots/` directory.

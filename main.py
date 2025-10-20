# main.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import functions from other modules
from data_preprocessing import load_data, create_preprocessor
from model_training import train_and_evaluate_models, tune_random_forest
from visualizations import plot_feature_importance

# --- Configuration for Kaggle Dataset ---
TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'
TARGET_COLUMN = 'SalePrice'

def main():
    """
    Main function to run the entire pipeline:
    1. Compare models on a validation set and save results.
    2. Train the best model on all data and generate final submission file.
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # --- Part 1: Load and Prepare Data ---
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)
    test_ids = test_df['Id']
    
    X_train_full = train_df.drop(columns=[TARGET_COLUMN, 'Id'])
    y_train_full = train_df[TARGET_COLUMN]
    X_test_final = test_df.drop(columns=['Id'])
    
    # Align columns to prevent errors
    X_train_full, X_test_final = X_train_full.align(X_test_final, join='outer', axis=1, fill_value=0)

    numeric_features = X_train_full.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train_full.select_dtypes(include=['object']).columns.tolist()
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    # --- Part 2: Model Comparison ---
    print("--- Step 1: Comparing Models on a Validation Set ---")
    # Split the full training data to create a temporary validation set for comparison
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    results_df = train_and_evaluate_models(X_train_split, y_train_split, X_val, y_val, preprocessor)
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\n✅ Model comparison results saved to model_comparison_results.csv")
    print(results_df)

    # --- Part 3: Train Final Model and Generate Submission ---
    print("\n--- Step 2: Training Final Model on ALL Data and Predicting ---")
    # Now, use the FULL training data to train the best model (tuned Random Forest)
    final_model = tune_random_forest(X_train_full, y_train_full, preprocessor)

    print("\n--- Generating Predictions for Submission ---")
    predictions = final_model.predict(X_test_final)
    
    submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})
    submission_df.to_csv('submission.csv', index=False)

    print("\n✅ Final predictions saved to submission.csv!")
    print("Here's a preview:")
    print(submission_df.head())

    # (Optional) Generate feature importance plot from the final model
    try:
        ohe_features = final_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        all_features = numeric_features + list(ohe_features)
        plot_feature_importance(final_model, all_features)
        print("\nFeature importance plot saved.")
    except Exception as e:
        print(f"\nCould not generate feature importance plot: {e}")


if __name__ == '__main__':
    main()

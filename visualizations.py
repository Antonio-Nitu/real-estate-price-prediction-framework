# visualizations.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_model_performance(y_test, y_pred, model_name):
    """Saves a scatter plot of actual vs. predicted values."""
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs. Predicted - {model_name}')
    plt.savefig(f'plots/{model_name.replace(" ", "")}_predictions.jpeg', dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names):
    """Saves a bar plot of top 15 feature importances."""
    importances = model.named_steps['regressor'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.jpeg', dpi=300)
    plt.close()
    
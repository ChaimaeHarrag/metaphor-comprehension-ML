# 🧠 Metaphor Comprehension Prediction using Machine Learning
# Author: Chaimae Harrag
# This project compares Ordinal Logistic Regression, Random Forest, and Gradient Boosting
# to model metaphor comprehension based on age, fluid intelligence, and working memory.

# 📦 1. Install required packages (for Google Colab)
!pip install mord -q
!pip install statsmodels seaborn pandas matplotlib scikit-learn scipy -q

# 📙 2. Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import mord
from google.colab import files

# Set random seed for reproducibility
np.random.seed(42)

# 🌐 3. Load and preprocess dataset
try:
    uploaded = files.upload()
    df = pd.read_csv(next(iter(uploaded)))
    required_cols = ['Age', 'FluidIntelligence', 'ComplexSpanScore', 'MetaphorComprehension']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataset must contain columns: {required_cols}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Validate and preprocess
df = df.dropna()
if not df['MetaphorComprehension'].isin([1, 2, 3, 4]).all():
    raise ValueError("MetaphorComprehension must be in range 1–4")
df['MetaphorComprehension'] = df['MetaphorComprehension'] - 1

scaler = StandardScaler()
df[['Age', 'FluidIntelligence', 'ComplexSpanScore']] = scaler.fit_transform(
    df[['Age', 'FluidIntelligence', 'ComplexSpanScore']]
)

# 🔧 4. Split data
X = df[['Age', 'FluidIntelligence', 'ComplexSpanScore']]
y = df['MetaphorComprehension']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 5. Initialize models
models = {
    'LogisticIT': mord.LogisticIT(alpha=0.1),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# 📆 Create output directory
os.makedirs("outputs", exist_ok=True)

# 📊 6. Train, evaluate, visualize
results = {}
feature_names = ['Age', 'FluidIntelligence', 'ComplexSpanScore']

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=['Abstract Complete', 'Abstract Partial',
                                                 'Concrete', 'Other/Unrelated'], output_dict=True)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    results[name] = {
        'Accuracy': accuracy,
        'MAE': mae,
        'Classification Report': pd.DataFrame(report).transpose(),
        'CV Mean Accuracy': cv_scores.mean(),
        'CV Std': cv_scores.std()
    }

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Abstract Complete', 'Abstract Partial', 'Concrete', 'Other/Unrelated'],
                yticklabels=['Abstract Complete', 'Abstract Partial', 'Concrete', 'Other/Unrelated'])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrix_{name}.png")
    plt.close()

    results[name]['Classification Report'].to_csv(f"outputs/classification_report_{name}.csv")

    # LogisticIT Coefficients
    if name == 'LogisticIT':
        coef = model.coef_
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef[0]
        })
        sns.barplot(x='Coefficient', y='Feature', data=coef_df, color='skyblue')
        plt.axvline(0, color='gray', linestyle='--')
        plt.title("LogisticIT Coefficients")
        plt.tight_layout()
        plt.savefig("outputs/logistic_coefficients.png")
        plt.close()

# 📉 7. Feature Importance (tree models)
for name in ['RandomForest', 'GradientBoosting']:
    model = models[name]
    importances = model.feature_importances_
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances, y=feature_names, palette='viridis')
    plt.title(f"Feature Importance - {name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"outputs/feature_importance_{name}.png")
    plt.close()

# 🔢 8. Correlation Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(df[feature_names].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png")
plt.close()

# 📃 9. Save summary table
summary = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['Accuracy'] for m in results],
    'MAE': [results[m]['MAE'] for m in results],
    'CV Mean Accuracy': [results[m]['CV Mean Accuracy'] for m in results],
    'CV Std': [results[m]['CV Std'] for m in results]
})
summary.to_csv("outputs/model_comparison.csv", index=False)
print("\nModel Comparison Summary:\n", summary)

# 🚚 10. Download outputs as zip
!zip -r outputs_ml_comparison.zip outputs
files.download("outputs_ml_comparison.zip")

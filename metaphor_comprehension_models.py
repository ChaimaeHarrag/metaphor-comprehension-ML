# 🧠 Metaphor Comprehension Prediction using Machine Learning
# Author: Chaimae Harrag
# This project compares Ordinal Logistic Regression, Random Forest, and Gradient Boosting
# to model metaphor comprehension based on age, fluid intelligence, and working memory.

# 📚 1. Import libraries
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

# Set random seed for reproducibility
np.random.seed(42)

# 📂 2. Load and preprocess dataset
df = pd.read_csv("metaphor_comprehension.csv")

# Ensure required columns exist
required_cols = ['Age', 'FluidIntelligence', 'ComplexSpanScore', 'MetaphorComprehension']
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"Dataset must contain columns: {required_cols}")

# Validate comprehension scores
if not df['MetaphorComprehension'].isin([1, 2, 3, 4]).all():
    raise ValueError("MetaphorComprehension must be in range 1–4")

# Adjust labels to 0–3
df['MetaphorComprehension'] = df['MetaphorComprehension'] - 1

# Drop missing data
df = df.dropna()

# Standardize predictors
scaler = StandardScaler()
df[['Age', 'FluidIntelligence', 'ComplexSpanScore']] = scaler.fit_transform(
    df[['Age', 'FluidIntelligence', 'ComplexSpanScore']]
)

# 🔧 3. Prepare data
X = df[['Age', 'FluidIntelligence', 'ComplexSpanScore']]
y = df['MetaphorComprehension']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📊 4. Define models
models = {
    'OrdinalLogistic': mord.LogisticIT(alpha=0.1),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Create output directory
os.makedirs("outputs", exist_ok=True)

# 📈 5. Train and evaluate
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

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

    # Save confusion matrix
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

    # Save classification report
    results[name]['Classification Report'].to_csv(f"outputs/classification_report_{name}.csv")

# 🌟 6. Feature importance (for tree-based models)
for name in ['RandomForest', 'GradientBoosting']:
    model = models[name]
    importances = model.feature_importances_
    feature_names = ['Age', 'FluidIntelligence', 'ComplexSpanScore']

    plt.bar(feature_names, importances)
    plt.title(f"Feature Importance - {name}")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"outputs/feature_importance_{name}.png")
    plt.close()

# 📋 7. Save model summary
summary = pd.DataFrame({
    'Model': [name for name in results],
    'Accuracy': [results[name]['Accuracy'] for name in results],
    'MAE': [results[name]['MAE'] for name in results],
    'CV Mean Accuracy': [results[name]['CV Mean Accuracy'] for name in results],
    'CV Std': [results[name]['CV Std'] for name in results]
})
summary.to_csv("outputs/model_comparison.csv", index=False)
print("\nModel Comparison Summary:\n", summary)

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

try:
    import mord
except ImportError:
    raise ImportError("Please install the 'mord' package using pip install mord before running this script.")

# Set random seed
np.random.seed(42)

# 📂 2. Load and preprocess data
df = pd.read_csv("metaphor_comprehension.csv")

# Validate structure
required_cols = ['Age', 'FluidIntelligence', 'ComplexSpanScore', 'MetaphorComprehension']
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"Dataset must contain the columns: {required_cols}")

# Ensure MetaphorComprehension is ordinal (1–4)
if not df['MetaphorComprehension'].isin([1, 2, 3, 4]).all():
    raise ValueError("MetaphorComprehension values must be in the range 1–4")

df['MetaphorComprehension'] -= 1  # Shift to 0–3
df = df.dropna()

# Scale predictors
scaler = StandardScaler()
df[['Age', 'FluidIntelligence', 'ComplexSpanScore']] = scaler.fit_transform(
    df[['Age', 'FluidIntelligence', 'ComplexSpanScore']]
)

# 🔧 3. Train-test split
X = df[['Age', 'FluidIntelligence', 'ComplexSpanScore']]
y = df['MetaphorComprehension']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📊 4. Define models
models = {
    'OrdinalLogistic': mord.LogisticIT(alpha=0.1),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Create output folder
os.makedirs("outputs", exist_ok=True)

# 📈 5. Train, evaluate, visualize
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                    target_names=['Abstract Complete', 'Abstract Partial',
                                                  'Concrete', 'Other/Unrelated'],
                                    output_dict=True)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    results[name] = {
        'Accuracy': accuracy,
        'MAE': mae,
        'Classification Report': pd.DataFrame(report).transpose(),
        'CV Mean Accuracy': cv_scores.mean(),
        'CV Std': cv_scores.std()
    }

    # 📌 Confusion matrix
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

    # 📌 Save classification report
    results[name]['Classification Report'].to_csv(f"outputs/classification_report_{name}.csv")

# 📉 6. Feature importance for tree-based models
for name in ['RandomForest', 'GradientBoosting']:
    model = models[name]
    importances = model.feature_importances_
    feature_names = ['Age', 'FluidIntelligence', 'ComplexSpanScore']

    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances, y=feature_names, palette='viridis')
    plt.title(f"Feature Importance - {name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"outputs/feature_importance_{name}.png")
    plt.close()

# 🔬 7. Correlation matrix
plt.figure(figsize=(6, 5))
sns.heatmap(df[['Age', 'FluidIntelligence', 'ComplexSpanScore']].corr(),
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png")
plt.close()

# 🧾 8. Save model comparison summary
summary = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['Accuracy'] for m in results],
    'MAE': [results[m]['MAE'] for m in results],
    'CV Mean Accuracy': [results[m]['CV Mean Accuracy'] for m in results],
    'CV Std': [results[m]['CV Std'] for m in results]
})
summary.to_csv("outputs/model_comparison.csv", index=False)
print("\nModel Comparison Summary:\n", summary)

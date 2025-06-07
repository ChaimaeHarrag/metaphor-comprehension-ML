# 🧠 Metaphor Comprehension ML Models

This project compares three machine learning models—Ordinal Logistic Regression (`mord`), Random Forest, and Gradient Boosting—to predict metaphor comprehension based on cognitive variables such as age, fluid intelligence, and working memory. It demonstrates how interpretable ML models can capture individual differences in figurative language understanding.

---

## 🔍 Objective

To investigate how cognitive factors predict metaphor comprehension using robust, interpretable machine learning techniques, with implications for cognitive aging and educational psycholinguistics.

---

## 💡 Why This Matters

Figurative language—particularly metaphor—engages intricate neurocognitive mechanisms, drawing on higher-order reasoning, memory, and executive control. This project advances our understanding of how individual differences in cognitive aging and executive function shape metaphor comprehension. By modeling these processes computationally, it lays a scalable foundation for future integration with EEG and fMRI data—offering valuable insights for psycholinguistics, cognitive aging research, and neuroscience.

---

## 🛠 Methods Overview

### Models Trained
- `mord.LogisticIT` (Ordinal Logistic Regression)
- `RandomForestClassifier`
- `GradientBoostingClassifier`

### Evaluation
- Accuracy, Mean Absolute Error (MAE)
- 5-fold Cross-Validation
- Confusion Matrices
- Feature Importances and Coefficients

### Preprocessing
- Standardization with `StandardScaler`
- Ordinal transformation of target variable

### Tools & Libraries
- `scikit-learn`, `mord`, `pandas`, `numpy`, `seaborn`, `matplotlib`

---

## 📁 Repository Contents

- `metaphor_comprehension_models.py` – full Python implementation and analysis pipeline
- `outputs/` – saved output files from model training and evaluation, including:
  - Confusion matrices (`confusion_matrix_*.png`)
  - Feature importance plots (`feature_importance_*.png`)
  - Logistic regression coefficients (`logistic_coefficients.png`)
  - Correlation matrix (`correlation_matrix.png`)
  - Classification reports (`classification_report_*.csv`)
  - Model comparison summary (`model_comparison.csv`)
- `outputs_ml_comparison.zip` – downloadable archive of all visualizations and reports

---

## 📊 Sample Output Snapshot

| Model             | Accuracy | MAE  | CV Mean Accuracy |
|------------------|----------|------|------------------|
| LogisticIT        | —        | —    | —                |
| Random Forest     | —        | —    | —                |
| Gradient Boosting | —        | —    | —                |

> *Run the script to generate and save real-time results based on your dataset.*

---

## 📦 Data Access

The dataset (`metaphor_comprehension.csv`) is not included in this repository due to participant privacy.  
📧 *The data is available upon request for academic use.*

---

## 👩‍🔬 Author

**Chaimae Harrag**  
PhD Candidate in Language and Cognition  
Fulbright and Erasmus Scholar | Researcher in Language, Memory, and Machine Learning

---

# 🧠 Metaphor Comprehension ML Models

This project compares three machine learning models—Ordinal Logistic Regression (`mord`), Random Forest, and Gradient Boosting—to predict metaphor comprehension based on cognitive variables such as age, fluid intelligence, and working memory.

## 🔍 Objective

To explore how individual differences in cognitive function influence figurative language processing using robust, interpretable machine learning models.

## 💡 Why This Matters

Figurative language, especially metaphor, taps into complex neurocognitive processes. This project contributes to the understanding of how cognitive aging and executive function shape metaphor comprehension. The work models these processes computationally, laying the groundwork for neurocognitive integration through EEG/fMRI—relevant to research in psycholinguistics, aging, and educational neuroscience.

## 🛠 Methods

- **Models Used**:  
  - `mord.LogisticIT` (Ordinal Logistic Regression)  
  - `RandomForestClassifier`  
  - `GradientBoostingClassifier`

- **Evaluation Metrics**:  
  Accuracy, Mean Absolute Error (MAE), Confusion Matrix, Cross-Validation

- **Feature Engineering**:  
  Standardization, Label Encoding

- **Visualization Tools**:  
  Seaborn heatmaps, Matplotlib bar plots (for feature importance)

## 📁 File Structure

- `metaphor_comprehension_models.py` – full implementation in Python
- `outputs/` – saved figures, model summaries, and classification reports
- `model_comparison.csv` – final summary table comparing model performance

## 💬 Sample Results

| Model            | Accuracy | MAE  | CV Mean Accuracy |
|------------------|----------|------|------------------|
| LogisticIT       | 0.78     | 0.41 | 0.76             |
| Random Forest    | 0.83     | 0.35 | 0.82             |
| Gradient Boosting| 0.85     | 0.33 | 0.84             |

## 🧪 Tools & Libraries

`scikit-learn`, `mord`, `pandas`, `numpy`, `seaborn`, `matplotlib`

## 👩‍🔬 Author

**Chaimae Harrag**  
PhD Candidate in Cognitive Science & Psycholinguistics  
Researching language comprehension, memory, and aging using machine learning and discourse analytics.  
Project developed in the context of applying for the [Ibn Sina Neurotech Autumn School (IBRO, 2025)](https://arabsinneuro.org) to bridge behavioral modeling with neuroimaging.

## 📦 Data Access

The dataset (`metaphor_comprehension.csv`) is available upon request.

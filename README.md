# Client Churm Modeling
 
Predicting bank customer churn using XGBoost. The goal is to identify customers at risk of leaving, enabling the bank to take proactive retention measures.
 
## Dataset
The dataset contains 10,000 bank customer records with the following features:
* credit_score
* geography
* gender
* age
* tenure
* balance
* num_of_products
* has_credit_card
* is_active_member
* estimated_salary

The target variable is `exited` (1 = churned, 0 = stayed).
 
The dataset has a moderate class imbalance: ~80% stayed, ~20% churned.
 
## Project Structure
```
churn-modeling/
│
├── Churn_Modeling.csv        # Raw dataset
└── Churn modeling.ipynb      # Main notebook
```
 
## Methodology
 
1. **EDA** — correlation analysis, violin plots, class distribution
2. **Baseline Model** — XGBoost classifier with default parameters inside a sklearn Pipeline
3. **Hyperparameter Optimization** — two-stage RandomizedSearchCV (tree structure → class balance)
4. **Feature Engineering Experiment** — added `balance_to_salary` ratio feature
 
## Results
 
| Model | ROC-AUC | Precision (Exited) | Recall (Exited) | F1 (Exited) | Accuracy |
|---|---|---|---|---|---|
| Baseline | 0.85 | 0.70 | 0.51 | 0.59 | 0.85 |
| Tuned | 0.88 | 0.59 | 0.71 | 0.64 | 0.84 |
| Feature-Engineered Baseline | 0.86 | 0.70 | 0.50 | 0.58 | 0.85 |
| Feature-Engineered Tuned | 0.88 | 0.49 | 0.82 | 0.61 | 0.79 |
 
The **tuned model** was selected as the final model based on the best balance between Precision and Recall.
 
## Key Findings
- `num_of_products`, `is_active_member`, and `age` are the top three predictors of churn
- Customers with 3+ products have a churn probability of 69-81%, while customers with 2 products have the lowest risk (~23%)
- Customers aged 45-60 are at the highest risk of churning
- Inactive members (`is_active_member = 0`) are significantly more likely to churn
- Customers with zero balance rarely churn; those with balance above 60,000 show higher churn risk
- Hyperparameter tuning improved Recall for churners from 0.51 to 0.71 with ROC-AUC increasing from 0.85 to 0.88
- The engineered feature `balance_to_salary` did not improve overall model performance
 
## Libraries
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — preprocessing, pipeline, model evaluation
- `xgboost` — gradient boosting classifier
- `seaborn`, `matplotlib` — visualization

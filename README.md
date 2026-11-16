# CatBoost Titanic Survival

Predicting who survived the Titanic with **CatBoost**, a gradient boosting algorithm that handles categorical data natively. Built for the CatBoost Activity Day project.

**Live app:** https://catboost-titanic-survival-fs.streamlit.app/

## What we did

1. **Explored the data** (891 passengers from Kaggle): who survived, and how that relates to sex, class, age, fare and port, plus missing values.
2. **Prepared the features:** extracted a passenger `Title` from the name, added family size, cabin and deck features, and dropped unhelpful columns. CatBoost takes the categorical columns directly, with no one-hot encoding.
3. **Split the data** into train (569), validation (143) and test (179) with a fixed seed. Validation is used for early stopping, and the test set is used once at the end.
4. **Trained `CatBoostClassifier`** and evaluated it with accuracy, precision, recall, F1, ROC-AUC, a confusion matrix and an ROC curve.
5. **Explained the model** with feature importance and SHAP.
6. **Tuned it with Optuna** (25 trials, 5-fold cross-validation).
7. **Compared it fairly** against Random Forest, XGBoost and LightGBM using the same split and folds.
8. **Built a Streamlit app** where you enter a passenger and get a survival probability with an explanation.

## Results

Held-out test set (179 passengers):

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| CatBoost (default) | 0.793 | 0.704 | 0.839 |
| CatBoost (tuned with Optuna) | 0.782 | 0.719 | 0.836 |
| Random Forest (tuned) | 0.804 | 0.720 | 0.845 |

In repeated 5-fold cross-validation, CatBoost and Random Forest tie at about 0.876 ROC-AUC, with XGBoost at 0.872 and LightGBM at 0.868.

**Takeaways**
- Sex, title and passenger class drive the predictions, which matches the "women and children first" policy.
- On a dataset this small, no model clearly wins. The differences are within noise.
- CatBoost's practical advantage is less preprocessing: raw categories and missing ages work as they are.

## Project structure

```
catboost-titanic-survival/
├── CatBoost_Titanic.ipynb   # full analysis (Colab or VS Code)
├── train.csv, test.csv, gender_submission.csv   # Kaggle data
├── submission.csv           # Kaggle-format predictions
└── streamlit_app/
    ├── app.py
    ├── requirements.txt
    └── train.csv
```

## Limitations

- Small, historical dataset (891 labeled passengers from 1912), not representative of other populations.
- Feature importance and SHAP show what the model uses to predict, not what caused survival.
- Fare, cabin and title reflect social class, so a model like this should not be used for real decisions about people.

## References

- [Kaggle: Titanic](https://www.kaggle.com/competitions/titanic)
- [CatBoost documentation](https://catboost.ai/en/docs/)
- Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features*, NeurIPS 2018
- [Optuna](https://optuna.org)
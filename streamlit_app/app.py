"""Titanic Survival Predictor — CatBoost + Streamlit.

Trains CatBoostClassifier on train.csv at startup (cached) using the same features,
split (569/143/179, seed 42) and settings as the CatBoost_Titanic notebook.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve)

SEED = 42
CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin"]
FEATURES = CAT_COLS + NUM_COLS
RARE_TITLES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
               "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare"}

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    # Row-wise feature engineering (identical to the notebook) — no statistics from other rows, so no leakage.
    d = data.copy()
    d["Title"] = d["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip().replace(RARE_TITLES)
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    d["HasCabin"] = d["Cabin"].notna().astype(int)
    d["Deck"] = d["Cabin"].str[0].fillna("Unknown")
    d["Embarked"] = d["Embarked"].fillna("Missing")
    d["Pclass"] = d["Pclass"].astype(str)
    return d


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("train.csv")


@st.cache_resource(show_spinner="Training CatBoost on the Titanic data…")
def train_model():
    df = load_data()
    full = add_features(df)
    X, y = full[FEATURES], full["Survived"]
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tv, y_tv, test_size=0.20, stratify=y_tv, random_state=SEED)
    model = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=4, loss_function="Logloss", eval_metric="AUC",
                               l2_leaf_reg=3, random_seed=SEED, early_stopping_rounds=100, verbose=0)
    model.fit(X_tr, y_tr, cat_features=CAT_COLS, eval_set=(X_val, y_val), use_best_model=True)
    return model, X_test, y_test


df = load_data()
model, X_test, y_test = train_model()
test_proba = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🚢 Titanic Survival")
st.sidebar.caption("CatBoostClassifier · Activity Day project")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Predict 'Survived' when the probability is at least this value.")
st.sidebar.markdown("---")
st.sidebar.caption("Educational demo on 1912 data. Predictions show learned statistical patterns, "
                   "not causal effects, and must not be used for real decisions about people.")

st.title("Titanic Survival Predictor")
tab_pred, tab_perf, tab_data, tab_about = st.tabs(["🔮 Predict", "📊 Model performance", "🔎 Data explorer", "ℹ️ About"])

# ------------------------------------------------------------------ predict
with tab_pred:
    st.subheader("Describe a passenger")
    c1, c2, c3 = st.columns(3)
    with c1:
        sex = st.selectbox("Sex", ["female", "male"])
        pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
        title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"],
                             index=1 if sex == "female" else 0,
                             help="'Rare' groups titles such as Dr, Rev, Col, Countess.")
    with c2:
        age_known = st.checkbox("Age known", value=True)
        age = st.slider("Age", 0, 80, 30, disabled=not age_known)
        sibsp = st.number_input("Siblings / spouses aboard", 0, 8, 0)
        parch = st.number_input("Parents / children aboard", 0, 6, 0)
    with c3:
        fare = st.number_input("Fare (£)", 0.0, 520.0, 32.0, step=1.0)
        embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"],
                                format_func=lambda p: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[p])
        cabin_known = st.checkbox("Cabin recorded", value=False)
        deck = st.selectbox("Deck", list("ABCDEFG"), index=2, disabled=not cabin_known)

    row = pd.DataFrame([{
        "Pclass": str(pclass), "Sex": sex, "Embarked": embarked, "Title": title,
        "Deck": deck if cabin_known else "Unknown",
        "Age": float(age) if age_known else np.nan, "SibSp": sibsp, "Parch": parch, "Fare": fare,
        "FamilySize": sibsp + parch + 1, "IsAlone": int(sibsp + parch == 0), "HasCabin": int(cabin_known)}])[FEATURES]

    proba = float(model.predict_proba(row)[0, 1])
    verdict = "Survived" if proba >= threshold else "Did not survive"
    m1, m2 = st.columns(2)
    m1.metric("Probability of survival", f"{proba:.1%}")
    m2.metric(f"Prediction (threshold {threshold:.2f})", verdict)
    st.progress(min(max(proba, 0.0), 1.0))

    st.markdown("**Why did the model say this?** SHAP contributions (log-odds): blue pushes toward survival, red toward death.")
    sv = model.get_feature_importance(Pool(row, cat_features=CAT_COLS), type="ShapValues")[0][:-1]
    contrib = pd.Series(sv, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(contrib.index, contrib.values, color=["#c44e52" if v < 0 else "#4c72b0" for v in contrib.values])
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("SHAP value (log-odds)")
    st.pyplot(fig); plt.close(fig)
    st.caption("Contributions describe what the model uses, not what caused survival.")

# ------------------------------------------------------------------ performance
with tab_perf:
    st.subheader(f"Held-out test set ({len(y_test)} passengers) at threshold {threshold:.2f}")
    pred = (test_proba >= threshold).astype(int)
    cols = st.columns(5)
    for col, (name, val) in zip(cols, [("Accuracy", accuracy_score(y_test, pred)),
                                       ("Precision", precision_score(y_test, pred, zero_division=0)),
                                       ("Recall", recall_score(y_test, pred)), ("F1", f1_score(y_test, pred)),
                                       ("ROC-AUC", roc_auc_score(y_test, test_proba))]):
        col.metric(name, f"{val:.3f}")
    st.caption("Metrics are computed live from the held-out test split (never used for training or early stopping). "
               "Precision: of predicted survivors, how many survived. Recall: of real survivors, how many were found. "
               "ROC-AUC does not depend on the threshold.")
    a, b, c = st.columns(3)
    with a:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=[0, 1]), display_labels=["Died", "Survived"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion matrix"); st.pyplot(fig); plt.close(fig)
    with b:
        fpr, tpr, _ = roc_curve(y_test, test_proba)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_test, test_proba):.3f}"); ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curve"); ax.legend()
        st.pyplot(fig); plt.close(fig)
    with c:
        imp = pd.Series(model.get_feature_importance(Pool(X_test, y_test, cat_features=CAT_COLS)), index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(4.5, 4))
        imp.plot.barh(ax=ax, color="#4c72b0"); ax.set_title("Feature importance"); st.pyplot(fig); plt.close(fig)
    st.info(f"Early stopping kept {model.tree_count_} trees. In the notebook, a tuned Random Forest, XGBoost and LightGBM "
            "scored within about one standard deviation of CatBoost in repeated cross-validation.")

# ------------------------------------------------------------------ data explorer
with tab_data:
    st.subheader("Training data (891 labeled passengers)")
    st.dataframe(df.head(20), width="stretch")
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df, errorbar=None, ax=ax); ax.set_title("Survival rate by class and sex")
        st.pyplot(fig); plt.close(fig)
    with b:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True, ax=ax); ax.set_title("Age distribution by outcome")
        st.pyplot(fig); plt.close(fig)
    miss = (df.isna().mean() * 100).round(1)
    st.write("**Missing values (% per column)**")
    st.bar_chart(miss[miss > 0])

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**CatBoost** (Categorical Boosting) is a gradient boosting library that builds shallow symmetric trees one after another.
Its key ideas: *ordered boosting* (reduces prediction shift), *ordered target statistics* for categorical features
(no one-hot encoding needed) and native missing-value handling.

**This app:** the model is trained when the app starts, on the same features and the same stratified 569 / 143 / 179
train / validation / test split (seed 42) as the notebook. Validation data drives early stopping; the test set is used only
for the metrics shown.

**Limitations:** 891 labeled passengers from 1912 — small, historical and not representative of other populations.
Importance and SHAP show predictive association, **not causation**. Features such as fare, cabin and title encode
socio-economic status, so real-world use would raise fairness concerns.
""")

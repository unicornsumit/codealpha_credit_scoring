"""
TASK 1: Credit Scoring Model
=============================
Objective : Predict an individual's creditworthiness using past financial data.
Approach  : Classification algorithms - Logistic Regression, Decision Tree, Random Forest.

Dataset: UCI Statlog (German Credit Data) - 1000 applicants, 20 attributes
covering income proxies, debts, payment/credit history, employment, etc.
Target: 1 = Good credit risk, 0 = Bad credit risk (after remapping from 1/2).

The script:
    1. Loads the raw UCI data and decodes the coded categorical attributes
       into human-readable values.
    2. Engineers a few extra features from the existing financial history
       (e.g. credit amount per month of duration, age bucket).
    3. One-hot encodes categoricals + scales numeric features.
    4. Trains Logistic Regression, Decision Tree, and Random Forest.
    5. Evaluates with Accuracy, Precision, Recall, F1-score, ROC-AUC
       (Recall on the "bad credit" class matters most for a lender).
    6. Plots confusion matrices, ROC curves, a model-comparison chart,
       and feature importance (Random Forest).
    7. Saves the best model + preprocessing pipeline to disk.

Run:
    python credit_scoring.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sns.set_style("whitegrid")
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. Load + decode the UCI German Credit dataset
# ----------------------------------------------------------------------
def load_credit_data():
    cols = ["checking_account", "duration_months", "credit_history", "purpose", "credit_amount",
            "savings_account", "employment_since", "installment_rate_pct", "personal_status_sex",
            "other_debtors", "present_residence_since", "property", "age", "other_installment_plans",
            "housing", "existing_credits", "job", "num_dependents", "telephone", "foreign_worker",
            "target"]
    df = pd.read_csv(os.path.join(DATA_DIR, "german_credit.csv"), names=cols)

    # --- Decode UCI attribute codes into readable categories ---
    maps = {
        "checking_account": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no account"},
        "credit_history": {
            "A30": "no credits/all paid", "A31": "all credits paid (this bank)",
            "A32": "existing credits paid duly", "A33": "delay in past",
            "A34": "critical/other credits existing"
        },
        "purpose": {
            "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
            "A43": "radio/TV", "A44": "domestic appliances", "A45": "repairs",
            "A46": "education", "A47": "vacation", "A48": "retraining",
            "A49": "business", "A410": "other"
        },
        "savings_account": {
            "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
            "A64": ">= 1000 DM", "A65": "unknown/none"
        },
        "employment_since": {
            "A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years",
            "A74": "4-7 years", "A75": ">= 7 years"
        },
        "personal_status_sex": {
            "A91": "male:divorced/separated", "A92": "female:divorced/separated/married",
            "A93": "male:single", "A94": "male:married/widowed", "A95": "female:single"
        },
        "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
        "property": {
            "A121": "real estate", "A122": "building society/life insurance",
            "A123": "car/other", "A124": "unknown/none"
        },
        "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
        "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
        "job": {
            "A171": "unemployed/unskilled-nonresident", "A172": "unskilled-resident",
            "A173": "skilled employee", "A174": "management/self-employed/highly-qualified"
        },
        "telephone": {"A191": "none", "A192": "yes"},
        "foreign_worker": {"A201": "yes", "A202": "no"},
    }
    for col, mapping in maps.items():
        df[col] = df[col].map(mapping)

    # Target: UCI uses 1 = good, 2 = bad -> remap to 1 = good, 0 = bad
    df["target"] = df["target"].map({1: 1, 2: 0})

    return df


# ----------------------------------------------------------------------
# 2. Feature engineering from financial history
# ----------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Monthly credit burden: how much credit per month of repayment
    df["credit_per_month"] = df["credit_amount"] / df["duration_months"]

    # Debt-to-installment proxy: credit amount weighted by installment rate (% of income)
    df["debt_burden_score"] = df["credit_amount"] * df["installment_rate_pct"] / 100

    # Age bucket: younger applicants are statistically higher risk in this dataset
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 100],
        labels=["<=25", "26-35", "36-45", "46-55", "56+"]
    ).astype(str)

    # Long-duration high-amount loans combined risk flag
    df["is_long_high_amount"] = (
        (df["duration_months"] > df["duration_months"].median()) &
        (df["credit_amount"] > df["credit_amount"].median())
    ).astype(int)

    # Stability proxy: has a checking + savings account
    df["has_both_accounts"] = (
        (df["checking_account"] != "no account") &
        (df["savings_account"] != "unknown/none")
    ).astype(int)

    return df


# ----------------------------------------------------------------------
# 3. Build preprocessing + models
# ----------------------------------------------------------------------
NUMERIC_FEATURES = [
    "duration_months", "credit_amount", "installment_rate_pct",
    "present_residence_since", "age", "existing_credits", "num_dependents",
    "credit_per_month", "debt_burden_score", "is_long_high_amount", "has_both_accounts"
]
CATEGORICAL_FEATURES = [
    "checking_account", "credit_history", "purpose", "savings_account", "employment_since",
    "personal_status_sex", "other_debtors", "property", "other_installment_plans",
    "housing", "job", "telephone", "foreign_worker", "age_group"
]


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


# ----------------------------------------------------------------------
# 4. Main pipeline
# ----------------------------------------------------------------------
def main():
    print("Loading and preparing German Credit dataset...")
    df = load_credit_data()
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["target"]

    print(f"Samples: {X.shape[0]}   Features (pre-encoding): {X.shape[1]}")
    print(f"Class balance -> Good credit: {(y==1).sum()}  Bad credit: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()

    results = []
    fitted_pipelines = {}

    for name, model in get_models().items():
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
        })
        fitted_pipelines[name] = pipe

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, digits=3, target_names=["Bad Credit", "Good Credit"]))

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    print("\nSummary:\n", results_df.round(3).to_string(index=False))
    results_df.to_csv(os.path.join(OUT_DIR, "model_results.csv"), index=False)

    # --- Confusion matrices ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    for ax, (name, pipe) in zip(axes, fitted_pipelines.items()):
        cm = confusion_matrix(y_test, pipe.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()

    # --- ROC curves ---
    plt.figure(figsize=(7, 6))
    for name, pipe in fitted_pipelines.items():
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    # --- Model comparison bar chart ---
    plt.figure(figsize=(8, 5))
    plot_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
    plt.title("Model Comparison")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # --- Feature importance (Random Forest) ---
    rf_pipe = fitted_pipelines["Random Forest"]
    feature_names = rf_pipe.named_steps["prep"].get_feature_names_out()
    importances = rf_pipe.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df, y="feature", x="importance", color="steelblue")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    imp_df.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)

    # --- Save best model (by ROC-AUC) ---
    best_name = results_df.iloc[0]["Model"]
    joblib.dump(fitted_pipelines[best_name], os.path.join(MODEL_DIR, "best_credit_model.pkl"))
    print(f"\nBest model: {best_name} (saved to models/best_credit_model.pkl)")
    print(f"All plots and CSVs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD PREPARED DATA
# ============================================================

X_train = pd.read_csv("X_train_processed.csv")
X_test = pd.read_csv("X_test_processed.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()


print("\n========== LOAD DATA ==========")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 2. CREATE BASELINE MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ============================================================
# 3. TRAIN MODEL
# ============================================================

print("\n========== TRAINING ==========")

model.fit(
    X_train,
    y_train
)

print("Logistic Regression training completed.")


# ============================================================
# 4. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 5. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 6. DISPLAY METRICS
# ============================================================

print("\n========== MODEL PERFORMANCE ==========")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n========== CONFUSION MATRIX ==========")

print(cm)


print("\nInterpretation:")

print(
    "True Negatives :",
    cm[0, 0]
)

print(
    "False Positives:",
    cm[0, 1]
)

print(
    "False Negatives:",
    cm[1, 0]
)

print(
    "True Positives  :",
    cm[1, 1]
)


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "On Time",
            "Late"
        ]
    )
)


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

feature_names = pd.read_csv(
    "ml_feature_names.csv"
)["feature_name"]


coefficients = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": model.coef_[0]
    }
)


coefficients["absolute_coefficient"] = (
    coefficients["coefficient"].abs()
)


coefficients = coefficients.sort_values(
    "absolute_coefficient",
    ascending=False
)


print("\n========== TOP FEATURES ==========")

print(
    coefficients[
        [
            "feature",
            "coefficient"
        ]
    ].head(20)
)


# ============================================================
# 10. SAVE MODEL RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "model": ["Logistic Regression"],
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1],
        "roc_auc": [roc_auc]
    }
)


results.to_csv(
    "model_results.csv",
    index=False
)


coefficients.to_csv(
    "logistic_regression_feature_importance.csv",
    index=False
)


# ============================================================
# 11. SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame(
    {
        "actual": y_test,
        "predicted": y_pred,
        "late_probability": y_probability
    }
)


predictions.to_csv(
    "logistic_regression_predictions.csv",
    index=False
)


print("\n==============================================")
print("BASELINE MODEL COMPLETED")
print("==============================================")

print("\nFiles created:")
print("model_results.csv")
print("logistic_regression_feature_importance.csv")
print("logistic_regression_predictions.csv")
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# 1. LOAD MODEL PREDICTIONS
# ============================================================

logistic = pd.read_csv("logistic_regression_predictions.csv")
random_forest = pd.read_csv("random_forest_predictions.csv")

print("Logistic Regression columns:")
print(logistic.columns.tolist())

print("\nRandom Forest columns:")
print(random_forest.columns.tolist())


# ============================================================
# 2. GET ACTUAL VALUES AND PROBABILITIES
# ============================================================

logistic_actual = logistic["actual"]
logistic_probability = logistic["late_probability"]

rf_actual = random_forest["actual"]
rf_probability = random_forest["probability_late"]


# ============================================================
# 3. ROC-AUC
# ============================================================

logistic_auc = roc_auc_score(
    logistic_actual,
    logistic_probability
)

rf_auc = roc_auc_score(
    rf_actual,
    rf_probability
)

print("\n================ ROC-AUC ================")

print(
    f"Logistic Regression ROC-AUC: "
    f"{logistic_auc:.4f}"
)

print(
    f"Random Forest ROC-AUC: "
    f"{rf_auc:.4f}"
)


# ============================================================
# 4. ROC CURVE
# ============================================================

logistic_fpr, logistic_tpr, _ = roc_curve(
    logistic_actual,
    logistic_probability
)

rf_fpr, rf_tpr, _ = roc_curve(
    rf_actual,
    rf_probability
)


plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression (AUC = {logistic_auc:.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Guessing"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Model Comparison")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "ROC_Curve_Model_Comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 5. PRECISION-RECALL CURVE
# ============================================================

logistic_precision, logistic_recall, _ = precision_recall_curve(
    logistic_actual,
    logistic_probability
)

rf_precision, rf_recall, _ = precision_recall_curve(
    rf_actual,
    rf_probability
)


logistic_ap = average_precision_score(
    logistic_actual,
    logistic_probability
)

rf_ap = average_precision_score(
    rf_actual,
    rf_probability
)


plt.figure(figsize=(8, 6))

plt.plot(
    logistic_recall,
    logistic_precision,
    label=f"Logistic Regression (AP = {logistic_ap:.3f})"
)

plt.plot(
    rf_recall,
    rf_precision,
    label=f"Random Forest (AP = {rf_ap:.3f})"
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve - Model Comparison")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "Precision_Recall_Curve_Model_Comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. SAVE EVALUATION RESULTS
# ============================================================

evaluation_results = pd.DataFrame({

    "model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "roc_auc": [
        logistic_auc,
        rf_auc
    ],

    "average_precision": [
        logistic_ap,
        rf_ap
    ]
})


evaluation_results.to_csv(
    "model_evaluation_results.csv",
    index=False
)


# ============================================================
# 7. FINAL OUTPUT
# ============================================================

print("\n================ MODEL EVALUATION ================")

print(evaluation_results)

print("\nFiles created:")

print("1. ROC_Curve_Model_Comparison.png")
print("2. Precision_Recall_Curve_Model_Comparison.png")
print("3. model_evaluation_results.csv")

print("\nModel evaluation completed successfully.")
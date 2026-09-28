import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. Load processed ML data
# --------------------------------------------------

X_train = pd.read_csv("X_train_processed.csv")
X_test = pd.read_csv("X_test_processed.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# --------------------------------------------------
# 2. Train Logistic Regression baseline
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 3. Get probability of late order
# --------------------------------------------------

y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 4. Test different thresholds
# --------------------------------------------------

thresholds = np.arange(0.20, 0.81, 0.05)

results = []

for threshold in thresholds:

    y_pred = (y_probability >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "threshold": round(threshold, 2),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    })


# --------------------------------------------------
# 5. Create results table
# --------------------------------------------------

threshold_results = pd.DataFrame(results)

print("\n========== THRESHOLD ANALYSIS ==========")

print(
    threshold_results.to_string(index=False)
)


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

threshold_results.to_csv(
    "threshold_analysis.csv",
    index=False
)


# --------------------------------------------------
# 7. Find threshold with highest F1
# --------------------------------------------------

best_f1_row = threshold_results.loc[
    threshold_results["f1_score"].idxmax()
]

print("\n========== BEST F1 THRESHOLD ==========")

print(
    f"Threshold : {best_f1_row['threshold']:.2f}"
)

print(
    f"Accuracy  : {best_f1_row['accuracy']:.4f}"
)

print(
    f"Precision : {best_f1_row['precision']:.4f}"
)

print(
    f"Recall    : {best_f1_row['recall']:.4f}"
)

print(
    f"F1 Score  : {best_f1_row['f1_score']:.4f}"
)


# --------------------------------------------------
# 8. Find threshold with recall >= 50%
# --------------------------------------------------

recall_target = threshold_results[
    threshold_results["recall"] >= 0.50
]

if not recall_target.empty:

    # Among thresholds reaching 50% recall,
    # select the one with the highest precision.

    best_recall_balance = recall_target.loc[
        recall_target["precision"].idxmax()
    ]

    print("\n========== BEST THRESHOLD WITH RECALL >= 50% ==========")

    print(
        f"Threshold : {best_recall_balance['threshold']:.2f}"
    )

    print(
        f"Accuracy  : {best_recall_balance['accuracy']:.4f}"
    )

    print(
        f"Precision : {best_recall_balance['precision']:.4f}"
    )

    print(
        f"Recall    : {best_recall_balance['recall']:.4f}"
    )

    print(
        f"F1 Score  : {best_recall_balance['f1_score']:.4f}"
    )

else:

    print(
        "\nNo tested threshold achieved 50% recall."
    )


print("\nThreshold analysis completed successfully.")
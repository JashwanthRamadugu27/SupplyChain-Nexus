# ============================================================
# STEP 15: SHAP EXPLAINABILITY
# SupplyChain Nexus
# ============================================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. LOAD PROCESSED ML DATA
# ============================================================

print("\nLoading processed ML data...")

X_train = pd.read_csv("X_train_processed.csv")
X_test = pd.read_csv("X_test_processed.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

feature_names_df = pd.read_csv("ml_feature_names.csv")

# Get feature names safely
if "feature" in feature_names_df.columns:
    feature_names = feature_names_df["feature"].tolist()
else:
    feature_names = feature_names_df.iloc[:, 0].tolist()

# Make sure feature names match the processed data
X_train.columns = feature_names
X_test.columns = feature_names

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)
print("Features     :", len(feature_names))


# ============================================================
# 2. TRAIN THE FINAL LOGISTIC REGRESSION MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# 3. CREATE A SAMPLE FOR SHAP
# ============================================================

# We don't need to explain all 22,856 test rows.
# A representative sample is enough for global explainability.

sample_size = min(1000, len(X_test))

X_test_sample = X_test.sample(
    n=sample_size,
    random_state=42
)

print("\nSHAP sample size:", len(X_test_sample))


# ============================================================
# 4. CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explainer...")

explainer = shap.Explainer(
    model,
    X_train.sample(
        n=min(500, len(X_train)),
        random_state=42
    )
)

shap_values = explainer(X_test_sample)

print("SHAP values calculated.")


# ============================================================
# 5. HANDLE SHAP OUTPUT
# ============================================================

# For binary classification, SHAP versions may return
# slightly different output shapes.
#
# We want the contribution toward the "Late" class (class 1).

values = shap_values.values

if values.ndim == 3:

    # Shape can be:
    # (samples, features, classes)

    if values.shape[2] == 2:
        values = values[:, :, 1]
    else:
        values = values[:, :, 0]

elif values.ndim != 2:

    raise ValueError(
        f"Unexpected SHAP value shape: {values.shape}"
    )


# ============================================================
# 6. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = np.abs(values).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap_value": mean_abs_shap
})

shap_importance = shap_importance.sort_values(
    by="mean_absolute_shap_value",
    ascending=False
).reset_index(drop=True)


# Save feature importance

shap_importance.to_csv(
    "shap_feature_importance.csv",
    index=False
)

print("\nTop SHAP features:")

print(
    shap_importance.head(20).to_string(index=False)
)


# ============================================================
# 7. SHAP SUMMARY PLOT
# ============================================================

print("\nCreating SHAP summary plot...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    values,
    X_test_sample,
    feature_names=feature_names,
    show=False,
    max_display=20
)

plt.title(
    "SHAP Summary Plot - Logistic Regression",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "SHAP_Summary_Plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: SHAP_Summary_Plot.png")


# ============================================================
# 8. SHAP BAR PLOT
# ============================================================

print("\nCreating SHAP bar plot...")

plt.figure(figsize=(10, 8))

shap.summary_plot(
    values,
    X_test_sample,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    max_display=20
)

plt.title(
    "SHAP Feature Importance - Logistic Regression",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "SHAP_Bar_Plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: SHAP_Bar_Plot.png")


# ============================================================
# 9. CREATE INDIVIDUAL PREDICTION EXAMPLES
# ============================================================

print("\nCreating individual prediction explanations...")

# Select a few examples
example_count = min(10, len(X_test_sample))

example_data = X_test_sample.iloc[:example_count]

example_shap_values = values[:example_count]

example_probabilities = model.predict_proba(
    example_data
)[:, 1]

example_actual = y_test.loc[
    example_data.index
].values


rows = []

for i in range(example_count):

    row_values = example_shap_values[i]

    # Most positive contributors
    positive_indices = np.argsort(row_values)[-3:][::-1]

    # Most negative contributors
    negative_indices = np.argsort(row_values)[:3]

    positive_features = [
        feature_names[j]
        for j in positive_indices
    ]

    negative_features = [
        feature_names[j]
        for j in negative_indices
    ]

    rows.append({
        "test_row": int(example_data.index[i]),
        "actual_late_order": int(example_actual[i]),
        "predicted_late_probability": float(
            example_probabilities[i]
        ),
        "top_positive_features": " | ".join(
            positive_features
        ),
        "top_negative_features": " | ".join(
            negative_features
        )
    })


prediction_examples = pd.DataFrame(rows)

prediction_examples.to_csv(
    "shap_prediction_examples.csv",
    index=False
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n================ SHAP SUMMARY ================")

print("Model              : Logistic Regression")
print("SHAP samples       :", sample_size)
print("Features explained :", len(feature_names))

print("\nTop 10 influential features:")

print(
    shap_importance.head(10).to_string(index=False)
)

print("\nFiles created:")
print("1. shap_feature_importance.csv")
print("2. SHAP_Summary_Plot.png")
print("3. SHAP_Bar_Plot.png")
print("4. shap_prediction_examples.csv")

print("\n================================================")
print("STEP 15 - SHAP EXPLAINABILITY COMPLETED")
print("================================================")
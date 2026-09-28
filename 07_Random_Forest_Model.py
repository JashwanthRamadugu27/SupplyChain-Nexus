import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
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
# 2. Create Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 3. Train model
# --------------------------------------------------

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 4. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 5. Evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== RANDOM FOREST RESULTS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# --------------------------------------------------
# 6. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 7. Classification Report
# --------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 8. Feature Importance
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 20 Important Features:")
print(feature_importance.head(20))


# --------------------------------------------------
# 9. Save Random Forest feature importance
# --------------------------------------------------

feature_importance.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)


# --------------------------------------------------
# 10. Save predictions
# --------------------------------------------------

predictions = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred,
    "probability_late": y_prob
})

predictions.to_csv(
    "random_forest_predictions.csv",
    index=False
)


# --------------------------------------------------
# 11. Save model results
# --------------------------------------------------

new_result = pd.DataFrame({
    "model": ["Random Forest"],
    "accuracy": [accuracy],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1],
    "roc_auc": [roc_auc]
})

try:
    results = pd.read_csv("model_results.csv")

    # Remove any previous Random Forest result
    results = results[results["model"] != "Random Forest"]

    # Add the latest Random Forest result
    results = pd.concat([results, new_result], ignore_index=True)

except FileNotFoundError:
    results = new_result

results.to_csv(
    "model_results.csv",
    index=False
)

print("\nUpdated model_results.csv:")
print(results)

print("\nRandom Forest model completed successfully.")
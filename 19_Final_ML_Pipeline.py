import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "feature_engineered_supply_chain_data.csv"
)

print("Dataset:", df.shape)


# ============================================================
# 2. REMOVE UNUSED / LEAKAGE-PRONE FEATURES
# ============================================================

columns_to_remove = [
    "order_id",
    "product_id",
    "hub_order_volume",
    "customer_order_volume",
    "three_pl_order_volume",
    "late_order"
]

X = df.drop(
    columns=columns_to_remove
)

y = df["late_order"].astype(int)


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

categorical_features = [
    "origin_port",
    "3pl",
    "customs_procedures",
    "logistic_hub",
    "customer",
    "material_handling",
    "weight_class",
    "distance_category",
    "shipment_size"
]

numerical_features = [
    "units",
    "weight",
    "distance",
    "cost_per_unit",
    "co2_per_unit",
    "route_data_available",
    "direct_shipment",
    "estimated_route_cost",
    "estimated_co2",
    "weight_per_unit",
    "total_weight",
    "log_units",
    "log_distance",
    "log_total_weight",
    "product_data_missing",
    "route_data_missing"
]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# 5. NUMERICAL PIPELINE
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 6. CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# 7. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ============================================================
# 9. COMPLETE ML PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 10. TRAIN
# ============================================================

print("\nTraining final pipeline...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 11. PREDICTIONS
# ============================================================

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]

y_prediction = (
    y_probability >= 0.50
).astype(int)


# ============================================================
# 12. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)

precision = precision_score(
    y_test,
    y_prediction
)

recall = recall_score(
    y_test,
    y_prediction
)

f1 = f1_score(
    y_test,
    y_prediction
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n========== FINAL MODEL RESULTS ==========")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# 13. SAVE COMPLETE PIPELINE
# ============================================================

joblib.dump(
    pipeline,
    "supply_chain_ml_pipeline.pkl"
)

print(
    "\nSaved as: supply_chain_ml_pipeline.pkl"
)


# ============================================================
# 14. SAVE TEST DATA FOR REFERENCE
# ============================================================

X_test.to_csv(
    "final_X_test.csv",
    index=False
)

y_test.to_csv(
    "final_y_test.csv",
    index=False
)

print(
    "Test data saved."
)

print(
    "\n========================================"
)

print(
    "FINAL ML PIPELINE COMPLETED"
)

print(
    "========================================"
)
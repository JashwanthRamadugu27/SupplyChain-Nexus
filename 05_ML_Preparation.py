import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATA
# ============================================================

df = pd.read_csv("feature_engineered_supply_chain_data.csv")

print("\n========== LOAD DATA ==========")
print("Shape:", df.shape)


# ============================================================
# 2. REMOVE FEATURES NOT USED DIRECTLY FOR ML
# ============================================================

# order_id is only an identifier
# product_id is not useful as a numeric value
# derived volume features are removed because they were
# calculated before the train/test split and could cause leakage

columns_to_remove = [
    "order_id",
    "product_id",
    "hub_order_volume",
    "customer_order_volume",
    "three_pl_order_volume"
]

df = df.drop(
    columns=columns_to_remove
)


# ============================================================
# 3. DEFINE TARGET
# ============================================================

X = df.drop(
    columns=["late_order"]
)

y = df["late_order"].astype(int)


print("\n========== TARGET ==========")

print(y.value_counts())

print("\nTarget percentage:")
print(y.value_counts(normalize=True) * 100)


# ============================================================
# 4. IDENTIFY FEATURE TYPES
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


print("\n========== FEATURE TYPES ==========")

print("Categorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n========== TRAIN / TEST SPLIT ==========")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True) * 100)


# ============================================================
# 6. NUMERICAL PREPROCESSING
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
# 7. CATEGORICAL PREPROCESSING
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
# 8. COMBINE PREPROCESSING
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
# 9. FIT ONLY ON TRAINING DATA
# ============================================================

print("\n========== FITTING PREPROCESSOR ==========")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# 10. GET FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


print("\n========== PROCESSED DATA ==========")

print(
    "Original training features:",
    X_train.shape[1]
)

print(
    "Processed training features:",
    X_train_processed.shape[1]
)

print(
    "Processed testing features:",
    X_test_processed.shape[1]
)


# ============================================================
# 11. CONVERT TO DATAFRAMES
# ============================================================

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)


# ============================================================
# 12. SAVE PREPARED DATA
# ============================================================

X_train_processed.to_csv(
    "X_train_processed.csv",
    index=False
)

X_test_processed.to_csv(
    "X_test_processed.csv",
    index=False
)

y_train.to_csv(
    "y_train.csv",
    index=False
)

y_test.to_csv(
    "y_test.csv",
    index=False
)


# ============================================================
# 13. SAVE FEATURE NAMES
# ============================================================

pd.DataFrame(
    {
        "feature_name": feature_names
    }
).to_csv(
    "ml_feature_names.csv",
    index=False
)


# ============================================================
# 14. FINAL CHECK
# ============================================================

print("\n========== FINAL CHECK ==========")

print(
    "Training missing values:",
    X_train_processed.isnull().sum().sum()
)

print(
    "Testing missing values:",
    X_test_processed.isnull().sum().sum()
)

print(
    "Training rows:",
    len(X_train_processed)
)

print(
    "Testing rows:",
    len(X_test_processed)
)


print("\n==============================================")
print("ML PREPARATION COMPLETED")
print("==============================================")

print("\nFiles created:")

print("X_train_processed.csv")
print("X_test_processed.csv")
print("y_train.csv")
print("y_test.csv")
print("ml_feature_names.csv")
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


print("Starting preprocessing...")


# Load data
df = pd.read_csv("feature_engineered_supply_chain_data.csv")

print("Data loaded:", df.shape)


# Remove unused columns
columns_to_remove = [
    "order_id",
    "product_id",
    "hub_order_volume",
    "customer_order_volume",
    "three_pl_order_volume",
    "late_order"
]

X = df.drop(columns=columns_to_remove)


# Features
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


# Numerical preprocessing
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# Complete preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# Fit
preprocessor.fit(X)

print("Preprocessing pipeline fitted.")
print("Input features:", X.shape[1])


# Check processed features
processed = preprocessor.transform(X)

print("Processed features:", processed.shape[1])


# Save
joblib.dump(
    preprocessor,
    "supply_chain_preprocessor.pkl"
)

print("Preprocessor saved successfully.")
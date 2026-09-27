import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

df = pd.read_csv("processed_supply_chain_data.csv")

print("\n========== LOAD DATA ==========")

print("Shape:")
print(df.shape)


# ============================================================
# 2. CREATE ROUTE DATA AVAILABILITY FEATURE
# ============================================================

df["route_data_available"] = (
    df["distance"].notna()
).astype(int)


print("\n========== ROUTE DATA ==========")

print(
    df["route_data_available"]
    .value_counts()
    .rename(
        index={
            0: "Route data unavailable",
            1: "Route data available"
        }
    )
)


# ============================================================
# 3. CREATE DIRECT SHIPMENT FEATURE
# ============================================================

df["direct_shipment"] = (
    df["logistic_hub"].isna()
).astype(int)


print("\n========== DIRECT SHIPMENTS ==========")

print(
    df["direct_shipment"]
    .value_counts()
    .rename(
        index={
            0: "Uses logistic hub",
            1: "Direct shipment"
        }
    )
)


# ============================================================
# 4. CREATE ROUTE COST FEATURES
# ============================================================

# Total estimated shipment cost
df["estimated_route_cost"] = (
    df["units"] * df["cost_per_unit"]
)


# Total estimated CO2
df["estimated_co2"] = (
    df["units"] * df["co2_per_unit"]
)


# ============================================================
# 5. CREATE PRODUCT-RELATED FEATURES
# ============================================================

# Weight per unit
df["weight_per_unit"] = df["weight"]


# Total shipment weight
df["total_weight"] = (
    df["weight"] * df["units"]
)


# ============================================================
# 6. CREATE LOG-TRANSFORMED FEATURES
# ============================================================

# Log transformation reduces the effect of very large values
df["log_units"] = np.log1p(
    df["units"]
)

df["log_distance"] = np.log1p(
    df["distance"]
)

df["log_total_weight"] = np.log1p(
    df["total_weight"]
)


# ============================================================
# 7. CREATE MISSING INFORMATION FLAGS
# ============================================================

df["product_data_missing"] = (
    df["weight"].isna()
).astype(int)


df["route_data_missing"] = (
    df["distance"].isna()
).astype(int)


# ============================================================
# 8. CREATE NETWORK-RELATED BASIC FEATURES
# ============================================================

# Number of orders associated with each logistic hub
hub_order_count = (
    df["logistic_hub"]
    .value_counts()
)

df["hub_order_volume"] = (
    df["logistic_hub"]
    .map(hub_order_count)
)


# Number of orders associated with each customer
customer_order_count = (
    df["customer"]
    .value_counts()
)

df["customer_order_volume"] = (
    df["customer"]
    .map(customer_order_count)
)


# Number of orders handled by each 3PL
three_pl_order_count = (
    df["3pl"]
    .value_counts()
)

df["three_pl_order_volume"] = (
    df["3pl"]
    .map(three_pl_order_count)
)


# ============================================================
# 9. CREATE ROUTE DISTANCE CATEGORY
# ============================================================

df["distance_category"] = pd.cut(
    df["distance"],
    bins=[
        -np.inf,
        500,
        1000,
        2000,
        np.inf
    ],
    labels=[
        "Short",
        "Medium",
        "Long",
        "Very_Long"
    ]
)


# ============================================================
# 10. CREATE SHIPMENT SIZE CATEGORY
# ============================================================

df["shipment_size"] = pd.qcut(
    df["units"],
    q=4,
    labels=[
        "Small",
        "Medium",
        "Large",
        "Very_Large"
    ],
    duplicates="drop"
)


# ============================================================
# 11. HANDLE UNKNOWN PRODUCT ID
# ============================================================

# product_id was already converted from -1 to missing
# during preprocessing.

print("\n========== PRODUCT DATA ==========")

print(
    "Missing product IDs:",
    df["product_id"].isna().sum()
)


# ============================================================
# 12. CHECK NEW FEATURES
# ============================================================

print("\n========== FEATURE LIST ==========")

print(
    df.columns.tolist()
)


# ============================================================
# 13. CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

print(
    df.isnull().sum()
)


# ============================================================
# 14. CHECK NUMERICAL FEATURES
# ============================================================

print("\n========== NUMERICAL FEATURES ==========")

numerical_features = [
    "units",
    "weight",
    "weight_class",
    "distance",
    "cost_per_unit",
    "co2_per_unit",
    "estimated_route_cost",
    "estimated_co2",
    "total_weight",
    "log_units",
    "log_distance",
    "log_total_weight",
    "hub_order_volume",
    "customer_order_volume",
    "three_pl_order_volume"
]

print(
    df[numerical_features].describe()
)


# ============================================================
# 15. TARGET DISTRIBUTION
# ============================================================

print("\n========== TARGET ==========")

print(
    df["late_order"].value_counts()
)

print("\nTarget Percentage:")

print(
    df["late_order"]
    .value_counts(normalize=True) * 100
)


# ============================================================
# 16. FINAL DATASET INFORMATION
# ============================================================

print("\n========== FINAL FEATURE DATASET ==========")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 17. SAVE FEATURE DATASET
# ============================================================

df.to_csv(
    "feature_engineered_supply_chain_data.csv",
    index=False
)


print("\n==============================================")
print("FEATURE ENGINEERING COMPLETED")
print("==============================================")

print(
    "\nSaved as: feature_engineered_supply_chain_data.csv"
)
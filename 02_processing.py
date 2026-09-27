import pandas as pd


# ============================================================
# 1. LOAD DATASETS
# ============================================================

orders = pd.read_csv("orders.csv", sep=";")
products = pd.read_csv("product_attributes.csv", sep=",")
weight_class = pd.read_csv("product_weight_class.csv", sep=",")
costs = pd.read_csv("cities_data_costs.csv", sep=",")


# ============================================================
# 2. MERGE PRODUCT INFORMATION
# ============================================================

df = orders.merge(
    products,
    on="product_id",
    how="left"
)

df = df.merge(
    weight_class,
    on="product_id",
    how="left"
)


# ============================================================
# 3. CHECK PRODUCT MERGE
# ============================================================

print("\n========== PRODUCT MERGE ==========")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Product Information:")
print(
    df[
        ["weight", "material_handling", "weight_class"]
    ].isnull().sum()
)

print("\nUnknown Product Orders:")
print((df["product_id"] == -1).sum())


# ============================================================
# 4. MERGE ROUTE INFORMATION
# ============================================================

df = df.merge(
    costs[
        [
            "city_from_name",
            "city_to_name",
            "weight_class",
            "distance",
            "cost_per_unit",
            "co2_per_unit"
        ]
    ],
    left_on=[
        "logistic_hub",
        "customer",
        "weight_class"
    ],
    right_on=[
        "city_from_name",
        "city_to_name",
        "weight_class"
    ],
    how="left"
)


# ============================================================
# 5. CHECK ROUTE MERGE
# ============================================================

print("\n========== ROUTE MERGE ==========")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Route Information:")
print(
    df[
        ["distance", "cost_per_unit", "co2_per_unit"]
    ].isnull().sum()
)

print("\nSample Route Data:")
print(
    df[
        [
            "logistic_hub",
            "customer",
            "weight_class",
            "distance",
            "cost_per_unit",
            "co2_per_unit"
        ]
    ].head(10)
)


# ============================================================
# 6. REMOVE UNNECESSARY MERGE COLUMNS
# ============================================================

df.drop(
    columns=[
        "city_from_name",
        "city_to_name"
    ],
    inplace=True
)


# ============================================================
# 7. HANDLE UNKNOWN PRODUCT IDs
# ============================================================

print("\n========== UNKNOWN PRODUCT HANDLING ==========")

print(
    "Rows before handling:",
    len(df)
)

unknown_product_count = (
    df["product_id"] == -1
).sum()

print(
    "Unknown product rows:",
    unknown_product_count
)

# Replace -1 with missing value
df.loc[
    df["product_id"] == -1,
    "product_id"
] = pd.NA


# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())


# ============================================================
# 9. HANDLE LOGISTIC HUB MISSING VALUES
# ============================================================

print("\n========== LOGISTIC HUB ==========")

print(
    "Missing logistic hubs:",
    df["logistic_hub"].isnull().sum()
)

# Keep missing logistic hubs for now.
# According to the dataset documentation,
# these represent direct shipments from port to customer.


# ============================================================
# 10. NUMERICAL FEATURE SUMMARY
# ============================================================

print("\n========== NUMERICAL FEATURES ==========")

print(
    df[
        [
            "units",
            "weight",
            "weight_class",
            "distance",
            "cost_per_unit",
            "co2_per_unit"
        ]
    ].describe()
)


# ============================================================
# 11. CATEGORICAL FEATURE SUMMARY
# ============================================================

print("\n========== CATEGORICAL FEATURES ==========")

categorical_columns = [
    "origin_port",
    "3pl",
    "customs_procedures",
    "logistic_hub",
    "customer"
]

for column in categorical_columns:

    print(f"\n{column}:")
    print(
        df[column].value_counts(dropna=False)
    )


# ============================================================
# 12. TARGET CHECK
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
# 13. FINAL DATASET INFORMATION
# ============================================================

print("\n========== FINAL DATASET ==========")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 14. SAVE PROCESSED DATASET
# ============================================================

df.to_csv(
    "processed_supply_chain_data.csv",
    index=False
)

print("\n==============================================")
print("PREPROCESSING COMPLETED")
print("==============================================")

print(
    "\nSaved as: processed_supply_chain_data.csv"
)
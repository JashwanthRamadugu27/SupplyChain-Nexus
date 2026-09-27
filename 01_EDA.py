import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATA
# ============================================================

orders = pd.read_csv("orders.csv", sep=";")
products = pd.read_csv("product_attributes.csv", sep=",")
weight_class = pd.read_csv("product_weight_class.csv", sep=",")
cities = pd.read_csv("cities_data.csv", sep=";")
costs = pd.read_csv("cities_data_costs.csv", sep=",")


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n========== ORDERS DATASET ==========")

print("\nShape:")
print(orders.shape)

print("\nColumns:")
print(orders.columns.tolist())

print("\nFirst 5 Rows:")
print(orders.head())

print("\nData Types:")
print(orders.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

print(orders.isnull().sum())


# ============================================================
# 4. DUPLICATES
# ============================================================

print("\n========== DUPLICATES ==========")

print("Duplicate rows:", orders.duplicated().sum())


# ============================================================
# 5. UNIQUE VALUES
# ============================================================

print("\n========== UNIQUE VALUES ==========")

for column in [
    "origin_port",
    "3pl",
    "customs_procedures",
    "logistic_hub",
    "customer",
    "product_id"
]:
    print(column, ":", orders[column].nunique())


# ============================================================
# 6. TARGET DISTRIBUTION
# ============================================================

print("\n========== TARGET: LATE ORDER ==========")

print(orders["late_order"].value_counts())

print("\nPercentage:")
print(orders["late_order"].value_counts(normalize=True) * 100)


# ============================================================
# 7. CATEGORICAL ANALYSIS
# ============================================================

categorical_columns = [
    "3pl",
    "origin_port",
    "customs_procedures",
    "logistic_hub",
    "customer"
]

print("\n========== CATEGORICAL ANALYSIS ==========")

for column in categorical_columns:

    print(f"\n--- {column} ---")

    result = (
        orders.groupby(column)["late_order"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    result["late_percentage"] = result["mean"] * 100

    print(result)


# ============================================================
# 8. NUMERICAL ANALYSIS
# ============================================================

print("\n========== NUMERICAL ANALYSIS ==========")

print("\nUnits Statistics:")
print(orders["units"].describe())

print("\nUnits vs Late Order:")
print(
    orders.groupby("late_order")["units"]
    .agg(["count", "mean", "median", "min", "max"])
)


# ============================================================
# 9. PRODUCT DATA
# ============================================================

print("\n========== PRODUCT DATA ==========")

print("Product Attributes Shape:", products.shape)
print("Weight Class Shape:", weight_class.shape)

print("\nProduct Attributes:")
print(products.head())

print("\nWeight Classes:")
print(weight_class.head())

print("\nMissing Product Attribute Values:")
print(products.isnull().sum())

print("\nMissing Weight Class Values:")
print(weight_class.isnull().sum())


# ============================================================
# 10. CHECK PRODUCT ID MATCHING
# ============================================================

print("\n========== PRODUCT ID MATCHING ==========")

order_products = set(orders["product_id"])
master_products = set(products["product_id"])

missing_products = order_products - master_products

print("Products in orders:", len(order_products))
print("Products in product table:", len(master_products))
print("Products missing from product table:", len(missing_products))

if missing_products:
    print("Missing Product IDs:")
    print(list(missing_products)[:20])


# ============================================================
# 11. MERGE PRODUCT INFORMATION
# ============================================================

orders_products = orders.merge(
    products,
    on="product_id",
    how="left"
)

orders_products = orders_products.merge(
    weight_class,
    on="product_id",
    how="left"
)

print("\n========== MERGED PRODUCT DATA ==========")

print("Merged Shape:", orders_products.shape)

print("\nMerged Columns:")
print(orders_products.columns.tolist())

print("\nMissing Values After Product Merge:")
print(
    orders_products[
        ["weight", "material_handling"]
    ].isnull().sum()
)


# ============================================================
# 12. PRODUCT FEATURES VS LATE ORDER
# ============================================================

print("\n========== PRODUCT FEATURES VS LATE ORDER ==========")

print("\nWeight vs Late Order:")

print(
    orders_products.groupby("late_order")["weight"]
    .agg(["count", "mean", "median"])
)


print("\nMaterial Handling vs Late Order:")

print(
    orders_products.groupby("material_handling")["late_order"]
    .agg(["count", "mean"])
)


# ============================================================
# 13. CITY DATA
# ============================================================

print("\n========== CITY DATA ==========")

print("Cities Shape:", cities.shape)

print("\nColumns:")
print(cities.columns.tolist())

print("\nFirst 5 Rows:")
print(cities.head())

print("\nMissing Values:")
print(cities.isnull().sum())

print("\nDistance Statistics:")
print(cities["distance"].describe())


# ============================================================
# 14. COST & CO2 DATA
# ============================================================

print("\n========== COST & CO2 DATA ==========")

print("Costs Shape:", costs.shape)

print("\nColumns:")
print(costs.columns.tolist())

print("\nFirst 5 Rows:")
print(costs.head())

print("\nMissing Values:")
print(costs.isnull().sum())


# ============================================================
# 15. VISUALIZATION - TARGET
# ============================================================

plt.figure(figsize=(6, 4))

sns.countplot(
    data=orders,
    x="late_order"
)

plt.title("Late Order Distribution")
plt.xlabel("Late Order")
plt.ylabel("Number of Orders")

plt.show()


# ============================================================
# 16. VISUALIZATION - UNITS
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=orders,
    x="late_order",
    y="units"
)

plt.title("Units vs Late Order")
plt.xlabel("Late Order")
plt.ylabel("Units")

plt.show()


# ============================================================
# 17. VISUALIZATION - 3PL
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=orders,
    x="3pl",
    y="late_order"
)

plt.title("Late Order Rate by 3PL")
plt.xlabel("3PL")
plt.ylabel("Late Order Rate")

plt.show()


# ============================================================
# 18. VISUALIZATION - ORIGIN PORT
# ============================================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=orders,
    x="origin_port",
    y="late_order"
)

plt.title("Late Order Rate by Origin Port")
plt.xlabel("Origin Port")
plt.ylabel("Late Order Rate")

plt.xticks(rotation=20)

plt.show()


# ============================================================
# 19. VISUALIZATION - LOGISTIC HUB
# ============================================================

plt.figure(figsize=(10, 5))

sns.barplot(
    data=orders,
    x="logistic_hub",
    y="late_order"
)

plt.title("Late Order Rate by Logistic Hub")
plt.xlabel("Logistic Hub")
plt.ylabel("Late Order Rate")

plt.xticks(rotation=30)

plt.show()


# ============================================================
# 20. VISUALIZATION - CUSTOMS
# ============================================================

plt.figure(figsize=(7, 5))

sns.barplot(
    data=orders,
    x="customs_procedures",
    y="late_order"
)

plt.title("Late Order Rate by Customs Procedure")
plt.xlabel("Customs Procedure")
plt.ylabel("Late Order Rate")

plt.show()


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("EDA COMPLETED")
print("==============================================")

print("\nOrders:", len(orders))
print("Products:", orders["product_id"].nunique())
print("Customers:", orders["customer"].nunique())
print("Origin Ports:", orders["origin_port"].nunique())
print("3PL Providers:", orders["3pl"].nunique())
print("Logistic Hubs:", orders["logistic_hub"].nunique())

late_rate = orders["late_order"].mean() * 100

print(f"Overall Late Order Rate: {late_rate:.2f}%")

print("\nEDA analysis finished successfully.")
import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("processed_supply_chain_data.csv")

costs = pd.read_csv("cities_data_costs.csv")


# ============================================================
# 2. FIND ORDERS WITH MISSING ROUTE INFORMATION
# ============================================================

missing_routes = df[
    df["distance"].isna()
].copy()


print("\n========== ROUTE MISMATCH INVESTIGATION ==========")

print("Total orders:", len(df))

print("Orders with missing route information:",
      len(missing_routes))


# ============================================================
# 3. CHECK LOGISTIC HUB STATUS
# ============================================================

print("\n========== LOGISTIC HUB STATUS ==========")

print(
    missing_routes["logistic_hub"]
    .value_counts(dropna=False)
)


# ============================================================
# 4. CHECK WEIGHT CLASS STATUS
# ============================================================

print("\n========== WEIGHT CLASS STATUS ==========")

print(
    missing_routes["weight_class"]
    .value_counts(dropna=False)
)


# ============================================================
# 5. CHECK CUSTOMER STATUS
# ============================================================

print("\n========== CUSTOMERS WITH MISSING ROUTES ==========")

print(
    missing_routes["customer"]
    .value_counts()
)


# ============================================================
# 6. CHECK HUB → CUSTOMER COMBINATIONS
# ============================================================

print("\n========== HUB → CUSTOMER COMBINATIONS ==========")

route_combinations = (
    missing_routes[
        ["logistic_hub", "customer"]
    ]
    .value_counts()
    .reset_index(
        name="order_count"
    )
)

print(
    route_combinations.head(50)
)


# ============================================================
# 7. CHECK IF MISSING ROUTES ARE DIRECT SHIPMENTS
# ============================================================

direct_shipments = missing_routes[
    missing_routes["logistic_hub"].isna()
]

print("\n========== DIRECT SHIPMENTS ==========")

print(
    "Missing logistic hub:",
    len(direct_shipments)
)

print(
    "Missing route but logistic hub available:",
    len(missing_routes) - len(direct_shipments)
)


# ============================================================
# 8. ROUTES WHERE LOGISTIC HUB EXISTS
# ============================================================

hub_route_missing = missing_routes[
    missing_routes["logistic_hub"].notna()
].copy()


print("\n========== MISSING HUB → CUSTOMER ROUTES ==========")

print(
    "Count:",
    len(hub_route_missing)
)

print(
    hub_route_missing[
        [
            "logistic_hub",
            "customer",
            "weight_class"
        ]
    ]
    .value_counts()
    .reset_index(name="order_count")
    .head(50)
)


# ============================================================
# 9. CHECK AVAILABLE CITY ROUTES
# ============================================================

available_routes = costs[
    [
        "city_from_name",
        "city_to_name",
        "weight_class"
    ]
].drop_duplicates()


print("\n========== AVAILABLE ROUTES ==========")

print(
    "Unique route + weight class combinations:",
    len(available_routes)
)


# ============================================================
# 10. CHECK WHETHER MISSING HUB ROUTES EXIST
# ============================================================

check_routes = (
    hub_route_missing[
        [
            "logistic_hub",
            "customer",
            "weight_class"
        ]
    ]
    .drop_duplicates()
)

check_routes = check_routes.merge(
    available_routes,
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
    how="left",
    indicator=True
)


print("\n========== ROUTE EXISTENCE CHECK ==========")

print(
    check_routes["_merge"]
    .value_counts()
)


# ============================================================
# 11. SHOW ACTUAL MISSING ROUTES
# ============================================================

actual_missing = check_routes[
    check_routes["_merge"] == "left_only"
].copy()


print("\n========== ACTUAL MISSING ROUTES ==========")

print(
    actual_missing[
        [
            "logistic_hub",
            "customer",
            "weight_class"
        ]
    ]
    .head(100)
)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n========== SUMMARY ==========")

print(
    "Total orders:",
    len(df)
)

print(
    "Orders with missing route data:",
    len(missing_routes)
)

print(
    "Direct shipments:",
    len(direct_shipments)
)

print(
    "Orders with hub but missing route:",
    len(hub_route_missing)
)

print(
    "Unique genuinely missing routes:",
    len(actual_missing)
)

print("\n==============================================")
print("ROUTE INVESTIGATION COMPLETED")
print("==============================================")
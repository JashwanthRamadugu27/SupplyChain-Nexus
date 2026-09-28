import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "feature_engineered_supply_chain_data.csv"
)

print("Data loaded successfully.")
print("Shape:", df.shape)


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "logistic_hub",
    "customer",
    "units",
    "late_order",
    "estimated_route_cost",
    "estimated_co2"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nMissing columns:")

    for column in missing_columns:
        print("-", column)

    raise ValueError(
        "Required columns are missing from the dataset."
    )


# ============================================================
# 3. GET AVAILABLE LOGISTIC HUBS
# ============================================================

hubs = sorted(
    df["logistic_hub"]
    .dropna()
    .unique()
)

print("\n================ LOGISTIC HUBS ================")

for hub in hubs:
    print("-", hub)


# ============================================================
# 4. FUNCTION TO SIMULATE HUB DISRUPTION
# ============================================================

def simulate_hub_disruption(data, disrupted_hub):

    affected = data[
        data["logistic_hub"] == disrupted_hub
    ].copy()

    # --------------------------------------------------------
    # Basic impact
    # --------------------------------------------------------

    affected_orders = len(affected)

    affected_units = affected["units"].sum()

    affected_customers = (
        affected["customer"]
        .nunique()
    )

    # --------------------------------------------------------
    # Late-order impact
    # --------------------------------------------------------

    late_orders = affected["late_order"].sum()

    if affected_orders > 0:

        late_order_rate = (
            late_orders / affected_orders
        )

    else:

        late_order_rate = 0


    # --------------------------------------------------------
    # Cost impact
    # --------------------------------------------------------

    estimated_cost = (
        affected["estimated_route_cost"]
        .sum()
    )

    # --------------------------------------------------------
    # CO2 impact
    # --------------------------------------------------------

    estimated_co2 = (
        affected["estimated_co2"]
        .sum()
    )


    return {

        "disrupted_hub": disrupted_hub,

        "affected_orders": affected_orders,

        "affected_units": affected_units,

        "affected_customers": affected_customers,

        "late_orders": late_orders,

        "late_order_rate": late_order_rate,

        "estimated_route_cost": estimated_cost,

        "estimated_co2": estimated_co2

    }


# ============================================================
# 5. SIMULATE ALL HUB DISRUPTIONS
# ============================================================

results = []


for hub in hubs:

    result = simulate_hub_disruption(
        df,
        hub
    )

    results.append(result)


disruption_results = pd.DataFrame(
    results
)


# ============================================================
# 6. SORT BY AFFECTED ORDERS
# ============================================================

disruption_results = (
    disruption_results
    .sort_values(
        "affected_orders",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print(
    "\n================ DISRUPTION IMPACT ================"
)

print(
    disruption_results.to_string(
        index=False
    )
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

disruption_results.to_csv(
    "hub_disruption_impact.csv",
    index=False
)


# ============================================================
# 9. TOP IMPACT HUBS
# ============================================================

print(
    "\n================ TOP IMPACT HUBS ================"
)

print(
    disruption_results[
        [
            "disrupted_hub",
            "affected_orders",
            "affected_units",
            "affected_customers",
            "late_order_rate"
        ]
    ]
    .head(5)
    .to_string(index=False)
)


# ============================================================
# 10. FINAL OUTPUT
# ============================================================

print(
    "\nFile created:"
)

print(
    "hub_disruption_impact.csv"
)

print(
    "\nDisruption simulation completed successfully."
)
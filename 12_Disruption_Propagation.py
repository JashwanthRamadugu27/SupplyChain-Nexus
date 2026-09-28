import pandas as pd
import networkx as nx


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "feature_engineered_supply_chain_data.csv"
)

print("Data loaded successfully.")
print("Shape:", df.shape)


# ============================================================
# 2. CREATE SUPPLY CHAIN GRAPH
# ============================================================

G = nx.DiGraph()


# ============================================================
# 3. BUILD NETWORK
# ============================================================

for _, row in df.iterrows():

    origin = row["origin_port"]
    hub = row["logistic_hub"]
    customer = row["customer"]

    # Port → Hub
    if pd.notna(origin) and pd.notna(hub):

        G.add_edge(
            f"PORT:{origin}",
            f"HUB:{hub}"
        )

    # Hub → Customer
    if pd.notna(hub) and pd.notna(customer):

        G.add_edge(
            f"HUB:{hub}",
            f"CUSTOMER:{customer}"
        )

    # Direct Port → Customer shipment
    elif pd.notna(origin) and pd.notna(customer):

        G.add_edge(
            f"PORT:{origin}",
            f"CUSTOMER:{customer}"
        )


# ============================================================
# 4. GET HUBS
# ============================================================

hubs = sorted(
    df["logistic_hub"]
    .dropna()
    .unique()
)


# ============================================================
# 5. GET CUSTOMERS
# ============================================================

customers = sorted(
    df["customer"]
    .dropna()
    .unique()
)


# ============================================================
# 6. FUNCTION TO ANALYZE DISRUPTION
# ============================================================

def analyze_disruption(data, disrupted_hub):

    # --------------------------------------------------------
    # Shipments currently using the disrupted hub
    # --------------------------------------------------------

    affected = data[
        data["logistic_hub"] == disrupted_hub
    ].copy()


    # --------------------------------------------------------
    # Customers depending on this hub
    # --------------------------------------------------------

    affected_customers = sorted(
        affected["customer"]
        .dropna()
        .unique()
    )


    # --------------------------------------------------------
    # Find alternative hubs for each affected customer
    # --------------------------------------------------------

    propagation_records = []


    for customer in affected_customers:

        alternative_hubs = []

        for hub in hubs:

            # Do not consider the disrupted hub
            if hub == disrupted_hub:
                continue


            # Check whether this hub serves the customer
            exists = (
                (data["logistic_hub"] == hub)
                &
                (data["customer"] == customer)
            ).any()


            if exists:

                alternative_hubs.append(hub)


        # ----------------------------------------------------
        # Customer-level impact
        # ----------------------------------------------------

        customer_orders = affected[
            affected["customer"] == customer
        ]


        affected_orders = len(
            customer_orders
        )


        affected_units = customer_orders[
            "units"
        ].sum()


        late_orders = customer_orders[
            "late_order"
        ].sum()


        if affected_orders > 0:

            late_rate = (
                late_orders /
                affected_orders
            )

        else:

            late_rate = 0


        propagation_records.append({

            "disrupted_hub": disrupted_hub,

            "customer": customer,

            "affected_orders": affected_orders,

            "affected_units": affected_units,

            "late_orders": late_orders,

            "late_order_rate": late_rate,

            "alternative_hub_count": len(
                alternative_hubs
            ),

            "alternative_hubs": ", ".join(
                alternative_hubs
            )

        })


    return pd.DataFrame(
        propagation_records
    )


# ============================================================
# 7. ANALYZE ALL HUB DISRUPTIONS
# ============================================================

all_results = []


for hub in hubs:

    result = analyze_disruption(
        df,
        hub
    )

    all_results.append(result)


propagation_results = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# 8. SAVE CUSTOMER-LEVEL RESULTS
# ============================================================

propagation_results.to_csv(
    "disruption_propagation_results.csv",
    index=False
)


# ============================================================
# 9. CREATE HUB SUMMARY
# ============================================================

hub_summary = (
    propagation_results
    .groupby("disrupted_hub")
    .agg(

        affected_customers=(
            "customer",
            "nunique"
        ),

        affected_orders=(
            "affected_orders",
            "sum"
        ),

        affected_units=(
            "affected_units",
            "sum"
        ),

        average_late_order_rate=(
            "late_order_rate",
            "mean"
        ),

        customers_with_alternatives=(
            "alternative_hub_count",
            lambda x: (x > 0).sum()
        )

    )
    .reset_index()
)


# ============================================================
# 10. CALCULATE ALTERNATIVE COVERAGE
# ============================================================

hub_summary["alternative_customer_coverage"] = (

    hub_summary[
        "customers_with_alternatives"
    ]
    /
    hub_summary[
        "affected_customers"
    ]

)


# ============================================================
# 11. SAVE HUB SUMMARY
# ============================================================

hub_summary.to_csv(
    "disruption_propagation_summary.csv",
    index=False
)


# ============================================================
# 12. DISPLAY SUMMARY
# ============================================================

print(
    "\n================ DISRUPTION PROPAGATION SUMMARY ================"
)

print(

    hub_summary
    .sort_values(
        "affected_orders",
        ascending=False
    )
    .to_string(index=False)

)


# ============================================================
# 13. DISPLAY SAMPLE CUSTOMER ANALYSIS
# ============================================================

print(
    "\n================ SAMPLE CUSTOMER IMPACT ================"
)

sample_hub = hubs[0]

sample_results = propagation_results[
    propagation_results["disrupted_hub"]
    == sample_hub
].head(10)


print(
    sample_results.to_string(
        index=False
    )
)


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print(
    "\n================ FILES CREATED ================"
)

print(
    "1. disruption_propagation_results.csv"
)

print(
    "2. disruption_propagation_summary.csv"
)

print(
    "\nDisruption propagation analysis completed successfully."
)
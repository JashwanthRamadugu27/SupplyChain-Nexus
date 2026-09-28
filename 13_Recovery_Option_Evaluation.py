import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("feature_engineered_supply_chain_data.csv")
costs = pd.read_csv("cities_data_costs.csv")

print("Data loaded successfully.")
print("Orders shape:", df.shape)
print("Cost data shape:", costs.shape)


# ============================================================
# 2. BASIC CLEANING
# ============================================================

for col in ["logistic_hub", "customer"]:
    df[col] = df[col].astype("string").str.strip()
    costs["city_from_name"] = costs["city_from_name"].astype("string").str.strip()
    costs["city_to_name"] = costs["city_to_name"].astype("string").str.strip()

df["weight_class"] = pd.to_numeric(
    df["weight_class"], errors="coerce"
).astype("Int64")

costs["weight_class"] = pd.to_numeric(
    costs["weight_class"], errors="coerce"
).astype("Int64")


# Check required columns
required_cost_columns = [
    "city_from_name",
    "city_to_name",
    "weight_class",
    "distance",
    "cost_per_unit",
    "co2_per_unit"
]

missing_columns = [
    col for col in required_cost_columns
    if col not in costs.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in cities_data_costs.csv: {missing_columns}"
    )


# ============================================================
# 3. HISTORICAL PERFORMANCE OF EACH HUB → CUSTOMER ROUTE
# ============================================================

route_history = (
    df.dropna(subset=["logistic_hub", "customer"])
      .groupby(["logistic_hub", "customer"])
      .agg(
          historical_orders=("order_id", "count"),
          historical_units=("units", "sum"),
          historical_late_rate=("late_order", "mean")
      )
      .reset_index()
)

print("\nHistorical route statistics calculated.")


# ============================================================
# 4. LIST OF HUBS
# ============================================================

hubs = sorted(
    df["logistic_hub"]
    .dropna()
    .unique()
)

customers = sorted(
    df["customer"]
    .dropna()
    .unique()
)

print("Number of hubs:", len(hubs))
print("Number of customers:", len(customers))


# ============================================================
# 5. EVALUATE RECOVERY OPTIONS
# ============================================================

results = []

for disrupted_hub in hubs:

    # Orders affected if this hub becomes unavailable
    affected = df[
        df["logistic_hub"] == disrupted_hub
    ].copy()

    if affected.empty:
        continue

    affected_customers = sorted(
        affected["customer"].dropna().unique()
    )

    # Every other hub becomes a potential recovery option
    alternative_hubs = [
        hub for hub in hubs
        if hub != disrupted_hub
    ]

    for customer in affected_customers:

        customer_shipments = affected[
            affected["customer"] == customer
        ].copy()

        affected_orders = len(customer_shipments)
        affected_units = customer_shipments["units"].sum()

        if affected_units == 0:
            continue

        # ----------------------------------------------------
        # Evaluate every alternative hub
        # ----------------------------------------------------

        for alternative_hub in alternative_hubs:

            shipment_count = len(customer_shipments)

            # Create alternative route keys
            temp = customer_shipments[
                ["order_id", "units", "weight_class"]
            ].copy()

            temp["alternative_hub"] = alternative_hub
            temp["customer"] = customer

            # ------------------------------------------------
            # Match alternative hub → customer route costs
            # ------------------------------------------------

            route_costs = costs[
                (costs["city_from_name"] == alternative_hub)
                &
                (costs["city_to_name"] == customer)
            ][
                [
                    "city_from_name",
                    "city_to_name",
                    "weight_class",
                    "distance",
                    "cost_per_unit",
                    "co2_per_unit"
                ]
            ].copy()

            route_costs = route_costs.rename(
                columns={
                    "city_from_name": "alternative_hub"
                }
            )

            temp = temp.merge(
                route_costs,
                on=["alternative_hub", "weight_class"],
                how="left"
            )

            # ------------------------------------------------
            # Route coverage
            # ------------------------------------------------

            matched = temp["distance"].notna()

            available_orders = matched.sum()
            available_units = temp.loc[
                matched, "units"
            ].sum()

            route_order_coverage = (
                available_orders / shipment_count
                if shipment_count > 0
                else 0
            )

            route_unit_coverage = (
                available_units / affected_units
                if affected_units > 0
                else 0
            )

            # ------------------------------------------------
            # Weighted route metrics
            # ------------------------------------------------

            valid = temp[
                temp["distance"].notna()
            ].copy()

            if not valid.empty:

                weighted_distance = np.average(
                    valid["distance"],
                    weights=valid["units"]
                )

                weighted_cost = np.average(
                    valid["cost_per_unit"],
                    weights=valid["units"]
                )

                weighted_co2 = np.average(
                    valid["co2_per_unit"],
                    weights=valid["units"]
                )

                estimated_reroute_cost = (
                    valid["units"]
                    * valid["cost_per_unit"]
                ).sum()

                estimated_reroute_co2 = (
                    valid["units"]
                    * valid["co2_per_unit"]
                ).sum()

            else:

                weighted_distance = np.nan
                weighted_cost = np.nan
                weighted_co2 = np.nan
                estimated_reroute_cost = np.nan
                estimated_reroute_co2 = np.nan

            # ------------------------------------------------
            # Historical performance
            # ------------------------------------------------

            history = route_history[
                (route_history["logistic_hub"] == alternative_hub)
                &
                (route_history["customer"] == customer)
            ]

            if not history.empty:

                alternative_late_rate = (
                    history.iloc[0]["historical_late_rate"]
                )

                alternative_historical_orders = (
                    history.iloc[0]["historical_orders"]
                )

                alternative_historical_units = (
                    history.iloc[0]["historical_units"]
                )

            else:

                alternative_late_rate = np.nan
                alternative_historical_orders = 0
                alternative_historical_units = 0

            results.append({
                "disrupted_hub": disrupted_hub,
                "customer": customer,
                "alternative_hub": alternative_hub,

                "affected_orders": affected_orders,
                "affected_units": affected_units,

                "alternative_route_orders_available":
                    available_orders,

                "alternative_route_units_available":
                    available_units,

                "route_order_coverage":
                    route_order_coverage,

                "route_unit_coverage":
                    route_unit_coverage,

                "alternative_distance":
                    weighted_distance,

                "alternative_cost_per_unit":
                    weighted_cost,

                "alternative_co2_per_unit":
                    weighted_co2,

                "estimated_reroute_cost":
                    estimated_reroute_cost,

                "estimated_reroute_co2":
                    estimated_reroute_co2,

                "alternative_historical_late_rate":
                    alternative_late_rate,

                "alternative_historical_orders":
                    alternative_historical_orders,

                "alternative_historical_units":
                    alternative_historical_units
            })


# ============================================================
# 6. CREATE RESULTS DATAFRAME
# ============================================================

recovery_df = pd.DataFrame(results)

print("\nRecovery options generated.")
print("Recovery option rows:", len(recovery_df))


# ============================================================
# 7. CALCULATE SCORE
# ============================================================
#
# Lower is better:
#   - cost
#   - distance
#   - CO2
#   - historical late rate
#
# Higher is better:
#   - route coverage
#
# The weights are intentionally visible and configurable.
# This is a decision-support score, NOT an ML prediction.
# ============================================================

WEIGHTS = {
    "cost": 0.35,
    "distance": 0.20,
    "late_rate": 0.20,
    "co2": 0.15,
    "coverage": 0.10
}


def min_max_lower_better(series):

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            np.nan,
            index=series.index
        )

    if maximum == minimum:
        return pd.Series(
            1.0,
            index=series.index
        )

    return 1 - (
        (series - minimum)
        / (maximum - minimum)
    )


def min_max_higher_better(series):

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            np.nan,
            index=series.index
        )

    if maximum == minimum:
        return pd.Series(
            1.0,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


recovery_df["cost_score"] = (
    recovery_df
    .groupby(["disrupted_hub", "customer"])
    ["alternative_cost_per_unit"]
    .transform(min_max_lower_better)
)

recovery_df["distance_score"] = (
    recovery_df
    .groupby(["disrupted_hub", "customer"])
    ["alternative_distance"]
    .transform(min_max_lower_better)
)

recovery_df["late_rate_score"] = (
    recovery_df
    .groupby(["disrupted_hub", "customer"])
    ["alternative_historical_late_rate"]
    .transform(min_max_lower_better)
)

recovery_df["co2_score"] = (
    recovery_df
    .groupby(["disrupted_hub", "customer"])
    ["alternative_co2_per_unit"]
    .transform(min_max_lower_better)
)

recovery_df["coverage_score"] = (
    recovery_df["route_unit_coverage"]
)


# ============================================================
# 8. FINAL RECOVERY SCORE
# ============================================================

recovery_df["recovery_score"] = (
    recovery_df["cost_score"] * WEIGHTS["cost"]
    + recovery_df["distance_score"] * WEIGHTS["distance"]
    + recovery_df["late_rate_score"] * WEIGHTS["late_rate"]
    + recovery_df["co2_score"] * WEIGHTS["co2"]
    + recovery_df["coverage_score"] * WEIGHTS["coverage"]
) * 100


# Options without route information cannot receive a meaningful score
recovery_df.loc[
    recovery_df["route_unit_coverage"] == 0,
    "recovery_score"
] = np.nan


# ============================================================
# 9. RANK RECOVERY OPTIONS
# ============================================================

recovery_df["recovery_rank"] = (
    recovery_df
    .groupby(["disrupted_hub", "customer"])
    ["recovery_score"]
    .rank(
        ascending=False,
        method="min"
    )
)


# ============================================================
# 10. COMPARE ALTERNATIVE LATE RATE WITH DISRUPTED HUB
# ============================================================

disrupted_late_rates = (
    df.dropna(subset=["logistic_hub", "customer"])
      .groupby(["logistic_hub", "customer"])
      ["late_order"]
      .mean()
      .reset_index()
      .rename(
          columns={
              "logistic_hub": "disrupted_hub",
              "late_order": "disrupted_historical_late_rate"
          }
      )
)

recovery_df = recovery_df.merge(
    disrupted_late_rates,
    on=["disrupted_hub", "customer"],
    how="left"
)

recovery_df[
    "late_rate_difference_vs_disrupted"
] = (
    recovery_df["alternative_historical_late_rate"]
    - recovery_df["disrupted_historical_late_rate"]
)


# ============================================================
# 11. SAVE COMPLETE RESULTS
# ============================================================

recovery_df.to_csv(
    "recovery_option_results.csv",
    index=False
)


# ============================================================
# 12. CREATE SUMMARY
# ============================================================

summary = (
    recovery_df[
        recovery_df["recovery_score"].notna()
    ]
    .sort_values(
        [
            "disrupted_hub",
            "customer",
            "recovery_score"
        ],
        ascending=[True, True, False]
    )
    .groupby(
        ["disrupted_hub", "customer"],
        as_index=False
    )
    .first()
)

summary.to_csv(
    "recovery_option_summary.csv",
    index=False
)


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\n================ RECOVERY OPTION SUMMARY ================")

print(
    summary[
        [
            "disrupted_hub",
            "customer",
            "alternative_hub",
            "recovery_score",
            "alternative_cost_per_unit",
            "alternative_distance",
            "alternative_historical_late_rate",
            "route_unit_coverage"
        ]
    ].head(30).to_string(index=False)
)


# ============================================================
# 14. FINAL INFORMATION
# ============================================================

print("\n================ FILES CREATED ================")

print("1. recovery_option_results.csv")
print("2. recovery_option_summary.csv")

print("\nRecovery evaluation completed.")

print("\nConfigured recovery score weights:")
print("Cost       :", WEIGHTS["cost"])
print("Distance   :", WEIGHTS["distance"])
print("Late Rate  :", WEIGHTS["late_rate"])
print("CO2        :", WEIGHTS["co2"])
print("Coverage   :", WEIGHTS["coverage"])

print(
    "\nNOTE: Recovery score is a transparent decision-support "
    "score based on configured weights. It is not an ML prediction."
)
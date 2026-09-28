import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp


# ============================================================
# 1. LOAD DATA
# ============================================================

recovery = pd.read_csv("recovery_option_results.csv")

print("Recovery data loaded successfully.")
print("Shape:", recovery.shape)


# ============================================================
# 2. CLEAN DATA
# ============================================================

numeric_columns = [
    "affected_units",
    "alternative_cost_per_unit",
    "alternative_distance",
    "alternative_co2_per_unit",
    "alternative_historical_late_rate",
    "route_unit_coverage"
]

for col in numeric_columns:
    recovery[col] = pd.to_numeric(
        recovery[col],
        errors="coerce"
    )


# ============================================================
# 3. REMOVE OPTIONS WITHOUT ROUTE INFORMATION
# ============================================================

recovery = recovery[
    recovery["route_unit_coverage"] > 0
].copy()

recovery = recovery[
    recovery["alternative_cost_per_unit"].notna()
    & recovery["alternative_distance"].notna()
    & recovery["alternative_co2_per_unit"].notna()
    & recovery["alternative_historical_late_rate"].notna()
].copy()

print(
    "Usable recovery options:",
    len(recovery)
)


# ============================================================
# 4. OPTIMIZATION WEIGHTS
# ============================================================
#
# These weights define the objective.
#
# Higher weight = more importance.
#
# They can be changed later depending on the business scenario.
# ============================================================

COST_WEIGHT = 0.40
LATE_RISK_WEIGHT = 0.30
DISTANCE_WEIGHT = 0.15
CO2_WEIGHT = 0.15


# ============================================================
# 5. NORMALIZATION FUNCTION
# ============================================================

def normalize_lower_better(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            1.0,
            index=series.index
        )

    return 1 - (
        (series - minimum)
        / (maximum - minimum)
    )


# ============================================================
# 6. NORMALIZE METRICS
# ============================================================

recovery["cost_score"] = (
    recovery
    .groupby(["disrupted_hub", "customer"])
    ["alternative_cost_per_unit"]
    .transform(normalize_lower_better)
)

recovery["late_risk_score"] = (
    recovery
    .groupby(["disrupted_hub", "customer"])
    ["alternative_historical_late_rate"]
    .transform(normalize_lower_better)
)

recovery["distance_score"] = (
    recovery
    .groupby(["disrupted_hub", "customer"])
    ["alternative_distance"]
    .transform(normalize_lower_better)
)

recovery["co2_score"] = (
    recovery
    .groupby(["disrupted_hub", "customer"])
    ["alternative_co2_per_unit"]
    .transform(normalize_lower_better)
)


# ============================================================
# 7. CREATE OPTIMIZATION SCORE
# ============================================================

recovery["optimization_score"] = (
    recovery["cost_score"] * COST_WEIGHT
    + recovery["late_risk_score"] * LATE_RISK_WEIGHT
    + recovery["distance_score"] * DISTANCE_WEIGHT
    + recovery["co2_score"] * CO2_WEIGHT
)


# ============================================================
# 8. OPTIMIZE EACH DISRUPTION + CUSTOMER
# ============================================================

optimization_results = []


groups = recovery[
    ["disrupted_hub", "customer"]
].drop_duplicates()


for _, group_info in groups.iterrows():

    disrupted_hub = group_info["disrupted_hub"]
    customer = group_info["customer"]

    options = recovery[
        (recovery["disrupted_hub"] == disrupted_hub)
        &
        (recovery["customer"] == customer)
    ].copy()

    if options.empty:
        continue


    # --------------------------------------------------------
    # Create OR-Tools solver
    # --------------------------------------------------------

    solver = pywraplp.Solver.CreateSolver(
        "SCIP"
    )

    if solver is None:
        raise RuntimeError(
            "OR-Tools SCIP solver could not be created."
        )


    # --------------------------------------------------------
    # Decision variables
    # --------------------------------------------------------
    #
    # x[i] = proportion of affected shipment allocated
    #        to alternative hub i.
    #
    # 0 <= x[i] <= 1
    # --------------------------------------------------------

    variables = {}

    for index in options.index:

        variables[index] = solver.NumVar(
            0,
            1,
            f"x_{index}"
        )


    # --------------------------------------------------------
    # Demand constraint
    # --------------------------------------------------------
    #
    # All affected shipment volume must be allocated.
    # --------------------------------------------------------

    solver.Add(
        solver.Sum(
            variables[index]
            for index in options.index
        ) == 1
    )


    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------
    #
    # Minimize weighted recovery cost.
    # --------------------------------------------------------

    objective = solver.Objective()

    for index in options.index:

        row = options.loc[index]

        objective.SetCoefficient(
            variables[index],
            -row["optimization_score"]
        )

    objective.SetMaximization()


    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    status = solver.Solve()


    if status != pywraplp.Solver.OPTIMAL:
        print(
            f"Optimization failed for "
            f"{disrupted_hub} -> {customer}"
        )
        continue


    # --------------------------------------------------------
    # Store selected solution
    # --------------------------------------------------------

    for index in options.index:

        allocation = variables[index].solution_value()

        if allocation > 0.000001:

            row = options.loc[index]

            optimization_results.append({

                "disrupted_hub":
                    disrupted_hub,

                "customer":
                    customer,

                "selected_alternative_hub":
                    row["alternative_hub"],

                "affected_units":
                    row["affected_units"],

                "allocation_percentage":
                    allocation * 100,

                "allocated_units":
                    row["affected_units"] * allocation,

                "alternative_cost_per_unit":
                    row["alternative_cost_per_unit"],

                "alternative_distance":
                    row["alternative_distance"],

                "alternative_co2_per_unit":
                    row["alternative_co2_per_unit"],

                "alternative_historical_late_rate":
                    row["alternative_historical_late_rate"],

                "optimization_score":
                    row["optimization_score"]
            })


# ============================================================
# 9. CREATE RESULT DATAFRAME
# ============================================================

optimization_df = pd.DataFrame(
    optimization_results
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

optimization_df.to_csv(
    "recovery_optimization_results.csv",
    index=False
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print(
    "\n================ RECOVERY OPTIMIZATION ================"
)

print(
    optimization_df.head(40).to_string(
        index=False
    )
)


# ============================================================
# 12. SUMMARY
# ============================================================

print(
    "\n================ OPTIMIZATION SUMMARY ================"
)

print(
    "Disruption-customer combinations:",
    optimization_df[
        ["disrupted_hub", "customer"]
    ].drop_duplicates().shape[0]
)

print(
    "Selected recovery allocations:",
    len(optimization_df)
)


# ============================================================
# 13. WEIGHTS
# ============================================================

print(
    "\nOptimization weights:"
)

print(
    "Cost:",
    COST_WEIGHT
)

print(
    "Late risk:",
    LATE_RISK_WEIGHT
)

print(
    "Distance:",
    DISTANCE_WEIGHT
)

print(
    "CO2:",
    CO2_WEIGHT
)


print(
    "\nRecovery optimization completed."
)

print(
    "File created:"
)

print(
    "recovery_optimization_results.csv"
)

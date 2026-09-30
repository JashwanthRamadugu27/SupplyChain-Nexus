import pandas as pd
import numpy as np
import joblib


# ============================================================
# LOAD MODEL + PREPROCESSOR
# ============================================================

model = joblib.load(
    "logistic_regression_model.pkl"
)

preprocessor = joblib.load(
    "supply_chain_preprocessor.pkl"
)


# ============================================================
# LOAD FEATURE-ENGINEERED DATA
# ============================================================

df = pd.read_csv(
    "feature_engineered_supply_chain_data.csv"
)


# ============================================================
# RECREATE SHIPMENT SIZE BINS
# ============================================================

_, shipment_bins = pd.qcut(
    df["units"],
    q=4,
    retbins=True,
    duplicates="drop"
)


# ============================================================
# SELECT REAL SHIPMENTS
# ============================================================

samples = df.sample(
    10,
    random_state=42
).copy()


# ============================================================
# FUNCTION TO CREATE MODEL INPUT
# ============================================================

def prepare_input(row):

    input_data = pd.DataFrame([{

        "origin_port":
            row["origin_port"],

        "3pl":
            row["3pl"],

        "customs_procedures":
            row["customs_procedures"],

        "logistic_hub":
            row["logistic_hub"],

        "customer":
            row["customer"],

        "material_handling":
            row["material_handling"],

        "weight_class":
            row["weight_class"],

        "distance_category":
            row["distance_category"],

        "shipment_size":
            row["shipment_size"],

        "units":
            row["units"],

        "weight":
            row["weight"],

        "distance":
            row["distance"],

        "cost_per_unit":
            row["cost_per_unit"],

        "co2_per_unit":
            row["co2_per_unit"],

        "route_data_available":
            row["route_data_available"],

        "direct_shipment":
            row["direct_shipment"],

        "estimated_route_cost":
            row["estimated_route_cost"],

        "estimated_co2":
            row["estimated_co2"],

        "weight_per_unit":
            row["weight_per_unit"],

        "total_weight":
            row["total_weight"],

        "log_units":
            row["log_units"],

        "log_distance":
            row["log_distance"],

        "log_total_weight":
            row["log_total_weight"],

        "product_data_missing":
            row["product_data_missing"],

        "route_data_missing":
            row["route_data_missing"]
    }])

    return input_data


# ============================================================
# VALIDATE
# ============================================================

results = []


for _, row in samples.iterrows():

    input_data = prepare_input(row)

    processed = preprocessor.transform(
        input_data
    )

    probability = model.predict_proba(
        processed
    )[0][1]

    threshold = 0.35

    predicted_risk = (
        1
        if probability >= threshold
        else 0
    )

    results.append({

        "order_id":
            row["order_id"],

        "actual_late_order":
            int(row["late_order"]),

        "predicted_probability":
            probability,

        "predicted_risk":
            predicted_risk
    })


# ============================================================
# DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n========== API / MODEL VALIDATION ==========\n")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "api_validation_results.csv",
    index=False
)

print(
    "\nSaved as: api_validation_results.csv"
)
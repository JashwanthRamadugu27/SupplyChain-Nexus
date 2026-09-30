import pandas as pd
import numpy as np
import joblib


# ============================================================
# LOAD DATA AND FINAL PIPELINE
# ============================================================

df = pd.read_csv(
    "feature_engineered_supply_chain_data.csv"
)

pipeline = joblib.load(
    "supply_chain_ml_pipeline.pkl"
)


# ============================================================
# SAMPLE REAL SHIPMENTS
# ============================================================

sample = df.sample(
    20,
    random_state=42
).copy()


# ============================================================
# CREATE EXACT MODEL FEATURES
# ============================================================

model_data = pd.DataFrame({

    "origin_port": sample["origin_port"],
    "3pl": sample["3pl"],
    "customs_procedures": sample["customs_procedures"],
    "logistic_hub": sample["logistic_hub"],
    "customer": sample["customer"],

    "units": sample["units"],
    "weight": sample["weight"],
    "distance": sample["distance"],

    "cost_per_unit": sample["cost_per_unit"],
    "co2_per_unit": sample["co2_per_unit"],

    "route_data_available":
        sample["route_data_available"],

    "direct_shipment":
        sample["direct_shipment"],

    "estimated_route_cost":
        sample["estimated_route_cost"],

    "estimated_co2":
        sample["estimated_co2"],

    "weight_per_unit":
        sample["weight_per_unit"],

    "total_weight":
        sample["total_weight"],

    "log_units":
        sample["log_units"],

    "log_distance":
        sample["log_distance"],

    "log_total_weight":
        sample["log_total_weight"],

    "product_data_missing":
        sample["product_data_missing"],

    "route_data_missing":
        sample["route_data_missing"],

    "material_handling":
        sample["material_handling"],

    "weight_class":
        sample["weight_class"],

    "distance_category":
        sample["distance_category"],

    "shipment_size":
        sample["shipment_size"]
})


# ============================================================
# PREDICTION
# ============================================================

probabilities = pipeline.predict_proba(
    model_data
)[:, 1]


predictions = (
    probabilities >= 0.35
).astype(int)


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame({

    "order_id": sample["order_id"],

    "actual_late_order":
        sample["late_order"],

    "predicted_probability":
        probabilities,

    "predicted_risk":
        predictions

})


print()
print("=" * 70)
print("FINAL API / ML PIPELINE VALIDATION")
print("=" * 70)

print()

print(
    results.to_string(
        index=False
    )
)

print()

print("=" * 70)

print(
    "Actual late orders:",
    int(results["actual_late_order"].sum())
)

print(
    "Predicted high-risk orders:",
    int(results["predicted_risk"].sum())
)

print(
    "Average predicted probability:",
    round(
        results["predicted_probability"].mean(),
        4
    )
)

print("=" * 70)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    "final_api_validation_results.csv",
    index=False
)

print()
print(
    "Saved as: final_api_validation_results.csv"
)
print()
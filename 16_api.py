from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SupplyChain Nexus API",
    description="ML-driven supply-chain disruption and recovery decision support API",
    version="1.0.0"
)


# ============================================================
# LOAD DATA
# ============================================================

disruption_data = pd.read_csv("hub_disruption_impact.csv")
propagation_data = pd.read_csv("disruption_propagation_results.csv")
recovery_data = pd.read_csv("recovery_option_results.csv")
optimization_data = pd.read_csv("recovery_optimization_results.csv")

feature_data = pd.read_csv("feature_engineered_supply_chain_data.csv")


# ============================================================
# LOAD FINAL ML PIPELINE
# ============================================================

ml_pipeline = joblib.load("supply_chain_ml_pipeline.pkl")


# ============================================================
# SHIPMENT SIZE BINS
# ============================================================

_, shipment_bins = pd.qcut(
    feature_data["units"],
    q=4,
    retbins=True,
    duplicates="drop"
)


# ============================================================
# REQUEST MODELS
# ============================================================

class DisruptionRequest(BaseModel):
    hub: str


class RecoveryRequest(BaseModel):
    hub: str
    customer: str


class PredictionRequest(BaseModel):
    origin_port: str
    three_pl: str
    customs_procedures: str
    logistic_hub: str | None = None
    customer: str

    units: int
    weight: float
    weight_class: int
    material_handling: str

    distance: float | None = None
    cost_per_unit: float | None = None
    co2_per_unit: float | None = None

    product_data_missing: int = 0
    route_data_missing: int = 0


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "SupplyChain Nexus API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ml_pipeline": "loaded"
    }


# ============================================================
# NETWORK ENDPOINT
# ============================================================

@app.get("/network")
def network_summary():

    return {
        "nodes": 42,
        "connections": 343,
        "ports": 5,
        "hubs": 9,
        "customers": 28
    }


# ============================================================
# HUBS
# ============================================================

@app.get("/hubs")
def get_hubs():

    hubs = sorted(
        feature_data["logistic_hub"]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "hubs": hubs
    }


# ============================================================
# CUSTOMERS
# ============================================================

@app.get("/customers")
def get_customers():

    customers = sorted(
        feature_data["customer"]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "customers": customers
    }


# ============================================================
# DISRUPTION ANALYSIS
# ============================================================

@app.post("/disruption/analyze")
def analyze_disruption(request: DisruptionRequest):

    hub = request.hub

    result = disruption_data[
        disruption_data["logistic_hub"] == hub
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="Hub not found"
        )

    row = result.iloc[0]

    affected = propagation_data[
        propagation_data["disrupted_hub"] == hub
    ]

    return {
        "disrupted_hub": hub,

        "orders_affected": int(row["orders_affected"]),
        "units_affected": int(row["units_affected"]),
        "late_order_rate": float(row["late_order_rate"]),

        "affected_customers": int(
            affected["customer"].nunique()
        )
    }


# ============================================================
# RECOVERY OPTIONS
# ============================================================

@app.post("/recovery/options")
def recovery_options(request: RecoveryRequest):

    result = recovery_data[
        (recovery_data["disrupted_hub"] == request.hub)
        &
        (recovery_data["customer"] == request.customer)
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="No recovery options found"
        )

    result = result.sort_values(
        "recovery_score",
        ascending=False
    )

    options = []

    for _, row in result.iterrows():

        options.append({
            "alternative_hub": row["alternative_hub"],
            "coverage": float(row["coverage"]),
            "distance": float(row["weighted_distance"]),
            "cost": float(row["weighted_cost"]),
            "late_rate": float(row["historical_late_rate"]),
            "co2": float(row["weighted_co2"]),
            "recovery_score": float(row["recovery_score"])
        })

    return {
        "disrupted_hub": request.hub,
        "customer": request.customer,
        "options": options
    }


# ============================================================
# RECOVERY OPTIMIZATION
# ============================================================

@app.post("/recovery/optimize")
def optimize_recovery(request: RecoveryRequest):

    result = optimization_data[
        (optimization_data["disrupted_hub"] == request.hub)
        &
        (optimization_data["customer"] == request.customer)
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="No optimization result found"
        )

    row = result.iloc[0]

    return {
        "disrupted_hub": request.hub,
        "customer": request.customer,
        "selected_alternative_hub": row["selected_alternative_hub"],
        "optimization_score": float(row["optimization_score"])
    }


# ============================================================
# ML PREDICTION
# ============================================================

@app.post("/prediction")
def predict_late_order(request: PredictionRequest):

    # --------------------------------------------------------
    # 1. Route information
    # --------------------------------------------------------

    route_data_available = int(
        request.distance is not None
    )

    direct_shipment = int(
        request.logistic_hub is None
    )

    distance = (
        request.distance
        if request.distance is not None
        else np.nan
    )

    cost_per_unit = (
        request.cost_per_unit
        if request.cost_per_unit is not None
        else np.nan
    )

    co2_per_unit = (
        request.co2_per_unit
        if request.co2_per_unit is not None
        else np.nan
    )

    # --------------------------------------------------------
    # 2. Engineered features
    # --------------------------------------------------------

    estimated_route_cost = (
        request.units * cost_per_unit
        if not np.isnan(cost_per_unit)
        else np.nan
    )

    estimated_co2 = (
        request.units * co2_per_unit
        if not np.isnan(co2_per_unit)
        else np.nan
    )

    weight_per_unit = request.weight

    total_weight = (
        request.weight * request.units
    )

    log_units = np.log1p(
        request.units
    )

    log_distance = (
        np.log1p(distance)
        if not np.isnan(distance)
        else np.nan
    )

    log_total_weight = np.log1p(
        total_weight
    )

    # --------------------------------------------------------
    # 3. Distance category
    # --------------------------------------------------------

    if pd.isna(distance):
        distance_category = np.nan

    elif distance <= 500:
        distance_category = "Short"

    elif distance <= 1000:
        distance_category = "Medium"

    elif distance <= 2000:
        distance_category = "Long"

    else:
        distance_category = "Very_Long"

    # --------------------------------------------------------
    # 4. Shipment size
    # --------------------------------------------------------

    if request.units <= shipment_bins[1]:
        shipment_size = "Small"

    elif request.units <= shipment_bins[2]:
        shipment_size = "Medium"

    elif request.units <= shipment_bins[3]:
        shipment_size = "Large"

    else:
        shipment_size = "Very_Large"

    # --------------------------------------------------------
    # 5. Create model input
    # --------------------------------------------------------

    model_input = pd.DataFrame([{

        "origin_port": request.origin_port,
        "3pl": request.three_pl,
        "customs_procedures": request.customs_procedures,
        "logistic_hub": request.logistic_hub,
        "customer": request.customer,

        "material_handling": request.material_handling,
        "weight_class": request.weight_class,

        "units": request.units,
        "weight": request.weight,
        "distance": distance,

        "cost_per_unit": cost_per_unit,
        "co2_per_unit": co2_per_unit,

        "route_data_available": route_data_available,
        "direct_shipment": direct_shipment,

        "estimated_route_cost": estimated_route_cost,
        "estimated_co2": estimated_co2,

        "weight_per_unit": weight_per_unit,
        "total_weight": total_weight,

        "log_units": log_units,
        "log_distance": log_distance,
        "log_total_weight": log_total_weight,

        "product_data_missing": request.product_data_missing,
        "route_data_missing": request.route_data_missing,

        "distance_category": distance_category,
        "shipment_size": shipment_size

    }])

    # --------------------------------------------------------
    # 6. Predict using FINAL PIPELINE
    # --------------------------------------------------------

    probability = ml_pipeline.predict_proba(
        model_input
    )[0][1]

    # Current operating threshold
    threshold = 0.35

    if probability >= threshold:
        risk_level = "High"

    else:
        risk_level = "Low"

    # --------------------------------------------------------
    # 7. Response
    # --------------------------------------------------------

    return {

        "late_probability": float(probability),

        "late_probability_percentage": round(
            float(probability) * 100,
            2
        ),

        "risk_level": risk_level,

        "threshold_used": threshold,

        "engineered_features": {

            "route_data_available":
                route_data_available,

            "direct_shipment":
                direct_shipment,

            "estimated_route_cost":
                None if pd.isna(estimated_route_cost)
                else float(estimated_route_cost),

            "estimated_co2":
                None if pd.isna(estimated_co2)
                else float(estimated_co2),

            "total_weight":
                float(total_weight),

            "distance_category":
                distance_category,

            "shipment_size":
                shipment_size
        }
    }
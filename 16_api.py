from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd


app = FastAPI(
    title="SupplyChain Nexus API",
    description="Backend API for supply-chain disruption analysis and recovery planning",
    version="1.0"
)


# ============================================================
# LOAD PROJECT DATA
# ============================================================

disruption_data = pd.read_csv("hub_disruption_impact.csv")
recovery_data = pd.read_csv("recovery_option_results.csv")
optimization_data = pd.read_csv("recovery_optimization_results.csv")


# ============================================================
# REQUEST MODELS
# ============================================================

class DisruptionRequest(BaseModel):
    hub: str


class RecoveryRequest(BaseModel):
    hub: str
    customer: str


class PredictionRequest(BaseModel):
    probability: float


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "message": "SupplyChain Nexus API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# ============================================================
# NETWORK INFORMATION
# ============================================================

@app.get("/network")
def network_info():
    return {
        "nodes": 42,
        "connections": 343,
        "ports": 5,
        "logistic_hubs": 9,
        "customers": 28
    }


# ============================================================
# LIST AVAILABLE HUBS
# ============================================================

@app.get("/hubs")
def get_hubs():

    hubs = sorted(
        disruption_data["disrupted_hub"]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "hubs": hubs,
        "count": len(hubs)
    }


# ============================================================
# LIST CUSTOMERS
# ============================================================

@app.get("/customers")
def get_customers():

    customers = sorted(
        recovery_data["customer"]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "customers": customers,
        "count": len(customers)
    }


# ============================================================
# DISRUPTION ANALYSIS
# ============================================================

@app.post("/disruption/analyze")
def analyze_disruption(request: DisruptionRequest):

    hub = request.hub.strip()

    result = disruption_data[
        disruption_data["disrupted_hub"].str.lower() == hub.lower()
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Hub '{hub}' not found"
        )

    row = result.iloc[0]

    return {
        "disrupted_hub": row["disrupted_hub"],
        "affected_orders": int(row["affected_orders"]),
        "affected_units": int(row["affected_units"]),
        "affected_customers": int(row["affected_customers"]),
        "late_orders": int(row["late_orders"]),
        "late_order_rate": float(row["late_order_rate"]),
        "estimated_route_cost": float(row["estimated_route_cost"]),
        "estimated_co2": float(row["estimated_co2"])
    }


# ============================================================
# RECOVERY OPTIONS
# ============================================================

@app.post("/recovery/options")
def recovery_options(request: RecoveryRequest):

    hub = request.hub.strip()
    customer = request.customer.strip()

    result = recovery_data[
        (recovery_data["disrupted_hub"].str.lower() == hub.lower())
        &
        (recovery_data["customer"].str.lower() == customer.lower())
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="No recovery options found for this hub and customer"
        )

    options = []

    for _, row in result.iterrows():

        options.append({
            "alternative_hub": row["alternative_hub"],
            "recovery_score": float(row["recovery_score"]),
            "alternative_cost_per_unit": float(
                row["alternative_cost_per_unit"]
            ),
            "alternative_distance": float(
                row["alternative_distance"]
            ),
            "alternative_historical_late_rate": float(
                row["alternative_historical_late_rate"]
            ),
            "route_unit_coverage": float(
                row["route_unit_coverage"]
            )
        })

    return {
        "disrupted_hub": hub,
        "customer": customer,
        "options": options,
        "option_count": len(options)
    }


# ============================================================
# OPTIMIZED RECOVERY OPTION
# ============================================================

@app.post("/recovery/optimize")
def optimize_recovery(request: RecoveryRequest):

    hub = request.hub.strip()
    customer = request.customer.strip()

    result = optimization_data[
        (optimization_data["disrupted_hub"].str.lower() == hub.lower())
        &
        (optimization_data["customer"].str.lower() == customer.lower())
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="No optimized recovery result found"
        )

    row = result.iloc[0]

    response = {}

    for column in optimization_data.columns:

        value = row[column]

        if pd.isna(value):
            response[column] = None

        elif hasattr(value, "item"):
            response[column] = value.item()

        else:
            response[column] = value

    return response


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/prediction")
def prediction(request: PredictionRequest):

    probability = request.probability

    if probability < 0 or probability > 1:
        raise HTTPException(
            status_code=400,
            detail="Probability must be between 0 and 1"
        )

    if probability >= 0.35:
        risk = "High"
    else:
        risk = "Low"

    return {
        "late_probability": probability,
        "risk_level": risk,
        "threshold_used": 0.35
    }
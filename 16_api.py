from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SupplyChain Nexus API",
    description="ML-driven supply-chain disruption and recovery decision-support API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD DATA
# ============================================================

FEATURE_FILE = "feature_engineered_supply_chain_data.csv"
HUB_IMPACT_FILE = "hub_disruption_impact.csv"
PROPAGATION_FILE = "disruption_propagation_results.csv"
RECOVERY_FILE = "recovery_option_results.csv"
OPTIMIZATION_FILE = "recovery_optimization_results.csv"
ML_PIPELINE_FILE = "supply_chain_ml_pipeline.pkl"


try:
    feature_data = pd.read_csv(FEATURE_FILE)
    hub_impact = pd.read_csv(HUB_IMPACT_FILE)
    propagation_data = pd.read_csv(PROPAGATION_FILE)
    recovery_data = pd.read_csv(RECOVERY_FILE)
    optimization_data = pd.read_csv(OPTIMIZATION_FILE)

    ml_pipeline = joblib.load(ML_PIPELINE_FILE)

    print("All SupplyChain Nexus data and ML pipeline loaded successfully.")

except Exception as e:
    print("ERROR loading project files:")
    print(e)

    feature_data = pd.DataFrame()
    hub_impact = pd.DataFrame()
    propagation_data = pd.DataFrame()
    recovery_data = pd.DataFrame()
    optimization_data = pd.DataFrame()
    ml_pipeline = None


# ============================================================
# SHIPMENT SIZE BINS
# ============================================================

if not feature_data.empty and "units" in feature_data.columns:
    try:
        _, shipment_bins = pd.qcut(
            feature_data["units"],
            q=4,
            retbins=True,
            duplicates="drop",
        )
    except Exception:
        shipment_bins = np.array([0, 250, 500, 750, np.inf])
else:
    shipment_bins = np.array([0, 250, 500, 750, np.inf])


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
    logistic_hub: Optional[str] = None
    customer: str
    units: float
    weight: Optional[float] = None
    weight_class: Optional[float] = None
    material_handling: Optional[str] = None
    distance: Optional[float] = None
    cost_per_unit: Optional[float] = None
    co2_per_unit: Optional[float] = None
    product_data_missing: int = 0
    route_data_missing: int = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """
    Convert NumPy/Pandas values into JSON-safe Python values.
    """
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    return value


def clean_records(df):
    """
    Convert DataFrame to JSON-safe records.
    """
    if df.empty:
        return []

    records = df.replace([np.inf, -np.inf], np.nan).to_dict(
        orient="records"
    )

    cleaned = []

    for record in records:
        cleaned_record = {
            key: clean_value(value)
            for key, value in record.items()
        }

        cleaned.append(cleaned_record)

    return cleaned


def get_unique_values(column):
    """
    Return sorted unique values from a column.
    """
    if feature_data.empty or column not in feature_data.columns:
        return []

    values = (
        feature_data[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(values)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "application": "SupplyChain Nexus",
        "status": "API running",
        "version": "1.0.0",
        "description": "Disruption Intelligence & Recovery Planning",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ml_pipeline_loaded": ml_pipeline is not None,
        "feature_data_loaded": not feature_data.empty,
        "network_data_loaded": not propagation_data.empty,
    }


# ============================================================
# NETWORK
# ============================================================

@app.get("/network")
def get_network():

    if feature_data.empty:
        raise HTTPException(
            status_code=500,
            detail="Feature data could not be loaded.",
        )

    ports = get_unique_values("origin_port")
    hubs = get_unique_values("logistic_hub")
    customers = get_unique_values("customer")

    # Build directed network connections
    connections = set()

    for _, row in feature_data.iterrows():

        port = row.get("origin_port")
        hub = row.get("logistic_hub")
        customer = row.get("customer")

        if pd.notna(port) and pd.notna(hub):
            connections.add(
                (
                    str(port),
                    str(hub),
                )
            )

            connections.add(
                (
                    str(hub),
                    str(customer),
                )
            )

        elif pd.notna(port) and pd.notna(customer):
            connections.add(
                (
                    str(port),
                    str(customer),
                )
            )

    connection_records = [
        {
            "source": source,
            "target": target,
        }
        for source, target in sorted(connections)
    ]

    nodes = []

    for port in ports:
        nodes.append(
            {
                "id": port,
                "name": port,
                "type": "Port",
            }
        )

    for hub in hubs:
        nodes.append(
            {
                "id": hub,
                "name": hub,
                "type": "Hub",
            }
        )

    for customer in customers:
        nodes.append(
            {
                "id": customer,
                "name": customer,
                "type": "Customer",
            }
        )

    return {
        "nodes": nodes,
        "connections": connection_records,
        "summary": {
            "nodes": len(nodes),
            "connections": len(connection_records),
            "ports": len(ports),
            "hubs": len(hubs),
            "customers": len(customers),
        },
    }


# ============================================================
# HUBS
# ============================================================

@app.get("/hubs")
def get_hubs():

    if feature_data.empty:
        raise HTTPException(
            status_code=500,
            detail="Feature data could not be loaded.",
        )

    result = []

    hubs = (
        feature_data["logistic_hub"]
        .dropna()
        .unique()
        .tolist()
    )

    for hub in sorted(hubs):

        hub_df = feature_data[
            feature_data["logistic_hub"] == hub
        ]

        result.append(
            {
                "hub": hub,
                "orders": int(len(hub_df)),
                "units": int(hub_df["units"].sum()),
                "late_rate": round(
                    float(hub_df["late_order"].mean()),
                    4,
                ),
                "customers": int(
                    hub_df["customer"].nunique()
                ),
            }
        )

    return {
        "count": len(result),
        "hubs": result,
    }


# ============================================================
# CUSTOMERS
# ============================================================

@app.get("/customers")
def get_customers():

    if feature_data.empty:
        raise HTTPException(
            status_code=500,
            detail="Feature data could not be loaded.",
        )

    result = []

    customers = (
        feature_data["customer"]
        .dropna()
        .unique()
        .tolist()
    )

    for customer in sorted(customers):

        customer_df = feature_data[
            feature_data["customer"] == customer
        ]

        result.append(
            {
                "customer": customer,
                "orders": int(len(customer_df)),
                "units": int(customer_df["units"].sum()),
                "late_rate": round(
                    float(customer_df["late_order"].mean()),
                    4,
                ),
            }
        )

    return {
        "count": len(result),
        "customers": result,
    }


# ============================================================
# DISRUPTION ANALYSIS
# ============================================================

@app.post("/disruption/analyze")
def analyze_disruption(request: DisruptionRequest):

    hub = request.hub

    if feature_data.empty:
        raise HTTPException(
            status_code=500,
            detail="Feature data could not be loaded.",
        )

    if hub not in feature_data["logistic_hub"].dropna().unique():
        raise HTTPException(
            status_code=404,
            detail=f"Hub '{hub}' was not found.",
        )

    disrupted_orders = feature_data[
        feature_data["logistic_hub"] == hub
    ]

    affected_orders = len(disrupted_orders)
    affected_units = int(
        disrupted_orders["units"].sum()
    )

    affected_customers = sorted(
        disrupted_orders["customer"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    late_rate = float(
        disrupted_orders["late_order"].mean()
    )

    # Find alternative hubs for affected customers
    alternatives = {}

    for customer in affected_customers:

        customer_df = feature_data[
            feature_data["customer"] == customer
        ]

        alternative_hubs = sorted(
            customer_df[
                customer_df["logistic_hub"].notna()
                & (
                    customer_df["logistic_hub"]
                    != hub
                )
            ]["logistic_hub"]
            .astype(str)
            .unique()
            .tolist()
        )

        alternatives[customer] = alternative_hubs

    return {
        "disrupted_hub": hub,
        "affected_orders": int(affected_orders),
        "affected_units": affected_units,
        "affected_customers": len(affected_customers),
        "affected_customer_names": affected_customers,
        "historical_late_rate": round(late_rate, 4),
        "historical_late_rate_percentage": round(
            late_rate * 100,
            2,
        ),
        "alternative_hubs": alternatives,
        "message": (
            f"Disruption at {hub} affects "
            f"{affected_orders:,} orders and "
            f"{affected_units:,} units."
        ),
    }


# ============================================================
# RECOVERY OPTIONS
# ============================================================

@app.post("/recovery/options")
def recovery_options(request: RecoveryRequest):

    hub = request.hub
    customer = request.customer

    if recovery_data.empty:
        raise HTTPException(
            status_code=500,
            detail="Recovery data could not be loaded.",
        )

    df = recovery_data.copy()

    # Support the actual column naming used by the project.
    hub_column = None
    customer_column = None

    for column in [
        "disrupted_hub",
        "hub",
    ]:
        if column in df.columns:
            hub_column = column
            break

    for column in [
        "customer",
        "customer_name",
    ]:
        if column in df.columns:
            customer_column = column
            break

    if hub_column is None or customer_column is None:
        raise HTTPException(
            status_code=500,
            detail="Recovery dataset columns could not be identified.",
        )

    filtered = df[
        (df[hub_column].astype(str) == str(hub))
        & (df[customer_column].astype(str) == str(customer))
    ].copy()

    if filtered.empty:
        return {
            "disrupted_hub": hub,
            "customer": customer,
            "count": 0,
            "options": [],
        }

    # Sort using optimization score when available.
    if "optimization_score" in filtered.columns:
        filtered = filtered.sort_values(
            "optimization_score",
            ascending=False,
        )

    # Return top useful alternatives.
    filtered = filtered.head(10)

    return {
        "disrupted_hub": hub,
        "customer": customer,
        "count": len(filtered),
        "options": clean_records(filtered),
    }


# ============================================================
# RECOVERY OPTIMIZATION
# ============================================================

@app.post("/recovery/optimize")
def recovery_optimize(request: RecoveryRequest):

    hub = request.hub
    customer = request.customer

    if optimization_data.empty:
        raise HTTPException(
            status_code=500,
            detail="Optimization data could not be loaded.",
        )

    df = optimization_data.copy()

    if "disrupted_hub" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Optimization dataset is missing disrupted_hub.",
        )

    if "customer" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Optimization dataset is missing customer.",
        )

    filtered = df[
        (df["disrupted_hub"].astype(str) == str(hub))
        & (df["customer"].astype(str) == str(customer))
    ].copy()

    if filtered.empty:
        return {
            "disrupted_hub": hub,
            "customer": customer,
            "selected_option": None,
            "message": "No optimization result found.",
        }

    if "optimization_score" in filtered.columns:
        selected = filtered.sort_values(
            "optimization_score",
            ascending=False,
        ).iloc[0]
    else:
        selected = filtered.iloc[0]

    return {
        "disrupted_hub": hub,
        "customer": customer,
        "selected_option": {
            key: clean_value(value)
            for key, value in selected.to_dict().items()
        },
        "message": (
            "Recovery optimization result returned "
            "for decision-support evaluation."
        ),
    }


# ============================================================
# ML PREDICTION
# ============================================================

@app.post("/prediction")
def prediction(request: PredictionRequest):

    if ml_pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="ML pipeline is not loaded.",
        )

    # --------------------------------------------------------
    # Basic values
    # --------------------------------------------------------

    units = float(request.units)

    weight = (
        float(request.weight)
        if request.weight is not None
        else np.nan
    )

    distance = (
        float(request.distance)
        if request.distance is not None
        else np.nan
    )

    cost_per_unit = (
        float(request.cost_per_unit)
        if request.cost_per_unit is not None
        else np.nan
    )

    co2_per_unit = (
        float(request.co2_per_unit)
        if request.co2_per_unit is not None
        else np.nan
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    route_data_available = int(
        pd.notna(distance)
    )

    direct_shipment = int(
        request.logistic_hub is None
        or str(request.logistic_hub).strip() == ""
    )

    estimated_route_cost = (
        units * cost_per_unit
        if pd.notna(cost_per_unit)
        else np.nan
    )

    estimated_co2 = (
        units * co2_per_unit
        if pd.notna(co2_per_unit)
        else np.nan
    )

    weight_per_unit = weight

    total_weight = (
        weight * units
        if pd.notna(weight)
        else np.nan
    )

    log_units = np.log1p(units)

    log_distance = (
        np.log1p(distance)
        if pd.notna(distance)
        else np.nan
    )

    log_total_weight = (
        np.log1p(total_weight)
        if pd.notna(total_weight)
        else np.nan
    )

    # --------------------------------------------------------
    # Distance category
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
    # Shipment size
    # --------------------------------------------------------

    try:
        shipment_size = pd.cut(
            [units],
            bins=shipment_bins,
            labels=[
                "Small",
                "Medium",
                "Large",
                "Very_Large",
            ],
            include_lowest=True,
        )[0]
    except Exception:
        shipment_size = np.nan

    # --------------------------------------------------------
    # Create exact model input
    # --------------------------------------------------------

    model_input = pd.DataFrame(
        [
            {
                "origin_port": request.origin_port,
                "3pl": request.three_pl,
                "customs_procedures": request.customs_procedures,
                "logistic_hub": (
                    request.logistic_hub
                    if request.logistic_hub
                    else np.nan
                ),
                "customer": request.customer,
                "units": units,
                "weight": weight,
                "distance": distance,
                "cost_per_unit": cost_per_unit,
                "co2_per_unit": co2_per_unit,
                "material_handling": request.material_handling,
                "weight_class": request.weight_class,
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
                "shipment_size": shipment_size,
            }
        ]
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    try:
        probability = float(
            ml_pipeline.predict_proba(model_input)[0][1]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    # --------------------------------------------------------
    # Business threshold
    # --------------------------------------------------------

    threshold = 0.35

    if probability >= threshold:
        risk_level = "High"
    elif probability >= 0.20:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "late_probability": probability,
        "late_probability_percentage": round(
            probability * 100,
            2,
        ),
        "risk_level": risk_level,
        "threshold_used": threshold,
        "model": {
            "name": "Logistic Regression",
            "roc_auc": 0.8035,
        },
        "engineered_features": {
            "route_data_available": route_data_available,
            "direct_shipment": direct_shipment,
            "estimated_route_cost": clean_value(
                estimated_route_cost
            ),
            "estimated_co2": clean_value(
                estimated_co2
            ),
            "total_weight": clean_value(
                total_weight
            ),
            "distance_category": clean_value(
                distance_category
            ),
            "shipment_size": clean_value(
                shipment_size
            ),
        },
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model")
def model_information():

    return {
        "model": "Logistic Regression",
        "purpose": "Late-order risk prediction",
        "roc_auc": 0.8035,
        "operating_threshold": 0.35,
        "features": 25,
        "processed_features": 89,
        "training_rows": 91420,
        "testing_rows": 22856,
    }


# ============================================================
# ANALYTICS SUMMARY
# ============================================================

@app.get("/analytics")
def analytics():

    if feature_data.empty:
        raise HTTPException(
            status_code=500,
            detail="Feature data could not be loaded.",
        )

    total_orders = len(feature_data)

    late_orders = int(
        feature_data["late_order"].sum()
    )

    total_units = int(
        feature_data["units"].sum()
    )

    late_rate = (
        late_orders / total_orders
        if total_orders > 0
        else 0
    )

    return {
        "total_orders": total_orders,
        "late_orders": late_orders,
        "on_time_orders": total_orders - late_orders,
        "late_rate": round(late_rate, 4),
        "late_rate_percentage": round(
            late_rate * 100,
            2,
        ),
        "total_units": total_units,
        "network_nodes": 42,
        "network_connections": 343,
        "ports": 5,
        "hubs": 9,
        "customers": 28,
    }


# ============================================================
# STARTUP MESSAGE
# ============================================================

@app.on_event("startup")
def startup_message():

    print()
    print("=" * 60)
    print("SUPPLYCHAIN NEXUS API")
    print("=" * 60)

    print(
        f"Feature data loaded: {not feature_data.empty}"
    )

    print(
        f"ML pipeline loaded: {ml_pipeline is not None}"
    )

    print(
        f"Recovery data loaded: {not recovery_data.empty}"
    )

    print(
        f"Optimization data loaded: "
        f"{not optimization_data.empty}"
    )

    print("=" * 60)
    print("API READY")
    print("=" * 60)
    print()
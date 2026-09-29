import pandas as pd
import joblib


# Load model and preprocessor
model = joblib.load("logistic_regression_model.pkl")
preprocessor = joblib.load("supply_chain_preprocessor.pkl")

print("Model loaded.")
print("Preprocessor loaded.")


# Load original data
df = pd.read_csv("feature_engineered_supply_chain_data.csv")


# Take one shipment
sample = df.iloc[[0]].copy()

# Remove columns not used by the model
columns_to_remove = [
    "order_id",
    "product_id",
    "hub_order_volume",
    "customer_order_volume",
    "three_pl_order_volume",
    "late_order"
]

sample = sample.drop(columns=columns_to_remove)


# Preprocess
sample_processed = preprocessor.transform(sample)

print("Sample processed successfully.")
print("Processed shape:", sample_processed.shape)


# Predict probability
probability = model.predict_proba(sample_processed)[0][1]

print("Predicted late probability:", probability)


# Risk level
if probability >= 0.35:
    risk = "High"
else:
    risk = "Low"

print("Risk level:", risk)
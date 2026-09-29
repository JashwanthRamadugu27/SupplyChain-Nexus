import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression


# ============================================================
# LOAD PROCESSED TRAINING DATA
# ============================================================

X_train = pd.read_csv("X_train_processed.csv")
y_train = pd.read_csv("y_train.csv").squeeze()


print("Training data:", X_train.shape)
print("Target data:", y_train.shape)


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)


print("Model training completed.")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "logistic_regression_model.pkl"
)

print("Model saved as logistic_regression_model.pkl")
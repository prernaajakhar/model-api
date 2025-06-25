import os
import io
import uuid
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = FastAPI()
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Model selection factory with hyperparameters ---
def get_model(model_type: str, params: dict = {}):
    model_type = model_type.lower()
    if model_type == "xgboost":
        return XGBRegressor(**params)
    elif model_type == "randomforest":
        return RandomForestRegressor(**params)
    elif model_type == "bayesianridge":
        return BayesianRidge(**params)
    elif model_type == "linear":
        params.pop("normalize", None)  # remove unsupported param
        return LinearRegression(**params)
    else:
        raise ValueError("Unsupported model type")

# --- API endpoint to evaluate model ---
@app.post("/evaluate/")
async def evaluate_model(
    file_data: UploadFile = File(...),
    model_type: str = Form(...),
    n_estimators: int = Form(100),         # for RF and XGB
    max_depth: int = Form(3),              # for RF and XGB
    alpha: float = Form(1.0),              # for BayesianRidge
    normalize: bool = Form(False)          # for Linear/Bayesian
):
    contents = await file_data.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if "yield" not in df.columns:
        return {"error": "Missing 'yield' column in uploaded file."}

    # Prepare features and target
    df = df.dropna(axis=1, how='all')           # Drop columns that are all NaN
    X = df.drop(columns=["yield"])
    y = df["yield"]

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Fill missing numeric values with mean
    X = X.fillna(X.mean(numeric_only=True))

    # Ensure valid input
    if X.shape[0] == 0:
        return {"error": "No data available for training after preprocessing."}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose model
    try:
        params = {}
        if model_type.lower() in ["randomforest", "xgboost"]:
            params = {"n_estimators": n_estimators, "max_depth": max_depth}
        elif model_type.lower() == "bayesianridge":
            params = {"alpha_1": alpha}
        elif model_type.lower() == "linear":
            params = {"normalize": normalize}

        model = get_model(model_type, params)
    except ValueError as e:
        return {"error": str(e)}

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save graph
    fig_id = str(uuid.uuid4())
    fig_path = os.path.join(RESULTS_DIR, f"{fig_id}_pred_vs_actual.png")
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title("Predicted vs Actual Yield")
    plt.savefig(fig_path)
    plt.close()

    # Save model
    model_path = os.path.join(RESULTS_DIR, f"{fig_id}_model.pkl")
    joblib.dump(model, model_path)

    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "Graph": f"/graph/{fig_id}",
        "Model": f"/download_model/{fig_id}"
    }

# --- Endpoint to serve graph ---
@app.get("/graph/{graph_id}")
def get_graph(graph_id: str):
    path = os.path.join(RESULTS_DIR, f"{graph_id}_pred_vs_actual.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return {"error": "Graph not found"}

# --- Endpoint to download model ---
@app.get("/download_model/{model_id}")
def download_model(model_id: str):
    path = os.path.join(RESULTS_DIR, f"{model_id}_model.pkl")
    if os.path.exists(path):
        return FileResponse(path, media_type="application/octet-stream", filename=f"{model_id}_model.pkl")
    return {"error": "Model file not found"}








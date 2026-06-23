import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"   # must come before mlflow import

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost

app = FastAPI()

scaler = joblib.load(r"D:\XGBoost - Project\models\scaler.joblib")

# Use the exact run ID from your best tuned model (replace with your actual ID)
RUN_ID = "4da0253b94a742ed849b6f6964157e9f"
model = mlflow.xgboost.load_model(f"runs:/{RUN_ID}/xgboost_model")

class InputData(BaseModel):
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: int
    Torque_Nm: float
    Tool_wear_min: int

@app.post("/predict")
def predict(data: InputData):
    input_dict = {
        "Air_temperature_K": data.Air_temperature_K,
        "Process_temperature_K": data.Process_temperature_K,
        "Rotational_speed_rpm": data.Rotational_speed_rpm,
        "Torque_Nm": data.Torque_Nm,
        "Tool_wear_min": data.Tool_wear_min,
        "power": data.Torque_Nm * data.Rotational_speed_rpm
    }
    X = pd.DataFrame([input_dict])
    X_scaled = scaler.transform(X)
    proba = float(model.predict_proba(X_scaled)[0, 1])
    prediction = int(proba > 0.5)
    return {"probability": proba, "prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
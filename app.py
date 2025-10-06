from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import traceback
import os

app = FastAPI(title="LG Pricing Optimization API")

#absolute path for model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pricing_model.pkl")

#safe model loading
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print("Failed to load model.")
        print("Error details:", str(e))
        traceback.print_exc()
else:
    print(f"Model file not found at {MODEL_PATH}")

#request schema for prediction
class PricingRequest(BaseModel):
    LGPrice: float
    CompetitorPrice: float

#root route
@app.get("/")
def root():
    return {
        "message": "Running",
        "model_loaded": model is not None
    }

#health route
@app.get("/health")
def health_check():
    return {
        "status": "running",
        "model_loaded": model is not None
    }

#predict route
@app.post("/predict")
def predict(request: PricingRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        prediction = model.predict([[request.LGPrice, request.CompetitorPrice]])
        return {"predicted_spend": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import traceback
import os

app = FastAPI(title="LG Pricing Optimization API")

# Safe model loading with logging
model = None
MODEL_PATH = "pricing_model.pkl"

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully from", MODEL_PATH)
    except Exception as e:
        print("❌ Failed to load model.")
        print("Error details:", str(e))
        traceback.print_exc()
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}. Please check deployment files.")


# Request schema for prediction
class PricingRequest(BaseModel):
    lg_price: float
    competitor_price: float


@app.get("/")
def root():
    return {
        "message": "LG Pricing Optimization API is running 🚀",
        "status": "ready" if model else "model not loaded",
    }


@app.post("/predict")
def predict(data: PricingRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please redeploy with a valid pricing_model.pkl.")

    try:
        prediction = model.predict([[data.lg_price, data.competitor_price]])[0]
        return {
            "LGPrice": data.lg_price,
            "CompetitorPrice": data.competitor_price,
            "PredictedSpend": round(float(prediction), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Optional endpoint to verify deployment health
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

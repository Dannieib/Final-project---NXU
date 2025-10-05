import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import traceback

app = FastAPI(title="LG Pricing Optimization API")

# Absolute model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pricing_model.pkl")

model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print("❌ Failed to load model.")
        print("Error details:", str(e))
        traceback.print_exc()
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}. Please check deployment.")

from fastapi import FastAPI
import joblib

# Load model
model = joblib.load("pricing_model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "LG Pricing Optimization Model is running!"}

@app.get("/predict")
def predict(lg_price: float, competitor_price: float):
    predicted_spend = model.predict([[lg_price, competitor_price]])[0]
    return {
        "lg_price": lg_price,
        "competitor_price": competitor_price,
        "predicted_spend": predicted_spend
    }
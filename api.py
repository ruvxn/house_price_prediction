from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI()

# Allow frontend to access the backend (CORS policy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any frontend (React, Vue, etc.)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = tf.keras.models.load_model("house_price_model.keras")

# Define request body format
class HouseData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: HouseData):
    # Convert input data to numpy array
    input_array = np.array([data.features])
    prediction = model.predict(input_array)[0][0]
    
    # Return JSON response
    return {"predicted_price": float(prediction)}

# Root endpoint
@app.get("/")
def root():
    return {"message": "House Price Prediction API is running!"}

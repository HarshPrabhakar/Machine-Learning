from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import pickle

app = FastAPI()

# Allow browser requests from any origin (needed for the UI to call this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load scaler ──────────────────────────────────────────────
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ── Load model weights ───────────────────────────────────────
model_data = torch.load("model.pt", weights_only=False)
weights = model_data["weights"]
bias    = model_data["bias"]

# ── Input schema (30 feature values) ────────────────────────
class InputData(BaseModel):
    features: list[float]

# ── Health check — open http://localhost:8000 to confirm ─────
@app.get("/")
def root():
    return {"status": "✅ Breast Cancer NN API is running!"}

# ── Prediction endpoint ──────────────────────────────────────
@app.post("/predict")
def predict(data: InputData):
    if len(data.features) != 30:
        return {"error": f"Expected 30 features, got {len(data.features)}"}

    x        = np.array(data.features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_tensor = torch.from_numpy(x_scaled)

    with torch.no_grad():
        z    = torch.matmul(x_tensor, weights) + bias
        prob = torch.sigmoid(z).item()

    label      = "M" if prob > 0.5 else "B"
    confidence = round(max(prob, 1 - prob) * 100, 1)

    return {
        "prediction": label,
        "confidence": confidence,
        "probability": round(prob, 4),
        "reasoning": (
            f"The model assigned a malignancy probability of {round(prob*100,1)}%. "
            f"Key features drove this {'above' if prob>0.5 else 'below'} the 0.5 decision threshold."
        )
    }
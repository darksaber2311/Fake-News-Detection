import joblib
from fastapi import FastAPI

app = FastAPI()

# Load pre-trained model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.post("/predict")
async def predict(text: str):
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    return {"prediction": "Fake News" if prediction == 1 else "Real News"}

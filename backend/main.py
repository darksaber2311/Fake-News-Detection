from fastapi import FastAPI, Body
from base64 import b64encode, b64decode
import joblib
from fastapi.middleware.cors import CORSMiddleware  # <- import this
#from fastapi.staticfiles import StaticFiles

from crypto_utils import sign_content, verify_signature, export_public_key
#from predict import predict_fake_news

app = FastAPI(title="News Trust Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & vectorizer once
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.get("/")
def root():
    return {"message": "Backend is running!"}


@app.get("/public_key")
def get_public_key():
    return {"public_key": export_public_key()}


@app.post("/sign")
def sign_endpoint(payload: dict = Body(...)):
    content = payload.get("content", "")
    signature = sign_content(content)
    return {
        "content": content,
        "signature": b64encode(signature).decode()
    }


@app.post("/verify")
def verify_endpoint(payload: dict = Body(...)):
    content = payload.get("content", "")
    signature_b64 = payload.get("signature", "")
    try:
        signature = b64decode(signature_b64)
    except Exception:
        return {"valid": False, "error": "Invalid signature encoding"}

    valid = verify_signature(content, signature)
    return {"content": content, "valid": valid}






@app.post("/predict")
def predict_endpoint(payload: dict = Body(...)):
    text = payload.get("text", "")
    if not text.strip():
        return {"error": "No text provided"}

    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    return {
        "text": text,
        "prediction": "Fake News" if prediction == 1 else "Real News",
        "confidence": float(max(prob))
    }
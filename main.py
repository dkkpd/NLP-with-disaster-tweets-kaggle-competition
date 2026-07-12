from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware
import torch

MODEL_NAME = "dkkpd/disaster-tweets-distilbert"
MAX_LENGTH = 128

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()  # Set the model to evaluation mode

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity; adjust as needed
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please try again later."}
    )

class TweetRequest(BaseModel):
    text: str  

@app.post("/predict")
@limiter.limit("50/minute")  # Limit to 50 requests per minute per IP
def predict(request: Request, tweet_request: TweetRequest):
    # Tokenize the input text
    inputs = tokenizer(tweet_request.text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

    return {
        "text": tweet_request.text,
        "predicted_class": "disaster" if predicted_class == 1 else "not disaster",
        "confidence": round(confidence, 4)
    }

@app.get("/")
def root():
    return {"message": "Disaster Tweet Classification API is running. Use the /predict endpoint to classify tweets."}

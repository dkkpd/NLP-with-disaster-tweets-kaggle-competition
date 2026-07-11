from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_NAME = "dkkpd/disaster-tweets-distilbert"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()  # Set the model to evaluation mode

app = FastAPI()

class TweetRequest(BaseModel):
    text: str  

@app.post("/predict")
def predict(tweet_request: TweetRequest):
    # Tokenize the input text
    inputs = tokenizer(tweet_request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
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

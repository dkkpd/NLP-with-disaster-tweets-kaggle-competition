# %% Cell 1: Load model and tokenizer from the Hub
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

MODEL_NAME = "dkkpd/disaster-tweets-distilbert"
model_v4 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

trainer_v4 = Trainer(model=model_v4)

print("Model and tokenizer loaded successfully!")

# %% Cell 2: Load and tokenize test data
import pandas as pd

test = pd.read_csv("data/test.csv")

test_encodings = tokenizer(
    test["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

print(f"Test set size: {len(test)}")

# %% Cell 3: Wrap in a Dataset
import torch

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

test_dataset = TestDataset(test_encodings)

# %% Cell 4: Predict
predictions = trainer_v4.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(axis=1)

print(predicted_labels[:20])

# %% Cell 5: Create submission file
submission = pd.DataFrame({
    "id": test["id"],
    "target": predicted_labels
})
submission.to_csv("submission.csv", index=False)
# %%

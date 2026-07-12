# Disaster Tweets Classification

Classifying tweets as real disaster reports vs. non-disaster, using the Kaggle 
"NLP with Disaster Tweets" competition dataset.

## How to Run This Project Locally

This project isn't currently deployed publicly — to try it, you'll run the API and 
frontend on your own machine. Here's how:

### 1. Clone the repo
```bash
git clone https://github.com/dkkpd/NLP-with-disaster-tweets-kaggle-competition.git
cd NLP-with-disaster-tweets-kaggle-competition
```

### 2. Set up a Python virtual environment
```bash
python -m venv venv
```
Activate it:
- Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
- Mac/Linux: `source venv/bin/activate`

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI backend
```bash
uvicorn main:app --reload
```
This starts the API at `http://127.0.0.1:8000`. On first run, it will automatically 
download the fine-tuned model from Hugging Face Hub 
([dkkpd/disaster-tweets-distilbert](https://huggingface.co/dkkpd/disaster-tweets-distilbert)) 
— no separate setup needed.

You can confirm it's running by visiting `http://127.0.0.1:8000/docs` in a browser, 
which shows interactive API documentation.

### 5. Open the frontend
Open `index.html` directly in a browser (or use a tool like VS Code's Live Server 
extension). Make sure the FastAPI server from Step 4 is still running in a separate 
terminal — the frontend depends on it.

### 6. Try it
Type a tweet into the text box and click "Classify" to see a live prediction from 
the fine-tuned model.

**Note:** The API includes rate limiting (50 requests/minute per IP) to prevent 
abuse — if you hit this while testing, just wait a minute and try again.

---

### Reproducing the model training (optional)
The fine-tuning process itself was done in Google Colab (free GPU access). The full 
notebook, including training output and experiment results, is available at 
`notebooks/Kaggle_NLP_Disaster_Tweets.ipynb` and can be re-run in Colab if you want 
to reproduce or modify the training process.

## Data Analysis
- Dataset is reasonably balanced: ~57% non-disaster, ~43% disaster tweets
- No missing values in the `text` or `target` columns (the two used for modeling)
- `location` field is ~33% missing and inconsistent (free text), so it wasn't used
- Word frequency analysis (after removing URLs and stopwords) shows a clear content 
  difference between classes — disaster tweets surface words like "fire," "disaster," 
  "california," and "police," while non-disaster tweets skew more casual/personal 
  ("i'm," "don't," "it's")
- This confirmed there's a learnable signal in the text before investing time in modeling

## Logistical Regression - Baseline Model
 
 A classical ML approach using logistical regression used to establish a benchmark before fine tuning the model.

 **Approach:**
- Vectorizing the text using TF-IDF using the top 5000 vocab
- Training logistal regression classifier on a 80% train 20% test split

**Results (on held-out validation set):**

| Metric | Class 0 (Not Disaster) | Class 1 (Disaster) | Overall |
|---|---|---|---|
| Precision | 0.79 | 0.82 | — |
| Recall | 0.88 | 0.69 | — |
| F1 Score | 0.84 | 0.74 | — |
| Accuracy | — | — | **0.80** |

**Key observation:** The model is more conservative on the "disaster" class — when it 
predicts disaster, it's usually correct (82% precision), but it misses roughly 31% of 
real disaster tweets (69% recall), classifying them as non-disaster instead. This 
represents a meaningful weakness for a real-world early-warning use case, and is a key 
benchmark the fine-tuned transformer model (see below) aims to improve on.

This baseline (80% accuracy, 0.74 F1 on the disaster class) serves as the number to 
beat with the fine-tuned DistilBERT model in the next phase.

## Fine-Tuned Model: DistilBERT

To try to beat the classical baseline, I fine-tuned a pretrained DistilBERT model 
(`distilbert-base-uncased`) on the exact same train/validation split I used for the 
baseline, so I could compare the two fairly.

**Approach:**
- Tokenized tweets using DistilBERT's tokenizer (max length 128 tokens)
- Added a classification head on top of the pretrained model (`num_labels=2`)
- Fine-tuned for 3 epochs using Hugging Face's `Trainer` API on a free Google Colab 
  T4 GPU (batch size 16, learning rate 2e-5)

**Training results by epoch:**

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
|---|---|---|---|---|
| 1 | 0.414 | 0.379 | 0.845 | 0.810 |
| 2 | 0.317 | 0.398 | **0.846** | **0.810** |
| 3 | 0.254 | 0.436 | 0.838 | 0.805 |

**What I noticed — overfitting in practice:** My training loss kept dropping every 
epoch, which looked good at first glance, but my validation loss actually got worse 
after epoch 1, and accuracy/F1 peaked at epoch 2 before slipping slightly at epoch 3. 
That was my first real, hands-on look at overfitting — the model kept getting better 
at the training tweets specifically, but started losing some of its ability to 
generalize to tweets it hadn't seen. I'd set `load_best_model_at_end=True` (tracking 
F1) before training, specifically so the run would keep the best checkpoint 
automatically — which turned out to matter here, since my actual final model is the 
epoch 2 version, not epoch 3.

**Final model performance (epoch 2 checkpoint):** 84.6% accuracy, 0.810 F1

## Baseline vs. Fine-Tuned Comparison

| Model | Accuracy | F1 Score (Disaster class) |
|---|---|---|
| TF-IDF + Logistic Regression | 0.800 | 0.745 |
| Fine-tuned DistilBERT | **0.846** | **0.810** |

Fine-tuning improved F1 by about 6.5 points over my baseline. My guess for why: 
DistilBERT actually understands context (like telling a literal disaster report 
apart from a tweet that just uses disaster-y words figuratively or historically), 
while TF-IDF is just counting/weighting individual words with no sense of meaning 
or context around them.

## Kaggle Leaderboard Result

Submitted predictions from the fine-tuned DistilBERT model (v4) to the official 
Kaggle competition leaderboard.

**Result: F1 Score 0.83450 — Rank #177**

This closely matches the local validation F1 (0.8145) from the same model, which is 
a good sign the validation setup was solid and not overly optimistic (no meaningful 
gap between local validation and the real, unseen competition test set).

## Known Limitations
The model struggles with figurative/idiomatic use of disaster-related language. For 
example, "This traffic is an absolute disaster" gets classified as a real disaster, 
likely because the literal word "disaster" is a strong learned signal, and the model 
hasn't fully learned to distinguish literal from figurative usage. This mirrors a 
known challenge in the training data itself — some tweets use disaster vocabulary in 
historical or figurative contexts (see EDA section) without describing a real 
emergency, and a model this size, trained on a dataset this size, doesn't fully 
resolve that ambiguity quite yet.

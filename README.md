# Disaster Tweets Classification

Classifying tweets as real disaster reports vs. non-disaster, using the Kaggle 
"NLP with Disaster Tweets" competition dataset.

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


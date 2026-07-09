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

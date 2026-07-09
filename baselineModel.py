# %% Cell 1: Loading and splitting the data. 80% training, 20% validation
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv("data/train.csv")

x_train, x_val, y_train, y_val = train_test_split(
    train['text'], train['target'], test_size= 0.20, random_state = 42
)

print(f'Training Samples: {len(x_train)}')
print(f'Validation Samples: {len(x_val)}')
# %% Cell 2: TF-IDF Vectorization of the text data
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words='english', max_features=5000, lowercase=True
)

x_train_tfidf = vectorizer.fit_transform(x_train)
x_val_tfidf = vectorizer.transform(x_val)

print(f'TF-IDF Shape: {x_train_tfidf.shape}')

# %% Cell 3: Training a Logistic Regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(x_train_tfidf, y_train)

print('Model Trained!')

# %% Cell 4: Evaluating the model on the validation set
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

y_pred = model.predict(x_val_tfidf)

print(f'Accuracy: {accuracy_score(y_val, y_pred)}')
print(f'F1 Score: {f1_score(y_val, y_pred)}')
print()
print('Classification Report:')
print(classification_report(y_val, y_pred))
print()
print('Confusion Matrix:')
print(confusion_matrix(y_val, y_pred))
# %%

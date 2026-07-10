# %% Cell 1: Loading the data
import pandas as pd
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train.head()

# %%
train.info()
# %%
train.isnull().sum()

#%%
import matplotlib.pyplot as plt


# %%
train['target'].value_counts(normalize=True)*100

# %%
from collections import Counter
import re

stopwords = {
    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'and', 'is', 'it',
    'i', 'my', 'you', 'your', 'for', 'with', 'that', 'this', 'by', 'from',
    'be', 'are', 'was', 'were', 'as', 'if', 'or', 'but', 'not', 'so',
    'me', 'we', 'our', 'us', 'they', 'them', 'he', 'she', 'his', 'her',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can',
    'could', 'up', 'out', 'about', 'just', 'like', 'im', 'amp'
}

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    return text

def get_words(textSeries):
    allWords = []
    for text in textSeries:
        text = clean_text(text)
        words = re.findall(r"\b[\w']+\b", text.lower())  
        words = [w for w in words if w not in stopwords]  # filter out stopwords
        allWords.extend(words)
    
    return Counter(allWords)

disasterWords = get_words( train[train["target"] == 1]["text"])  # takes the text info of every disaster tweet and puts it into a list
non_disasterWords = get_words(train[train["target"] == 0]["text"]) # takes the text info of every non-disaster tweet and puts it into list

print("Top 15 words in DISASTER tweets")
print(disasterWords.most_common(15)) # print most common 15 words in disaster tweets
print()

print("Top 15 words in NON-DISASTER tweets")
print(non_disasterWords.most_common(15)) # print most common 15 words in non-disater tweets
# %%

# %% Cell 1: Loading the data
import pandas as pd
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train.head()

# %%
train.info()
# %%
train.isnull().sum()
# %%
train['target'].value_counts(normalize=True)*100
# %%

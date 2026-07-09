# %%
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv("data/train.csv")

x_train, x_val, y_train, y_val = train_test_split(
    train['text'], train['target'], test_size= 0.20, random_state = 42
)
# %%

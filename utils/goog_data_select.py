import pandas as pd

data = pd.read_csv("data/raw_GOOG.csv")
data = data.loc[:,"Open"]
data.to_csv("data/GOOG.csv")
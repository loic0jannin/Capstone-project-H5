import pandas as pd

data = pd.read_csv("data/raw_GOOG.csv")
data = data.loc[:,"Open"]
print(data.shape)
data.to_csv("data/GOOG.csv",index=False)



# data = pd.read_csv("data/train_GOOG/GOOG.csv")
# print(data.shape)
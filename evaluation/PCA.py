import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# data = pd.read_csv("default/samples/x3.csv").values.flatten()
data = pd.read_csv("data/test_GOOG/GOOG_0.csv")
data = preprocessing.scale(data).flatten()
plt.plot(data)
plt.savefig("evaluation/1.png")


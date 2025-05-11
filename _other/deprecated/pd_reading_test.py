import numpy as np
import pandas as pd

test = np.array(pd.read_csv("../processed_data/labels.csv")).tolist()
print(test)
for i in range(len(test)):
    test[i] = np.array([test[i]])
print(test)

test_2 = pd.read_csv("../processed_data/labels.csv", nrows=0).columns.tolist()
print(test_2)

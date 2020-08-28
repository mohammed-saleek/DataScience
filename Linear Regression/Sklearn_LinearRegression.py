import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = pd.read_csv('headbrain.csv')
df = pd.DataFrame(data)

head_size = df.values
X = head_size[:, 2]
Y = head_size[:, 3]

X = X.reshape(len(X), 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


reg = LinearRegression()
reg.fit(X_train, y_train)

y_predictions = reg.predict(X_test)

print("R-squared :", r2_score(y_test, y_predictions))

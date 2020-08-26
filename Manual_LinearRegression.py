import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

data = pd.read_csv('headbrain.csv')
df = pd.DataFrame(data)
print(df)

print(f'Check is there any null values')
print(df.isnull().any())

print(f'check unique values')
print(df.nunique())

x = sns.scatterplot(y='Brain Weight(grams)', x='Head Size(cm^3)', data=df)

head_size = df.values

X = head_size[:, 2]
Y = head_size[:, 3]


def Linear_Regression(X, Y):
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    n = len(X)
    numerator = 0
    denominator = 0
    for i in range(n):
        numerator += ((X[i] - mean_x) * (Y[i] - mean_y))
        denominator += ((X[i] - mean_x) ** 2)

    m = numerator / denominator  # b1
    c = mean_y - m * mean_x  # b0

    return m, c


def predict(X, m, c):
    pred_y = list()
    for i in range(len(X)):
        pred_y.append(c + m * X[i])

    return pred_y


def r2score(y_obs, y_pred):
    yhat = np.mean(y_obs)

    ss_res = 0.0
    ss_tot = 0.0

    for i in range(len(y_obs)):
        ss_res += (y_obs[i] - y_pred[i]) ** 2
        ss_tot += (y_obs[i] - yhat) ** 2

    r2 = 1 - (ss_res / ss_tot)

    return r2


plt.title("Linear Regression Plot of HeadSize Vs Brain Weight")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

m, c = Linear_Regression(X_train, y_train)
print("slope = ", m)
print('intercept = ', c)

y_pred = predict(X_test, m, c)

print("R-squared :", r2score(y_test, y_pred))
print(m, c)

plt.plot(X_test, y_pred, color='red', label='Linear Regression')
plt.scatter(X_train, y_train, c='b', label='Scatter Plot')
plt.xlabel("Head Size")
plt.ylabel("Brain Weight")
plt.legend()
plt.show()

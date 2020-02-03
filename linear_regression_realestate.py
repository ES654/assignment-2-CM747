import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

# Reading Data
data = pd.read_excel("./datasets/Real estate valuation data set.xlsx")
data = data.drop(["No"], axis=1)

X_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]

weights_5folds = list()
trainMAE_5folds = list()
testMAE_5folds = list()

for i in range(5):
    a = int((i/5)*len(data))
    b = int(((i+1)/5)*len(data))
    X = pd.concat([X_data.iloc[:a, :], X_data.iloc[b:,:]], ignore_index=True)
    X_test = X_data.iloc[a:b, :]
    y = pd.concat([y_data.iloc[:a], y_data.iloc[b:]], ignore_index=True)
    y_test = y_data.iloc[a:b]
    LR = LinearRegression(fit_intercept=True)
    LR.fit(X, y)
    weights_5folds.append(LR.weights)
    y_hat = LR.predict(X)
    y_test_hat = LR.predict(X_test)
    trainMAE_5folds.append(mae(y_hat, y))
    testMAE_5folds.append(mae(y_test_hat, y_test))


print(weights_5folds)

weightlabels = list()
for i in range(X_data.shape[1]+1):
    weightlabels.append("theta_"+str(i))

for i in range(5):
    plt.subplot(2,3,i+1)
    plt.bar(weightlabels, (weights_5folds[i]))
    plt.yscale("log")
    plt.ylabel("Value of co-effecients in log scale")
    plt.xlabel("coefficients")
    plt.title("Fold "+str(i+1))
plt.show()


print(trainMAE_5folds)
print(testMAE_5folds)

folds = [1,2,3,4,5]

plt.subplot(1,2,1)
plt.bar(x=folds,height=trainMAE_5folds)
plt.xlabel("Folds")
plt.ylabel("MAE Score")
plt.title("Train MAE Scores across 5 folds")

plt.subplot(1,2,2)
plt.bar(x=folds ,height=testMAE_5folds)
plt.xlabel("Folds")
plt.ylabel("MAE Score")
plt.title("Test MAE Scores across 5 folds")

plt.show()
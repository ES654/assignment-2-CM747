"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
# from sklearn.tree import DecisionTreeClassifier
from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria,max_depth=1)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features


f =  open("datasets/iris.data","rb")
iris_data = repr(f.read())[2:-1].strip("\\n").split("\\n")

# Pre-processing Data
np.random.shuffle(iris_data)



X_data = {"sepal length": list(), "sepal width": list(), "petal length":list(), "petal width": list()}
y_data = list()
for i in range(len(iris_data)):
    row = iris_data[i].strip(" ").split(",")
    X_data["sepal length"].append(float(row[0]))
    X_data["sepal width"].append(float(row[1]))
    X_data["petal length"].append(float(row[2]))
    X_data["petal width"].append(float(row[3]))
    if(row[4]=="Iris-virginica"):
        y_data.append(row[4])
    else:
        y_data.append("Not_Iris-virginica")
target_names = ["Iris-virginica", "Not_Iris-virginica"]
X_data = pd.DataFrame(data=X_data)
y_data = pd.Series(data=y_data).replace(to_replace=target_names, value=[0,1])
y_data = y_data.astype("category")

# Defining Train Test Split
train_test_split = int(0.6*len(iris_data))

X = X_data.iloc[:train_test_split, :].drop(["sepal length", "petal length"], axis=1)
X_test = X_data.iloc[train_test_split:, :].drop(["sepal length", "petal length"], axis=1)
y = y_data.iloc[:train_test_split]
y_test = y_data.iloc[train_test_split:]


criteria = 'information_gain'
tree = DecisionTree(criterion=criteria,max_depth=1)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=3 )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)

###Write code here

# Fetching Dataset
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
    y_data.append(row[4])
target_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
X_data = pd.DataFrame(data=X_data)
y_data = pd.Series(data=y_data).replace(to_replace=target_names, value=[0,1,2])
y_data = y_data.astype("category")


# Defining Train Test Split
train_test_split = int(0.6*len(iris_data))

X = X_data.iloc[:train_test_split, :].drop(["sepal length", "petal length"], axis=1)
X_test = X_data.iloc[train_test_split:, :].drop(["sepal length", "petal length"], axis=1)
y = y_data.iloc[:train_test_split]
y_test = y_data.iloc[train_test_split:]

for criteria in ['entropy', 'gini']:
    Classifier_RF = RandomForestClassifier(10, criterion = criteria)
    Classifier_RF.fit(X, y)
    print("Train Scores:")
    y_hat = Classifier_RF.predict(X)
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print("Class =",target_names[cls])
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))
    print("Test Scores:")
    y_test_hat = Classifier_RF.predict(X_test)
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_test_hat, y_test))
    for cls in y.unique():
        print("Class =",target_names[cls])
        print('Precision: ', precision(y_test_hat, y_test, cls))
        print('Recall: ', recall(y_test_hat, y_test, cls))

    # Plots
    plot_colors = "ryb"

    x_min, x_max = X.iloc[:, 0].min() - .5, X.iloc[:, 0].max() + .5
    y_min, y_max = X.iloc[:, 1].min() - .5, X.iloc[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    d = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)
    Z = Classifier_RF.predict(d)
    Z = np.array(Z).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=.8)
    for i, color in zip(range(3), plot_colors):
        idx = np.where(y == i)
        plt.scatter(np.array(X)[idx, 0], np.array(X)[idx, 1], c=color, label=target_names[i], cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
    
    plt.xlabel(str(X.columns[0]))
    plt.ylabel(str(X.columns[1]))
    plt.title("Final Random Forest, Criteria = "+criteria)
    plt.legend()
    plt.tight_layout()
    plt.show()

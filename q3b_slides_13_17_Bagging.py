import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
# from tree.base import DecisionTree
# Or use sklearn decision tree
from sklearn.tree import DecisionTreeClassifier
from linearRegression.linearRegression import LinearRegression

data = {'X1':list(), 'X2': list()}
labels = list()
for i in range(1,9):
    for j in range(1,9):
        data['X1'].append(i)
        data['X2'].append(j)
        if(i==3 and j==3):
            labels.append('B')
        elif(i==5 and j==8):
            labels.append('Y')
        elif(i<=5 and j<=5):
            labels.append('Y')
        else:
            labels.append('B')

X = pd.DataFrame(data=data)
y = pd.Series(labels, dtype="category")




criteria = 'entropy'
tree = DecisionTreeClassifier(criterion=criteria)
Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=5 )
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
[fig1, fig2] = Classifier_B.plot()
print('Criteria :', criteria)
print('Train Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))

X_test = pd.DataFrame(data={'X1':[3,5], 'X2':[3,8]})
print(X_test)
y_test_hat = Classifier_B.predict(X_test)
print(y_test_hat)

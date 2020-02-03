import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from matplotlib.colors import ListedColormap

class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        self.n_estimators = n_estimators
        self.classifiers = [copy.deepcopy(base_estimator) for i in range(self.n_estimators)]
        self.alphas = [None for i in range(self.n_estimators)]
        self.classes = None
        self.X = None
        self.Y = None
        self.weights_clfs = [None for i in range(self.n_estimators)]

    def fit(self, X1, y1):
        self.classes = np.unique(y1)
        assert(len(self.classes)==2)

        X =X1[:]
        y = y1[:]

        self.X = X1
        self.Y = y1

        y.replace(to_replace=self.classes, value=[-1,1], inplace=True)
        y = y.astype("category")

        weights = pd.Series([1/y.size for i in range(y.size)])

        for i in range(len(self.classifiers)):
            self.weights_clfs[i] = copy.deepcopy(weights)
            self.classifiers[i].fit(X, y, weights)
            y_hat = self.classifiers[i].predict(X)
            err = 0
            for j in range(y_hat.size):
                if(y_hat[j]!=y[j]):
                    err += weights.iat[j]
            self.alphas[i] = 0.5*np.log((1-err)/err)
            for j in range(y_hat.size):
                if(y_hat[j]!=y[j]):
                    weights.iat[j] = weights.iat[j]*np.exp(self.alphas[i])
                else:
                    weights.iat[j] = weights.iat[j]*np.exp(-1*self.alphas[i])
            k = np.sum(weights)
            weights = weights*(1/k)
            self.classifiers[i].plot()

    def predict(self, X):
        y_hat = self.classifiers[0].predict(X) * self.alphas[0]
        for i in range(1,self.n_estimators):
            y_hat += (self.classifiers[i].predict(X) * self.alphas[i])
        y_hat = np.sign(y_hat)
        y_hat = y_hat.replace(to_replace=[-1,1], value=self.classes)

        return y_hat

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        h=0.02
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        fig1 = plt.figure(figsize=(5*self.n_estimators, 5))

        x_min, x_max = self.X.iloc[:, 0].min() - .5, self.X.iloc[:, 0].max() + .5
        y_min, y_max = self.X.iloc[:, 1].min() - .5, self.X.iloc[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        d = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=self.X.columns)

        for i in range(self.n_estimators):    
            Z = self.classifiers[i].predict(d)
            Z = np.array(Z).reshape(xx.shape)

            plt.subplot(1,self.n_estimators,i+1)
            plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            plt.scatter(self.X.iloc[:,0], self.X.iloc[:,1], c=self.Y, s=self.weights_clfs[i]*1500 ,cmap=cm_bright, edgecolors='k')
            plt.xlabel(str(self.X.columns[0]))
            plt.ylabel(str(self.X.columns[1]))
            plt.title("Alpha = "+str(self.alphas[i]))
        
        plt.show()

        fig2 = plt.figure(figsize=(6, 6))

        Z = self.predict(d)
        Z = np.array(Z).reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        plt.scatter(self.X.iloc[:,0], self.X.iloc[:,1], c=self.Y,cmap=cm_bright, edgecolors='k')
        plt.xlabel(str(self.X.columns[0]))
        plt.ylabel(str(self.X.columns[1]))
        plt.title("Final Decesion Surface")
        plt.show()

        return [fig1,fig2]


from .base import DecisionTree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import operator
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import graphviz
from sklearn import tree
from sklearn.externals.six import StringIO
from subprocess import call
import matplotlib.image as mpimg

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin'


class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.classifiers = [None for i in range(self.n_estimators)]
        self.clf_features_not_included = [None for i in range(self.n_estimators)]
        self.X = [None for i in range(self.n_estimators)]
        self.Y = None
        self.X_org = None

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        assert(X.shape[0]==y.size)

        self.X_org = X[:]
        self.Y = y[:]

        m = min(2, X.shape[1])

        for round in range(self.n_estimators):
            tree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth)
            remove_features = random.sample(list(X.columns),k=(X.shape[1]-m))
            X1 = X.drop(remove_features, axis=1)
            self.X [round]= X1
            self.classifiers[round] = tree.fit(X1,y)
            self.clf_features_not_included[round] = remove_features

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hats = list()
        for i in range(len(self.classifiers)):
            X1 = X.drop(self.clf_features_not_included[i], axis=1)
            y_hats.append(self.classifiers[i].predict(X1))
        
        y_hat = list()

        for i in range(X.shape[0]):
            predictions = dict()
            for pred in y_hats:
                if(pred[i] in predictions):
                    predictions[pred[i]] += 1
                else:
                    predictions[pred[i]] = 1
            y_hat.append(max(predictions.items(), key=operator.itemgetter(1))[0])
        
        return pd.Series(y_hat)


    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """

        # fig1 = plt.figure(figsize=(self.n_estimators*5,5))
        
        for i in range(self.n_estimators):
            # plt.subplot(1,self.n_estimators,i+1)
            tree.export_graphviz(self.classifiers[i], out_file='tree.dot',  feature_names=self.X[i].columns,  
                class_names=[str(i) for i in self.Y.unique()],  filled=True, rounded=True, special_characters=True) 
            call(['dot', '-Tpng', 'tree.dot', '-o', './result_images/q4_randomForest_Classifiers/tree'+str(i)+'.png', '-Gdpi=100'])
            img = mpimg.imread('./result_images/q4_randomForest_Classifiers/tree'+str(i)+'.png')
            plt.imshow(img)
            plt.tight_layout()
            plt.show()


        h=0.02
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        fig2 = plt.figure(figsize=(self.n_estimators*5,2))

        for i in range(self.n_estimators):
            x_min, x_max = self.X[i].iloc[:, 0].min() - .5, self.X[i].iloc[:, 0].max() + .5
            y_min, y_max = self.X[i].iloc[:, 1].min() - .5, self.X[i].iloc[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            if hasattr(self.classifiers[i], "decision_function"):
                Z = self.classifiers[i].decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = self.classifiers[i].predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            
            Z = Z.reshape(xx.shape)

            plt.subplot(1,self.n_estimators,i+1)
            plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            plt.scatter(self.X[i].iloc[:,0], self.X[i].iloc[:,1], c=self.Y, cmap=cm_bright, edgecolors='k')
            
            plt.xlabel(str(self.X[i].columns[0]))
            plt.ylabel(str(self.X[i].columns[1]))
            plt.title("Round "+str(i+1))
        plt.tight_layout()
        plt.show()


        fig3 = plt.figure(figsize=(6,6))

        x_min, x_max = self.X_org.iloc[:, 0].min() - .5, self.X_org.iloc[:, 0].max() + .5
        y_min, y_max = self.X_org.iloc[:, 1].min() - .5, self.X_org.iloc[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        d = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=self.X_org.columns)
        Z = self.predict(d)
        Z = np.array(Z).reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        plt.scatter(self.X_org.iloc[:,0], self.X_org.iloc[:,1], c=self.Y, cmap=cm_bright, edgecolors='k')
        
        plt.xlabel(str(self.X_org.columns[0]))
        plt.ylabel(str(self.X_org.columns[1]))
        plt.title("Final Random Forest")
        plt.tight_layout()
        plt.show()
        





class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='mse', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.classifiers = [None for i in range(self.n_estimators)]
        self.clf_features_not_included = [None for i in range(self.n_estimators)]
        self.X = [None for i in range(self.n_estimators)]
        self.Y = None
        self.X_org = None

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        assert(X.shape[0]==y.size)

        self.X_org = X[:]
        self.Y = y[:]

        m = min(2, X.shape[1])

        for round in range(self.n_estimators):
            tree = DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth)
            remove_features = random.sample(list(X.columns),k=(X.shape[1]-m))
            X1 = X.drop(remove_features, axis=1)
            self.X [round]= X1
            self.classifiers[round] = tree.fit(X1,y)
            self.clf_features_not_included[round] = remove_features

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hats = list()
        for i in range(len(self.classifiers)):
            X1 = X.drop(self.clf_features_not_included[i], axis=1)
            y_hats.append(self.classifiers[i].predict(X1))
        
        y_hat = list()

        for i in range(X.shape[0]):
            y_hat.append(0)
            for pred in y_hats:
                y_hat[i] += pred[i]
            y_hat[i] = y_hat[i]/self.n_estimators
        
        return pd.Series(y_hat)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        fig1 = plt.figure(figsize=(self.n_estimators*7,5))
        
        for i in range(self.n_estimators):
            plt.subplot(1,self.n_estimators,i+1)
            tree.export_graphviz(self.classifiers[i], out_file='tree.dot',  feature_names=self.X[i].columns,  
                class_names=[str(i) for i in self.Y.unique()],  filled=True, rounded=True, special_characters=True) 
            call(['dot', '-Tpng', 'tree.dot', '-o', './result_images/q4_randomForest_regressors/tree'+str(i)+'.png', '-Gdpi=50'])
            img = mpimg.imread('./result_images/q4_randomForest_regressors/tree'+str(i)+'.png')
            plt.imshow(img)
            plt.tight_layout()
            plt.show()


        h=0.02
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        

        fig2 = plt.figure(figsize=(self.n_estimators*5,5))

        for i in range(self.n_estimators):
            x_min, x_max = self.X[i].iloc[:, 0].min() - .5, self.X[i].iloc[:, 0].max() + .5
            y_min, y_max = self.Y.min() - .5, self.Y.max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            if hasattr(self.classifiers[i], "decision_function"):
                Z = self.classifiers[i].decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = self.classifiers[i].predict(np.c_[xx.ravel()])[:]
            
            Z = Z.reshape(xx.shape)

            plt.subplot(1,self.n_estimators,i+1)
            plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            plt.scatter(self.X[i].iloc[:,0], self.Y, cmap=cm_bright, edgecolors='k')
            
            plt.xlabel(str(self.X[i].columns[0]))
            plt.ylabel("Label, Y")
            plt.title("Round "+str(i+1))
        plt.tight_layout()
        plt.show()


        fig3 = plt.figure(figsize=(6,6))

        x_min, x_max = self.X_org.iloc[:, 0].min() - .5, self.X_org.iloc[:, 0].max() + .5
        y_min, y_max = self.Y.min() - .5, self.Y.max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        d = pd.DataFrame(np.c_[xx.ravel()], columns=self.X_org.columns)
        Z = self.predict(d)
        Z = np.array(Z).reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        plt.scatter(self.X_org.iloc[:,0], self.Y, cmap=cm_bright, edgecolors='k')
        
        plt.xlabel(str(self.X_org.columns[0]))
        plt.ylabel("Label,Y")
        plt.title("Final Random Forest")
        plt.tight_layout()
        plt.show()

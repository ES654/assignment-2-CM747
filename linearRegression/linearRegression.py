import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegression():
    def __init__(self, fit_intercept=True, method='normal'):
        '''

        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        :param method:
        '''
        self.fit_intercept = fit_intercept
        self.method = method
        self.weights = None
        self.X = None
        self.Y = None
        self.residuals = None

    def fit(self, X1, y1):
        '''
        Function to train and construct the LinearRegression
        :param X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        :param y: pd.Series with rows corresponding to output variable (shape of Y is N)
        :return:
        '''
        X = X1[:]
        y = y1[:]

        self.X = X
        self.Y = y

        if(self.fit_intercept):
            X.insert(0, "intercept", 1)
        
        X = np.array(X)
        y = np.array(y)

        XT = np.transpose(X)
        XTX_inv = np.linalg.pinv(np.matmul(XT,X))
        K = np.matmul(XT,y)
        self.weights = np.matmul(XTX_inv,K)


    def predict(self, X1):
        '''
        Funtion to run the LinearRegression on a data point
        :param X: pd.DataFrame with rows as samples and columns as features
        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X = X1[:]
        if(self.fit_intercept and "intercept" not in X.columns):
            X.insert(0, "intercept", 1)
        X = np.array(X)
        y_hat = np.matmul(X,self.weights)

        return pd.Series(y_hat)


    def plot_residuals(self):
        """
        Function to plot the residuals for LinearRegression on the train set and the fit. This method can only be called when `fit` has been earlier invoked.

        This should plot a figure with 1 row and 3 columns
        Column 1 is a scatter plot of ground truth(y) and estimate(\hat{y})
        Column 2 is a histogram/KDE plot of the residuals and the title is the mean and the variance
        Column 3 plots a bar plot on a log scale showing the coefficients of the different features and the intercept term (\theta_i)

        """
        y_hat = self.predict(self.X)

        fig = plt.figure(figsize=(18,6))

        plt.subplot(1,3,1)
        plt.scatter(x=[i for i in range(self.Y.size)], y=self.Y, label="Ground Truth(Y)")
        plt.scatter(x=[i for i in range(self.Y.size)], y=y_hat, label="Estimated(Y_hat)")
        plt.xlabel("Sample No.")
        plt.ylabel("Y and Y_hat values")
        plt.legend()

        self.residuals = self.Y - y_hat
        plt.subplot(1,3,2)
        sns.kdeplot(self.residuals, color='b', shade=True)
        plt.ylabel("Probability Density of residuals")
        plt.title("the mean and variance of residuals")

        plt.subplot(1,3,3)
        plt.bar(self.X.columns.map(str),(self.weights))
        plt.yscale("log")
        plt.xlabel("Features/Attributes")
        plt.ylabel("Value of co-efficients in log scale")
        plt.title("Co-efficients of different Features")


        plt.show()


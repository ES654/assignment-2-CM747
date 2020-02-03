import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time

from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit(X, y)
    y_hat = LR.predict(X)
    LR.plot_residuals()

    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))


N_range = [i for i in range(30,200)]
P_range = [i for i in range(10,50)]

fit_timesVsN = list()
P = 5
for N in N_range:
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    LR = LinearRegression(fit_intercept=False)
    startTime = time.time()
    LR.fit(X,y)
    endTime = time.time()
    fit_timesVsN.append((endTime-startTime))

fit_timesVsP = list()
N = 100
for P in P_range:
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    LR = LinearRegression(fit_intercept=False)
    startTime = time.time()
    LR.fit(X,y)
    endTime = time.time()
    fit_timesVsP.append((endTime-startTime))


fig = plt.figure(figsize=(18,5))


plt.subplot(1,2,1)
plt.plot(N_range, fit_timesVsN)
plt.xlabel("N (No. of Samples)")
plt.ylabel("Emperical_fit_time")

plt.subplot(1,2,2)
plt.plot(P_range, fit_timesVsP)
plt.xlabel("P (No. of Attributes/Features)")
plt.ylabel("Emperical_fit_time")

plt.show()

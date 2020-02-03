# ES654-2020 Assignment 2

Chandan Maji - 17110037

------

## 2(a)
RMSE:  0.9044265250464999 <br>
MAE:  0.7114318433966357 <br>
RMSE:  0.9452375875781541 <br>
MAE:  0.7590075233630846 <br>
![Residuals with Intercept Term](./result_images/q2_linearRegression_1.png)
![Residuals without Intercept Term](./result_images/q2_linearRegression_2.png)


## 2(b) Time Complexity

Theoritical time complexity of Normal Equation is O(P^3 + N*P^2), where P is the number of features and N is the number of samples.

![Time Complexity Heatmap](./result_images/q2_linearRegression_3.png)


## 2(c) Real Estate Dataset

![Weights of the 5 folds](./result_images/q2_linearRegression_realEstate_1.png)
![Test and Train MAE Scores of 5 folds](./result_images/q2_linearRegression_realEstate_2.png)
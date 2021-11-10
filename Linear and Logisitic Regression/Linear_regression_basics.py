# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 08:04:55 2021

@author: Ali
Basics of linear regression in Python
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
x = 10*ss.uniform.rvs(size=n)
y = beta_0+beta_1 *x + ss.norm.rvs(loc =0, scale = 1, size =n)

plt.figure()
plt.plot(x,y,"o", ms = 5)
xx = np.array([0,10])
plt.plot(xx, beta_0+beta_1 *xx)
plt.xlabel("x")
plt.ylabel("y")

"Calculate RSS values for the above fit and data set"
rss = []
slopes = np.arange(-10,15,0.01)
for slope in slopes:
    rss.append(np.sum((y -beta_0-slope*x)**2))
 
"Find the slope with the lowest RSS value"
ind_min = np.argmin(rss)
print("Estimate for the slope: ", slopes[ind_min])

"Plot the Slopes vs their corresponding RSS values"
plt.figure()
plt.plot(slopes,rss)
plt.xlabel("Slopes")
plt.ylabel("RSS")

"An alternate method of fitting the data"
import statsmodels.api as sm

X = sm.add_constant(x)
mod = sm.OLS(y,X)
est = mod.fit()
print(est.summary())

"Using scikit-learn for linear regression"

n = 500 
beta_0 = 5
beta_1 = 2
beta_2 = -1

np.random.seed(1)

x_1 = 10*ss.uniform.rvs(size=n)
x_2 = 10*ss.uniform.rvs(size = n)
y = beta_0+ beta_1*x_1 + beta_2*x_2 +ss.norm.rvs(loc=0,scale=1,size=n)

X = np.stack([x_1,x_2], axis = 1)

"Create a 3D plot t0 visualize x_1, x_2 and y at the same time"
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection= '3d')
ax.scatter(X[:,0], X[:,1], y, c = y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y$")

"Now we'll use scikit-learn to fit this model"

from sklearn.linear_model import LinearRegression

lm = LinearRegression(fit_intercept=True)
lm.fit(X,y)


"Now let's try to use the model to predict the outcome for a value of x"
X_0 = np.array([2,4])
lm.predict(X_0.reshape(1,-1))

"Lets find r^2 and see how well the model works"

lm.score(X, y)

"Computing MSE with divided training and test data"

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.5, random_state =1)

lm = LinearRegression(fit_intercept=True)

lm.fit(X_train,y_train)

"Test the model on the test data"

lm.score(X_test, y_test)
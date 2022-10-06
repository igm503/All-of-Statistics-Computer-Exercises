import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt

data = np.loadtxt('All of Statistics Problem Sets/carmileage.dat', skiprows=28, dtype=str)
data = data[:,1:].astype(float)
n = data[:,1].size

#estimating regression function
X = data[:, 1].reshape(n,1)
Y = data[:, 2].reshape(n,1)
intercept_cov = np.ones(X.shape)
X = np.append(intercept_cov, X, axis=1)
XtXinv = lin.inv(np.matmul(np.transpose(X), X))
XtY = np.matmul(np.transpose(X), Y)
beta = np.matmul(XtXinv, XtY)

#estimating using log(mpg) as the response
logY = np.log(data[:, 2].reshape(n,1))
XtlogY = np.matmul(np.transpose(X), logY)
log_beta = np.matmul(XtXinv, XtlogY)

#Regression Function
hp = np.linspace(0, 300, 100)
predicted_mpg = hp * beta[1,0] + beta[0,0]
predicted_logmpg = hp * log_beta[1,0] + log_beta[0,0]

print(predicted_mpg)
fig, (axis1, axis2) = plt.subplots(1, 2)
axis1.scatter(X[:, 1], Y)
axis1.plot(hp, predicted_mpg)
axis1.set(title='Regression of MPG on HP', xlabel='HP', ylabel='MPG')
axis2.scatter(X[:, 1], logY)
axis2.plot(hp, predicted_logmpg)
axis2.set(title='Regression of logMPG on HP', xlabel='HP', ylabel='logMPG')
plt.show()

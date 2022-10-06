import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

data = np.loadtxt('faithful.dat', skiprows=26)[:,1:]
n = data[:,0].size

#mean waiting time estimator
mean = round(data[:,1].mean(), 3)
print(f'Mean Waiting Time: {mean}')

#se of mean estimator
E_x_2 = round(np.sum(data[:,1]*data[:,1])/n, 3)
variance_mean = (E_x_2 - mean * mean)/n
se_mean = round(np.sqrt(variance_mean), 3)

print(f'se of mean estimate: {se_mean}')

#90% confidence interval of mean estimator
z_90 = norm.ppf(.95)
epsilon = round(se_mean * z_90, 3)
print(f'90% Conf Interval of Mean Estimate: ({mean-epsilon:.3f}, {mean + epsilon:.3f})')

#median waiting time estimator
print(f'Median Waiting Time: {np.median(data[:,1])}')
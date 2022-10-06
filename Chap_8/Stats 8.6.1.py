import numpy as np
from scipy.stats import norm

LSAT = np.array([576, 635, 558, 578, 666, 580, 555, 661, 
                651, 605, 653, 575, 545, 572, 594])
GPA = np.array([3.39, 3.3 , 2.81, 3.03, 3.44, 3.07, 3.0, 
               3.43, 3.36, 3.13, 3.12, 2.74, 2.76, 2.88, 3.96])
n = LSAT.size

#Cor Estimate
def correlation(x, y):
    variance_x = np.sum((x - x.mean())**2)
    variance_y = np.sum((y - y.mean())**2)
    covariance = np.sum((x - x.mean())*(y-y.mean()))
    return covariance / np.sqrt(variance_x * variance_y)
cor = correlation(LSAT, GPA)
print(f'Estimated Correlation Coefficient: {cor:.3f}')

#Bootstrap Estimate of SE of Cor 
b = 1000
rng = np.random.default_rng()
x= rng.integers(low = 0, high = 15, size = 15 * b).reshape(b, 15)
LSAT_sample = LSAT[x]
GPA_sample = GPA[x]

tboot = np.zeros(b)
for i in range(b):
    tboot[i] = correlation(LSAT_sample[i], GPA_sample[i])

tboot_se = np.sqrt(np.sum((tboot - tboot.mean())**2)/(tboot.size-1))
print(f'Bootstrap estimate of se(theta): {tboot_se:.3f}')

#Bootstrap confidence intervals

#Normal Conf Interval
a = 0.05
z_a = norm.ppf(.975)
print(f'Normal 95% Conf Interval: ({cor - z_a * tboot_se:.3f},{cor + z_a * tboot_se:.3f})')

#Pivotal Interval
theta_star_1 = np.quantile(tboot, .975)
theta_star_2 = np.quantile(tboot, .025)
print(f'Pivotal 95% Conf Interval: ({2*cor - theta_star_1:.3f},{2*cor - theta_star_2:.3f})')

#Percentile Inverval
print(f'Percentile 95% Conf Interval: ({theta_star_2:.3f},{theta_star_1:.3f})')
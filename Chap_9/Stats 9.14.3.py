from locale import normalize
import numpy as np
from scipy.stats import norm

x = np.array([3.23, 1.03, 0.33, 1.54, 0.39, -2.50, -0.07, -0.31,
             2.28, 1.88, -0.68, -0.01, 0.76, 0.30, -0.61, 0.42,
             2.33, 4.43, 0.17, 1.76, 3.18, 1.52, 5.43, -1.03, 4.00])

n = x.size
mu_hat = x.mean()
sigma_hat = x.std()
tau_hat = norm.ppf(.95) * sigma_hat + mu_hat
print(f'MLE estimate of Tau: {tau_hat:.3f}')

#bootstrap se of tau_hat
trials = 100
rng = np.random.default_rng()
tboot = np.zeros(trials)
for i in range(trials):
    index = rng.integers(0,n)
    sim_sample = x[index]
    tboot[i] = norm.ppf(.95) * sim_sample.std() + sim_sample.mean()
se_tau_hat = tboot.std()
print(f'Bootstrap SE Estimate of Tau: {se_tau_hat:.3f}')

#Delta Method se of tau_hat
print('mean:', mu_hat)
print('sigma:', sigma_hat)
grad_g = np.array([1, norm.ppf(.95)])
print('gradient:', grad_g)
fisher = np.linalg.inv(np.array([[n/sigma_hat**2, 0],[0, 2*n/(sigma_hat**2)]]))
print('fisher:', fisher)
first_product = np.matmul(grad_g.reshape(1,2), fisher)
print('first product:', first_product)
product= np.matmul(first_product, grad_g.reshape(2,1))
se_tau_hat = np.sqrt(product)
print('Delta Method SE Estimate of Tau:', se_tau_hat)
alternate = sigma_hat * np.sqrt((2 + norm.ppf(.95)**2)/n)
print('Alternate:', alternate)


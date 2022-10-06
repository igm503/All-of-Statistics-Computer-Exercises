import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

n = 100
rng = np.random.default_rng()
x = rng.normal(5, 1, size=n)
mu_hat = x.mean()
se_mu = 1 / np.sqrt(n)
z = norm.ppf(.975)
mu_upper = mu_hat + z * se_mu
mu_lower = mu_hat - z * se_mu
print(f'95% Conf Int for Mu: ({mu_lower:.3f},{mu_upper:.3f})')

#delta method for se of theta_hat = e^mu_hat
theta_hat = np.exp(mu_hat)
se_theta = theta_hat * se_mu
theta_upper = theta_hat + z * se_theta
theta_lower = theta_hat - z * se_theta
print(f'Delta 95% Conf Int for Theta: ({theta_lower:.3f},{theta_upper:.3f})')

#Parametric Bootstrap of theta_hat
b = 100000
x_simulation = rng.normal(mu_hat, 1, size=b*n).reshape(b, n)
mu_boot = x_simulation.mean(axis = 1)
se_boot_mu = mu_boot.std()
mu_upper = mu_hat + z * se_boot_mu
mu_lower = mu_hat - z * se_boot_mu
print(f'\nParametric Bootstrap 95% Conf Int for Mu: ({mu_lower:.3f},{mu_upper:.3f})')

theta_boot = np.exp(mu_boot)
se_boot_theta = theta_boot.std()
theta_upper = theta_hat + z * se_boot_theta
theta_lower = theta_hat - z * se_boot_theta
print(f'Parametric Bootstrap 95% Conf Int for Theta: ({theta_lower:.3f},{theta_upper:.3f})')

#Nonparametric Bootstrap of theta_hat
simulation_index = rng.integers(0, n, size=b*n).reshape(b,n)
x_simulation = x[simulation_index]
mu_boot_np = x_simulation.mean(axis = 1)
se_boot_mu_np = mu_boot_np.std()
mu_upper_np = mu_hat + z * se_boot_mu_np
mu_lower_np = mu_hat - z * se_boot_mu_np
print(f'\nNonparametric Bootstrap 95% Conf Int for Mu: ({mu_lower_np:.3f},{mu_upper_np:.3f})')

theta_boot_np = np.exp(mu_boot_np)
se_boot_theta_np = theta_boot_np.std()
theta_upper_np = theta_hat + z * se_boot_theta_np
theta_lower_np = theta_hat - z * se_boot_theta_np
print(f'Nonparametric Bootstrap 95% Conf Int for Theta: ({theta_lower_np:.3f},{theta_upper_np:.3f})')

#True Distribution of theta_hat
x_simulation = rng.normal(5, 1, size=n*b).reshape(b,n)
mu_hat_sim = x_simulation.mean(axis=1)
theta_hat_sim = np.exp(mu_hat_sim)

#Compare Histograms of theta_hat using different methods
print(se_theta**2)
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
ax0.hist(theta_hat_sim, 100, range=[100, 200])
ax0.set(title='True Distribution')
ax1.hist(rng.normal(theta_hat, se_theta, size = b), 100, range=[100, 200])
ax1.set(title='Delta Method Distribution')
ax2.hist(theta_boot, 100, range=[100, 200])
ax2.set(title='Parametric Bootstrap Distribution')
ax3.hist(theta_boot_np, 100, range=[100, 200])
ax3.set(title='NonPar Bootstrap Distribution')
plt.show() 
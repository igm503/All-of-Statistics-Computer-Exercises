import numpy as np

data = np.loadtxt('clouds.dat', skiprows=30)
n = data[:,0].size

#estimate means and theta
mu_hat_seeded = data[:,1].mean()
mu_hat_normal = data[:,0].mean()
theta_hat = mu_hat_seeded - mu_hat_normal

print(f'Unseeded Mean: {mu_hat_normal:.2f}')
print(f'Seeded Mean: {mu_hat_seeded:.2f}')
print(f'Difference: {theta_hat:.2f}')

#estimate standard error of theta

E_x_2_seeded = (data[:,1] * data[:,1]).mean()
E_x_2_normal = (data[:,0] * data[:,0]).mean()
se_hat = np.sqrt((E_x_2_seeded + E_x_2_normal - mu_hat_seeded**2 - mu_hat_normal**2)/n)
print(f'Standard Error of Theta: {se_hat:.2f}')

#estimate confidence interval
print(f'95% Confidence Interval of Theta: ({theta_hat - 2 * se_hat:.2f},{theta_hat + 2 * se_hat:.2f})')
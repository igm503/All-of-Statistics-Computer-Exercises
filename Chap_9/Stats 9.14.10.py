import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

n = 50
theta = 1
rng = np.random.default_rng()
x = rng.uniform(0,theta, size=n)
theta_hat = np.max(x)

#parametric bootstrap for theta 
b=10000
x_simulation_p = rng.uniform(0,theta_hat, size=n*b).reshape(b, n)
theta_boot_p = np.max(x_simulation_p, axis=1)

#nonparametric bootstrap for theta
simulation_index = rng.integers(0, n, size=b*n).reshape(b, n)
x_simulation_np = x[simulation_index]
theta_boot_np = np.max(x_simulation_np, axis=1)

#simulation of theta
x_simulation_true = rng.uniform(0,1, size=n*b).reshape(b, n)
theta_hat_sim = np.max(x_simulation_true, axis=1)

#Histograms
print('ready for graphs')
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
ax0.hist(theta_hat_sim, 40, range=[.85,1])
ax0.set(title='True Distribution')
ax1.hist(theta_boot_p, 40, range=[.85,1])
ax1.set(title='Parametric Bootstrap')
ax2.hist(theta_boot_np, 40, range=[.85,1])
ax2.set(title='Nonparametric Bootstrap')

def pdf_theta_hat(x, theta, n):
    return n * x**(n-1) / theta**n
x = np.linspace(.85, 1, 100)

ax3.plot(x, pdf_theta_hat(x, theta, n))
ax3.set(title='PDF of Theta_Hat', xlim=[.85,1])

plt.show()
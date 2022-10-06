import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
theta = 100
n = 100
x = rng.uniform(0,theta,n)
x_max = x.max()

def posterior(theta, x_max, n):
    theta = (theta >= x_max) * theta
    return x_max**(n + 2) / (theta**(n + 1) * (n + 1))

theta_plot = np.linspace(0, 200, 1000)
posterior_plot = posterior(theta_plot, x_max, n)

fig, ax0 = plt.subplots(1,1)
ax0.plot(theta_plot, posterior_plot)
ax0.set(title='Posterior of Theta')
plt.show()
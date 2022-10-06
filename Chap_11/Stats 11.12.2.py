import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.integrate as integrate

rng = np.random.default_rng()
x = rng.normal(loc=5, scale=1, size=100)

def prior_mu(mu):
    return 1

def likelihood(mu, x):
    try:
        return np.product(norm.pdf(x, loc=mu, scale=1), axis=1)
    except:
        return np.product(norm.pdf(x, loc=mu, scale=1))

def integrand(mu, x):
    return likelihood(mu, x) * prior_mu(mu)

def normalizing_constant(x):
    bounds = integrate.quad(integrand, -np.inf, np.inf, args=(x,))
    return sum(bounds)/2

def posterior(mu, x):
    return likelihood(mu, x) * prior_mu(mu) / normalizing_constant(x)

granularity = 2000
mu_list = np.linspace(4,6,granularity).reshape(granularity,1)
posterior_list = posterior(mu_list, x)

fig, ax1 = plt.subplots(1,1)
ax1.plot(mu_list, posterior_list)
ax1.set(title='Posterior Probability of mu')
plt.show()
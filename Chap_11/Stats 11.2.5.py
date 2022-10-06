import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
x = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0])

def likelihood(x, p):
    return p**(x.sum()) * (1 - p)**(x.size - x.sum())

p_range = np.linspace(0, 1, 500)

posterior_1 = likelihood(x, p_range) * stats.beta.pdf(p_range, 0.5, 0.5)
posterior_2 = likelihood(x, p_range) * stats.beta.pdf(p_range, 1, 1)
posterior_3 = likelihood(x, p_range) * stats.beta.pdf(p_range, 10, 10)
posterior_4 = likelihood(x, p_range) * stats.beta.pdf(p_range, 100, 100)
likelihood_1 = likelihood(x, p_range)
prior_1 = stats.beta.pdf(p_range, 0.5, 0.5)/1000
prior_2 = stats.beta.pdf(p_range, 1, 1)/500
prior_3 = stats.beta.pdf(p_range, 10, 10)/500
prior_4 = stats.beta.pdf(p_range, 100, 100)/1000

fig, ax = plt.subplots(1,1)
ax.plot(p_range, posterior_1, label='Post Beta 0.5, 0.5')
ax.plot(p_range, posterior_2, label='Post Beta 1, 1')
ax.plot(p_range, posterior_3, label='Post Beta 10, 10')
ax.plot(p_range, posterior_4, label='Post Beta 100, 100')
#ax.plot(p_range, likelihood_1, label='Raw Likelihood')
ax.plot(p_range, prior_1, label='Beta 0.5, 0.5')
ax.plot(p_range, prior_2, label='Beta 1, 1')
ax.plot(p_range, prior_3, label='Beta 10, 10')
ax.plot(p_range, prior_4, label='Beta 100, 100')
plt.legend()
plt.show()
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#Method of Moments with Normal Distribution
np.set_printoptions(formatter={'all':lambda x: str(x)})
n = 50
rng = np.random.default_rng()
mu = 50
sigma = 5
sample = rng.normal(mu, sigma, size=n)

def find_moments(sample, k):
    moments = np.zeros(k)
    for i in range(1,k+1):
        moments[i-1] = round(np.sum(sample**i)/n, 3)
    return moments

normal_moments = find_moments(sample, 2)
mu_hat = normal_moments[0]
sigma_hat = np.sqrt(normal_moments[1] - mu_hat**2)
print(f'Mu Estimate: {mu_hat:.3f}; Sigma Estimate: {sigma_hat:.3f}')

#MLE with Normal Distribution
def likelihood(mu, sigma, sample):
    return np.log(np.prod(norm.pdf(sample, loc=mu, scale=sigma)))

#plot likelihood function
granularity = 100
x = np.linspace(0, 100, granularity)
w = np.linspace(0.1, 10, granularity)

y = np.zeros(granularity)
z = np.zeros(granularity)
for i in range(granularity):
    y[i] = likelihood(x[i], sigma, sample)
    z[i] = likelihood(mu, w[i], sample)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

ax1.plot(x, y)
ax2.plot(w, z)

fig.savefig("MLE.png");

print(f'sum of X\'s: {np.sum(sample)}')
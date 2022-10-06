import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

rng = np.random.default_rng()
k = 5000
n = 1000
theta = rng.integers(-5, 5, (k, 1)) * np.ones((k, n))
theta = np.ones((k,n))
x = rng.normal(theta, 1, (k, n))

stein_constant = max(0, 1 - (k - 2) / sum(x.mean(axis=1)**2))
stein_estimator = stein_constant * x.mean(axis=1)

mle = x.mean(axis=1)

def loss(estimator):
    return sum((theta[:,0] - estimator)**2)

print(f'MLE loss: {loss(mle)/k:.3f}')
print(f'Stein loss: {loss(stein_estimator)/k:.3f}')





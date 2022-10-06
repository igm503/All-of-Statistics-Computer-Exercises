import numpy as np
from scipy.stats import norm

rng = np.random.default_rng()
n = 20
alpha = 0.05
lam_null = 1
b = 100000
count = 0

for i in range(b):
    x = rng.poisson(lam=1, size=n)
    lam_hat = x.mean()
    se_lam_hat = np.sqrt(lam_hat / n)
    wald = (lam_hat - lam_null) / se_lam_hat
    p_value = 2 * norm.cdf(-1 * np.abs(wald))
    if p_value < alpha:
        count += 1
print(f'False Rejection Rate: {count / b:.3f}')

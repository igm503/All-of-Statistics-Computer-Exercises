import numpy as np
from scipy.stats import norm

Twain = np.array([.225, .262, .217, .240, .230, .229, .235, .217])
Snodgrass = np.array([.209, .205, .196, .210, .202, .207, .224, .223, .220, .201])

#Wald Test for Inequality of Means
theta = Twain.mean() - Snodgrass.mean()
se_theta = np.sqrt(Twain.std()**2 / Twain.size + Snodgrass.std()**2 / Snodgrass.size)
wald = np.abs(theta / se_theta)
p = 2 * (1 - norm.cdf(wald))
z = norm.ppf(.975)
print(f'Diff in Means Estimate: {theta:.3f}')
print(f'SE of Diff in Means: {se_theta:.3f}')
print(f'Wald Statistic: {wald:.3f}')
print(f'p-value for Estimate: {p:.4f}')
print(f'95% Conf Interval of Diff: ({theta - z * se_theta:.3f},{theta + z * se_theta:.3f})')

#Permutation Test for Inequality of Means
combined = np.append(Twain, Snodgrass)
n = combined.size
b = 100000
rng = np.random.default_rng()
index = rng.integers(0,n, size=n*b).reshape(b, n)
permutations = combined[index]
perm_diffs = permutations[:,0:8].mean(axis=1) - permutations[:,8:].mean(axis=1)
quantile = np.sum(perm_diffs > theta) / b
print('p-value for Permutation Test:', quantile)

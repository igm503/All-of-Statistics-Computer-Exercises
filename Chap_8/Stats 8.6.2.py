import numpy as np
from scipy.stats import norm

n = 50
rng = np.random.default_rng()
trials = 1000

def var(x):
    mu = x.mean()
    n = x.size
    return np.sum((x - mu)**2) / n

def skewness(x):
    mu = x.mean()
    n = x.size
    return np.sum((x - mu)**3 / (n * var(x)**1.5))

def tboot(b, sample):
    tboot = np.zeros(b)
    boot_selection = rng.integers(low = 0, high = n, size = n * b).reshape(b, n)
    boot_samples = sample[boot_selection]
    for i in range(b):
        tboot[i] = skewness(boot_samples[i])
    return tboot

def boot_conf_intervals(tboot, estimate, alpha):
    se_skew = np.sqrt(var(tboot))
    z_alpha = norm.ppf(1 - alpha / 2)
    upper_quantile = np.quantile(tboot, 1 - alpha / 2)
    lower_quantile = np.quantile(tboot, alpha / 2)
    normal_int = (estimate - z_alpha * se_skew, estimate + z_alpha * se_skew)
    percentile_int = (lower_quantile, upper_quantile)
    pivotal_int = (2 * estimate - upper_quantile, 2 * estimate - lower_quantile)
    return normal_int, percentile_int, pivotal_int

y = rng.standard_normal(n)
x = np.e ** y


#Estimate Skewness
skew_x = skewness(x)
print(f'Skewness: {skew_x:.3f}')

#Bootstrap Confidence Intervals
tboot = tboot(1000, x)
normal, percentile, pivotal = boot_conf_intervals(tboot, skew_x, 0.05)
print(f'Normal Conf Int: {normal}')
print(f'Pivotal Conf Int: {pivotal}')
print(f'Percentile Conf Int: {percentile}')




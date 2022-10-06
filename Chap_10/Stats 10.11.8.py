import numpy as np
from scipy.stats import norm

#estimates of p
placebo = [45, 80]
chlor = [26, 75]
dim = [52, 85]
pento_100 = [35, 67]
pento_150 = [37, 85]

def wald(test):
    test_p = test[0] / test[1]
    test_n = test[1]
    placebo_p = placebo[0] / placebo[1]
    placebo_n = placebo[1]
    placebo_var = placebo_p * (1 - placebo_p) / placebo_n
    test_var = test_p * (1 - test_p) / test_n
    return (test_p - placebo_p) / np.sqrt(placebo_var + test_var)

def p_value(wald_statistic):
    return 2 * norm.cdf(-1 * np.abs(wald_statistic))

print(f'Chlor p-value: {p_value(wald(chlor)):.3f}')
print(f'Dim p-value: {p_value(wald(dim)):.3f}')
print(f'Pento 100 p-value: {p_value(wald(pento_100)):.3f}')
print(f'Pento 150 p-value: {p_value(wald(pento_150)):.3f}')

print(f'Bonferonni Cutoff: {.05 / 4}')
from tkinter import N
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng()

#Parametric Bootstrap
b = 1000
n = 50
placebo = rng.binomial(n, .6, size=b)
treatment = rng.binomial(n, .8, size=b)
t = (treatment - placebo) / n
print(f'Par Bootstrap SE of t_hat: {t.std():.3f}')

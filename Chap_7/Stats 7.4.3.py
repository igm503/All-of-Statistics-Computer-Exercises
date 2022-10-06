import numpy as np
from scipy.stats import norm
import time
from tqdm import tqdm

method = 'easy'
count = 0
trials = 1000
n = 100
alpha = 0.05
distribution = 'normal'
epsilon = np.sqrt((1/(2*n))*np.log(2/alpha))


rng = np.random.default_rng()
tic = time.perf_counter()

for i in tqdm(range(trials)):
    if distribution == 'normal':
        sample = rng.standard_normal(n)
    else:
        sample = rng.standard_cauchy(n)

    F_n = lambda x: np.sum(sample <= x.reshape(x.size,1), axis = 1) / n
    L_n = lambda x: np.maximum(F_n(x) - epsilon, 0)
    U_n = lambda x: np.minimum(F_n(x) + epsilon, 1)
    CDF = norm.cdf

    if method == 'hard':
        granularity = 100
        x_values = np.linspace(-5, 5, granularity)
        count += ((CDF(x_values) > U_n(x_values)) + 
                (CDF(x_values) < L_n(x_values))).any()

    if method == 'easy':
        #More efficient way
        count += ((CDF(sample-.001) > U_n(sample-.001)) + 
                (CDF(sample) < L_n(sample))).any()

toc = time.perf_counter()
print(1-count/trials)

total_time = round(toc-tic, 2)
if method == 'easy':
    print('time the easy way:', total_time)
else:
    print('time the hard way:', total_time)

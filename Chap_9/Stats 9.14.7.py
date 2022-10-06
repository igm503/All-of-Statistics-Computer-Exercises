import numpy as np
from scipy.stats import norm

n = 200
x_1 = 160
x_2 = 148
p_1_hat = x_1 / n
p_2_hat = x_2 / n
psi_hat = p_1_hat - p_2_hat

#Delta Method estimate of 90% conf interval for psi_hat
z = norm.ppf(.95)
se_psi = np.sqrt((p_1_hat * (1 - p_1_hat) / n)+(p_2_hat * (1 - p_2_hat) / n)) 
print(f'Delta Method 90% Conf Int for psi_hat: ({psi_hat - z * se_psi:.3f},{psi_hat + z * se_psi:.3f})')

#Parametric Bootstrap for 90% Conf Int for psi_hat
trials = 1000
rng = np.random.default_rng()
boot_1 = rng.binomial(n, p_1_hat, trials)
boot_2 = rng.binomial(n, p_2_hat, trials)
tboot = boot_1 / n - boot_2 / n
se_psi = tboot.std()
print(f'Bootstrap 90% Conf Int for psi_hat: ({psi_hat - z * se_psi:.3f},{psi_hat + z * se_psi:.3f})')
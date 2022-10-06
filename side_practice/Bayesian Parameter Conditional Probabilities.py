import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

#PMF for p if an n=1 Bernouli trial yields X=1

#granularity is number of discrete possibilites for p, 
#distributed evenly between 0 and 1 inclusive
granularity = 100000

def pmf_p(p, granularity):
    return 2 * p / granularity**2

x = np.linspace(0,1,granularity)
y = pmf_p(x, granularity)

#plt.plot(x,y)
#plt.show()

#PDF for p if prior distribution of p is Uniform(0,1)
n = 50
p_true = .5
rng = np.random.default_rng()
x = rng.binomial(1, p_true, size=n)
k = x.sum()
print(k)

def prior_x(k, n):
    return gamma(k + 1) * gamma(n + 1 -k) / gamma(n + 2)

def joint_p_and_x(p, k, n):
    return p**k * (1 - p)**(n-k)

def posterior_p(p, k, n):
    return joint_p_and_x(p, k, n) / prior_x(k, n)

p = np.linspace(0,1,1000)
f_p = posterior_p(p, k, n)
z = np.arange(1,n+1)
f_z = prior_x(z,n)
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(p, f_p)
ax1.set(title='Posterior PDF of p')
ax2.plot(z, f_z)
ax2.set(title='Prior PDF of x')
plt.show()

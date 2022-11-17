import numpy as np
import matplotlib.pyplot as plt

fiji_data = np.loadtxt('fijiquakes.dat', dtype='str')
magnitudes = fiji_data[1:, 4].astype('float')
n = magnitudes.size
alpha = .05
epsilon = np.sqrt(np.log(2/alpha)/(2*n))

F_n = lambda x: np.sum(magnitudes.reshape(n,) < x.reshape(x.size, 1)
                       , axis = 1)/n
U_n = lambda x: np.minimum(F_n(x) + epsilon, 1)
L_n = lambda x: np.maximum(F_n(x) - epsilon, 0)

#plot estimate of F and 95% confidence interval
granularity = 36
x = np.linspace(3.5,7,granularity)
y = F_n(x)
fig, ax = plt.subplots()

ax.plot(x, y)
ax.plot(x, U_n(x))
ax.plot(x, L_n(x))

fig.savefig("fiji CDF.png");

#estimate F(4.9) - F(4.3)
F_n_1 = float(F_n(np.array(4.9)))
F_n_2 = float(F_n(np.array(4.3)))
theta = round(F_n_1 - F_n_2, 3)

print('Estimate of F(4.9) - F(4.3):', theta)

#estimate 95% confidence interval of F_n(4.9) - F_n(4.3)
variance = lambda x: x * (1-x) / n
se = round(np.sqrt(variance(F_n_1) + variance(F_n_2)),3)
print('se of estimate:', se)
print('95% Confidence Interval: (' + str(theta - 2 * se) + 
        ', ' + str(theta + 2 * se))
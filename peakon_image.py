import numpy as np
import matplotlib.pyplot as plt

f, ax = plt.subplots(1)

x_range = np.linspace(0, 1000, 100000)

peakon_1_func = lambda x: 5*np.exp(-abs(x - 250)/100)

peakon_2_func = lambda x: 3*np.exp(-abs(x - 750)/100)

comb_func = lambda x: peakon_1_func(x) + peakon_2_func(x)

print(comb_func(x_range))

ax.scatter(x_range[::2500], peakon_1_func(x_range)[::2500], s=0.25)
ax.scatter(x_range[::2500], peakon_2_func(x_range)[::2500], s=0.25)
ax.plot(x_range, comb_func(x_range), color='k')
plt.show()
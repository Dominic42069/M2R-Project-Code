import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
from time import perf_counter

t0 = perf_counter()

peakon_number = 5 # number of peaks
sample_number = 10000 # number of x values to sample u_func
x_range = np.linspace(0, 100, sample_number) # x values u_func is sampled out
t_max = 200 # max t value to sample u_func
delta_t = 0.025 # timestep to advance u_func by each iteration
m_range = 0.5*random(size=peakon_number) + 0.5 # array of peakon heights
np.sort(m_range)
c_range = 10*random(size=peakon_number) + 5
np.sort(c_range)
offset = 20*random(size=peakon_number) - 20
np.sort(offset)[::-1]

u_func = lambda x, t: np.sum(m_range[:,None] * np.exp(-np.abs(x[None,:]-c_range[:,None]*t - offset[:,None])),axis=0)


f, ax = plt.subplots(1)

plt.ion()
plt.show()

t=0

while t <= t_max:
    plt.cla()
    u_vec = u_func(x_range, t)
    ax.plot(x_range, u_vec)
    plt.ylim([0,2])
    t1 = perf_counter()
    plt.title(f"{(t/delta_t)/(t1-t0)} fps")
    plt.pause(0.000000000000000000000000000000001)
    t += delta_t

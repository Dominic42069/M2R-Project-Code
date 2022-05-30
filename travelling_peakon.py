import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

# t0 = perf_counter()

### NUMERICAL PROPERTIES

sample_number = 10000 # number of x values to sample u_func
x_range = np.linspace(-100, 100, sample_number) # x values u_func is sampled out
t_max = 200 # max t value to sample u_func
delta_t = 0.025 # timestep to advance u_func by each iteration

### PEAKON PROPERTIES

peakon_number = 2 # number of peaks
c_range = np.array([0, 5]) # speed of each peak
m_range = np.array([1, 1]) # height of each peak
offset = np.array([0, -50]) # peak starting location

u_func = lambda x, t: np.sum(m_range[:,None] * np.exp(-np.abs(x[None,:]-c_range[:,None]*t - offset[:,None])),axis=0)

### PLOTTING ZONE

f, ax = plt.subplots(1)

plt.ion()
plt.show()

t=0

while t <= t_max:
    plt.cla()
    u_vec = u_func(x_range, t)
    ax.plot(x_range, u_vec)
    plt.ylim([0,2.5]) # set slightly higher than largest u value
    # t1 = perf_counter()
    # plt.title(f"{(t/delta_t)/(t1-t0)} fps")
    plt.pause(0.000000000000000000000000000000001)
    t += delta_t

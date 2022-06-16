import numpy as np
import matplotlib.pyplot as plt
import sys

### NUMERICAL PROPERTIES

sample_number = 10000 # number of x values to sample u_func
x_range = np.linspace(0, 200, sample_number) # x values u_func is sampled out
t_max = 200 # max t value to sample u_func
delta_t = 2 # timestep to advance u_func by each iteration

### PEAKON PROPERTIES

peakon_number = 2 # number of peaks
c_range = 5*np.array([1, 3]) # speed of each peak
offset = np.array([5, 5]) # peak starting location.

u_func = lambda x, t: np.sum(1/25 * c_range[:,None] * np.exp(-np.abs((x[None,:]-c_range[:,None]*t) % 200 - offset[:,None])),axis=0)

### PLOTTING ZONE

f, ax = plt.subplots(1)

t=0

while t <= t_max:
    u_vec = u_func(x_range, t) + 0.1*t
    ax.plot(x_range, u_vec)
    plt.ylim([0,10]) # set slightly higher than largest u value
    plt.pause(0.000000000000000000000000000000001)
    t += delta_t

plt.show()
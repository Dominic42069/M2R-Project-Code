import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

# t0 = perf_counter()

### NUMERICAL PROPERTIES

sample_number = 10000  # number of x values to sample u_func
x_range = np.linspace(0, 300, sample_number)  # x values u_func is sampled out
t_max = 10  # max t value to sample u_func
delta_t = 0.025  # timestep to advance u_func by each iteration

### PEAKON PROPERTIES

c_range = np.array([2.5, 5])  # speed of each peak
offset = np.array([75, 25])  # peak starting location
spread = 5  # width of each peak

u_func = lambda x, t: np.sum(
    c_range[:, None]
    * np.exp(-np.abs((x[None, :] - offset[:, None]) / spread - c_range[:, None] * t)),
    axis=0,
)

### PLOTTING ZONE

f, ax = plt.subplots(1)

plt.ion()
plt.show()

t = 0

while t <= t_max:
    plt.cla()
    u_vec = u_func(x_range, t)
    ax.plot(x_range, u_vec)
    plt.ylim([0, 10])  # set slightly higher than largest u value
    # t1 = perf_counter()
    plt.title(f"t={t}")
    if t % 2 == 0:
        plt.pause(1)
        plt.savefig(f"2-peakon-t={int(t)}s.png")
    plt.pause(0.000000000000000000000000000000001)
    t += delta_t
    t = round(t, 3)

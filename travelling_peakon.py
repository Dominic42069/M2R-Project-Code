import numpy as np
import matplotlib.pyplot as plt
import sys

### NUMERICAL PROPERTIES

sample_number = 10000  # number of x values to sample u_func
endpoint = 25
x_range = np.linspace(0, endpoint, sample_number)  # x values u_func is sampled out
t_max = 2.5  # max t value to sample u_func
delta_t = 0.1  # timestep to advance u_func by each iteration

### PEAKON PROPERTIES

c_range = 5 * np.array([3, 6])  # speed of each peak
offset = np.array([5, 5])  # peak starting location.

u_func = lambda x, t: np.sum(
    1
    / 25
    * c_range[:, None]
    * np.exp(-np.abs((x[None, :] - c_range[:, None] * t) % endpoint - offset[:, None])),
    axis=0,
)

### PLOTTING ZONE

f, ax = plt.subplots(1)

t = 0

colours = np.linspace(0, 1, int(t_max / delta_t), endpoint=False)

hex = [hex(i)[-2:] for i in range(256)]

hex_colours = [f"#{r}00{b}" for r, b in zip(hex, hex[::-1])]
reduced_hex_colours = [hex_colours[i] for i in range(256) if i % 8 == 0]
reduced_hex_colours = [col.replace("x", "0") for col in reduced_hex_colours]

while t <= t_max:
    u_vec = u_func(x_range, t) + t / delta_t
    ax.plot(x_range, u_vec, color=reduced_hex_colours[int(t / delta_t)])
    plt.xticks([])
    plt.yticks([])
    plt.ylim([0, 25])  # set slightly higher than largest u value
    plt.pause(0.000000000000000000000000000000001)
    t += delta_t

plt.savefig("2-peakon_waterfall")
plt.show()

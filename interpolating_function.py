import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from time import perf_counter
import sys

### NOTE: J = j+1


def get_x_i(x):
    return (np.floor(x / delta_x)).astype(int)


def get_t_j(t):
    return int(np.floor(t / delta_t))


### SETUP AREA

T = 10  # maximum time value
l = 10000  # number of time steps
delta_t = T / l  # length of time steps
n = 2 ** 14  # term number in sequence
sample_number = 25000  # number of x values
a = 40  # x is on interval [0,a]
kappa = 0.001  # kappa in the equation
delta_x = a / sample_number  # length between sample points in space
c = 1 / (1 + 2 * n ** 2 * (1 - np.exp(-kappa)))
h = 2 ** (-40)

t_range = np.linspace(0, T, l+1)
x_range = np.linspace(0, a, sample_number)

### INITIAL PROFILE

u0 = lambda x: c * (np.cosh(np.minimum(x, a - x) - a / 2) / (np.sinh(a / 2)))

g = np.array(
    [
        c
        * (
            np.exp(-kappa * i)
            + np.exp(kappa * (i - sample_number)) / (1 - np.exp(-kappa * sample_number))
        )
        for i in range(sample_number)
    ]
)

f, ax = plt.subplots(1)

init_u_vec = u0(x_range)  # u_0^n,l

current_u_vec = init_u_vec
current_m_vec = np.zeros_like(current_u_vec)
next_m_vec = current_m_vec

plt.ion()
plt.show()

ax.plot(x_range, init_u_vec)
plt.pause(0.0000000000000000000000000000001)

### CREATE FINITE DIFFERENCE MATRICES

# 2nd derivative matrix
sec_zeros_array = np.zeros(len(current_u_vec) - 3)
sec_circ_vec = np.concatenate((np.array([-2, 1]), sec_zeros_array, np.array([1])))
sec_deriv_mat = 1 / h * np.transpose(circulant(sec_circ_vec))

# Forward 1st derivative matrix
fst_zeros_array = np.zeros(len(current_u_vec) - 2)
fwd_circ_vec = np.concatenate((np.array([-1, 1]), fst_zeros_array))
fwd_deriv_mat = 1 / h * np.transpose(circulant(fwd_circ_vec))

# Backward 1st derivative matrix
bwd_circ_vec = np.concatenate((np.array([1]), fst_zeros_array, np.array([-1])))
bwd_deriv_mat = 1 / h * np.transpose(circulant(bwd_circ_vec))

# D matrix (average of fwd and backwards)
D_mat = 1 / 2 * (fwd_deriv_mat + bwd_deriv_mat)

t = 0

while t < T:
    plt.cla()
    current_m_vec = current_u_vec - np.matmul(sec_deriv_mat, current_u_vec)  # STEP 2
    next_m_vec = current_m_vec + delta_t * (  # STEP 3
        -np.matmul(bwd_deriv_mat, current_m_vec * current_u_vec)
        - current_m_vec*np.matmul(D_mat, current_u_vec)
    )
    next_u_vec = (h * ifft(a * fft(g) * fft(next_m_vec))).real  # STEP 4 (may be broken)
    u_interp = (
        lambda x, t: n
        / delta_t
        * (
            (t_range[get_t_j(t) + 1] - t)
            * (
                (x_range[(get_x_i(x) + 1) % sample_number] - x)
                * current_u_vec[get_x_i(x) % sample_number]
                + (x - x_range[get_x_i(x) % sample_number])
                * current_u_vec[(get_x_i(x) + 1) % sample_number]
            )
            + (t - t_range[get_t_j(t)])
            * (
                (x_range[(get_x_i(x) + 1) % sample_number] - x)
                * next_u_vec[get_x_i(x) % sample_number]
                + (x - x_range[get_x_i(x) % sample_number])
                * next_u_vec[(get_x_i(x) + 1) % sample_number]
            )
        )
    )  # STEP 5 (also maybe broken)
    ax.plot(x_range, u_interp(x_range, t))
    plt.title(f"t={t}")
    plt.pause(0.000000000000000000001)
    t += delta_t
    current_u_vec = next_u_vec

sys.exit(0)

import numpy as np
import cupy as cp

import ode
import acceleration as acc
from initialize import initialize, initialize_spiral
from animation import animate_n_body

import time


def nbody_df(t, y, acc_f, m, shape=(2, -1, 3)):
    xp = cp.get_array_module(y)
    x, v = y.reshape(shape)
    a = acc_f(x, m)
    return xp.concatenate((v, a)).flatten()


N = 2000
timespan = 50
m, y0 = initialize_spiral(N, cp)

# Run the solver and take wall-clock reading.
tstart = time.time()
s = ode.erk4(lambda t, y: nbody_df(t, y, acc.vec, m), (0, timespan), y0, 1e-2)
tstop = time.time()

# print(s.message)
print(f"ODE solver runtime: {(tstop - tstart)} seconds. ")
animate_n_body(s, N, f"nbody_spiral_{N}_{timespan}_cupy")

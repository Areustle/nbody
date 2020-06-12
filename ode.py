import numpy as np
import cupy as cp
from scipy.integrate import solve_ivp


def scipy_solve(f, span, y0, method="RK45"):
    _, timespan = span
    s = solve_ivp(f, span, y0, method=method,
                  t_eval=np.linspace(0, timespan, int(timespan/1e-1)))
    return s


def erk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2., y + h*k1*.5)
    k3 = f(t + h/2., y + h*k2*.5)
    k4 = f(t + h, y + h*k3)
    yn = y + (h/6.)*(k1 + 2.*k2 + 2.*k3 + k4)
    tn = t + h
    return yn, tn


def erk4(f, tspan, y0, h=1e-4):
    xp = cp.get_array_module(y0)

    t, tU = tspan
    steps = int((tU - t) // h)

    y = y0
    ts = [t]
    ys = xp.empty((steps, *y.shape))

    # while t < tU:
    for step in range(steps):
        yn, tn = erk4_step(f, t, y, h)
        # ys.append(yn)
        ts.append(tn)
        ys[step, :] = yn
        t = tn
        y = yn

    # print(type(ys))
    # print(type(y))
    ys = cp.asnumpy(ys)
    ys = np.transpose(ys, (1, 0))

    return {'y': ys, 't': np.array(ts)}

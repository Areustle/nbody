import numpy as np


def pseudo_uniform_shell(N, r, dim=3):
    theta = np.random.uniform(0, 2*np.pi, N)
    phi = np.arccos(1 - 2*np.random.uniform(0, 1, N))
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    return r*np.column_stack((sinphi*costheta, sinphi*sintheta, cosphi))


def initialize(N, xp=np):
    x = xp.random.uniform(-100, 100, (N, 3))
    v = xp.random.uniform(-1, 1, (N, 3))
    y0 = xp.concatenate((x, v)).flatten()
    m = xp.random.uniform(1, 5, N)
    return m, y0


def initialize_spiral(N, xp=np):
    x = xp.random.normal(0, 50, (N, 3))
    x[:, 2] *= 0.1

    v = x[:, xp.array((1, 0, 2))]
    v[:, 0] *= -1
    v /= np.sqrt(xp.linalg.norm(v))
    # v += xp.random.uniform(-1, 1, (N, 3))

    y0 = xp.concatenate((x, v)).flatten()
    m = xp.random.uniform(1, 5, N)
    return m, y0

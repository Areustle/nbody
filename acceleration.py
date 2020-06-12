import numpy as np
import cupy as cp

from scipy.linalg.blas import zhpr, dspr2, zhpmv


def loops(x, m):
    N = m.size
    x = x.reshape(-1, 3)
    a = np.empty((N, 3))

    for i in range(N):
        a_sum = 0
        for j in range(N):
            d = x[j] - x[i]
            r = np.linalg.norm(d) if np.linalg.norm(d) != 0 else 1
            a_sum += (m[j]*d)/(r**3)
        a[i] = a_sum

    return a


def vec(x, m):
    xp = cp.get_array_module(x)
    M = m.reshape((1, -1, 1))*m.reshape((-1, 1, 1))
    d = x.reshape((1, -1, 3)) - x.reshape((-1, 1, 3))
    r = xp.linalg.norm(d, axis=2)
    # Avoid divide by zero warnings
    r[r == 0] = 1
    a = (M*d)/xp.expand_dims(r, 2)**3
    return a.sum(axis=1)/m.reshape(-1, 1)


def blas_cpu(pos, m):

    n = m.size

    # trick: use complex Hermitian to get the packed anti-symmetric
    # outer difference in the imaginary part of the zhpr answer
    # don't want to sum over dimensions yet, therefore must do them one-by-one
    trck = np.zeros((3, n*(n + 1) // 2), complex)
    for a, p in zip(trck, pos.T - 1j):
        # does  a  ->  a + alpha pp^H
        zhpr(
            n,    # matrix dimension
            -2,    # real scalar
            p,    # complex scalar
            a,    # output: packed Hermitian nxn matrix a i.e. n(n+1)/2 vector
            1,    # p stride
            0,    # p offset
            0,    # lower triangular storage
            1    # overwrite, inplace
        )

    # as a by-product we get pos pos^T:
    ppT = trck.real.sum(0) + 6

    # now compute matrix of squared distances ...
    # ... using (A-B)^2 = A^2 + B^2 - 2AB
    # ... that and the outer sum X (+) X.T equals X ones^T + ones X^T
    dspr2(n, -0.5, ppT[np.r_[0, 2:n + 1].cumsum()], np.ones((n, )), ppT, 1, 0,
          1, 0, 0, 1)

    # does  a  ->  a + alpha x y^T + alpha y x^T    in packed symmetric storage
    # scale anti-symmetric differences by distance^-3
    np.divide(trck.imag, ppT*np.sqrt(ppT), where=ppT.astype(bool),
              out=trck.imag)

    # it remains to scale by mass and sum
    # this can be done by matrix multiplication with the vector of m ...
    # ... unfortunately because we need anti-symmetry we need to work
    # with Hermitian storage, i.e. complex numbers, even though the actual
    # computation is only real:
    out = np.zeros((3, n), complex)

    for a, o in zip(trck, out):
        zhpmv(n, 0.5, a, -1j*m, 1, 0, 0, o, 1, 0, 0, 1)
        # multiplies packed Hermitian matrix by vector
    return out.real.T

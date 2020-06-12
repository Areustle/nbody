import numpy as np
import cupy as cp
import acceleration as acc
import initialize as init
from timeit import timeit
import matplotlib.pyplot as plt


def test_perf(f, xp, input_xp, max_ms=1000, max_exp=13):
    n, t = [], []
    for e in range(max_exp):
        if not t or t[-1] < max_ms:
            N = 2**e
            n.append(N)
            dat = init.initialize(N, input_xp)
            mass, y0 = dat
            x, _ = y0.reshape((2, -1, 3))
            t.append(
                timeit('f(xp.asarray(x), xp.asarray(mass))', globals=locals(),
                       number=10)*100)

    return (n, t)


fig, ax = plt.subplots(2)

fig.suptitle("N Body Acceleration Execution Times")

line_loops = test_perf(acc.loops, np, np)
line_npvec = test_perf(acc.vec, np, np)
line_blas = test_perf(acc.blas_cpu, np, np)
line_cpcpu = test_perf(acc.vec, cp, np)
line_cpgpu = test_perf(acc.vec, cp, cp)


ax[0].plot(*line_loops, label="CPU Loops", color='blue')
ax[0].plot(*line_npvec, label="CPU Numpy", color='navy')
ax[0].plot(*line_blas,  label="CPU Blas", color='deepskyblue')
ax[0].plot(*line_cpcpu, label="GPU Cupy CPU init", color="lime")
ax[0].plot(*line_cpgpu, label="GPU Cupy", color="green")
ax[0].set_ylim(0, 1000)
ax[0].set_ylabel("Execution Time (ms)")
ax[0].set_xlabel("N")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(*line_loops, label="CPU Loops", color='blue')
ax[1].plot(*line_npvec, label="CPU Numpy", color='navy')
ax[1].plot(*line_blas,  label="CPU Blas", color='deepskyblue')
ax[1].plot(*line_cpcpu, label="GPU Cupy CPU init", color='lime')
ax[1].plot(*line_cpgpu, label="GPU Cupy", color="green")
ax[1].set_ylabel("Log Execution Time (log ms)")
ax[1].set_xlabel("N")
ax[1].legend()
ax[1].set_yscale("log")
ax[1].grid(True)

plt.show()

# N = 2**4

# mass_np, y0_np = init.initialize(N, np)
# x_np, _ = y0_np.reshape((2, -1, 3))

# print(
#     f"loops:  {timeit('acc.loops(x_np, mass_np)', globals=globals(), number=10)*100:10.3f} ms"
# )
# print(
#     f"np:     {timeit('acc.vec(x_np, mass_np)', globals=globals(), number=10)*100:10.3f} ms"
# )
# print(
#     f"blas:   {timeit('acc.blas_cpu(x_np, mass_np)', globals=globals(), number=10)*100:10.3f} ms"
# )

# mass_cp = cp.asarray(mass_np)
# x_cp = cp.asarray(x_np)

# print(
#     f"cp_host:{timeit('acc.vec(x_cp, mass_cp)', globals=globals(), number=10)*100:10.3f} ms"
# )

# mass_cp, y0_cp = init.initialize(N, cp)
# x_cp, _ = y0_cp.reshape((2, -1, 3))

# print(
#     f"cp_dev: {timeit('acc.vec(x_cp, mass_cp)', globals=globals(), number=10)*100:10.3f} ms"
# )

# assert np.allclose(acc.blas_cpu(x_np, mass_np), acc.vec(x_np, mass_np))
# assert np.allclose(acc.loops(x_np, mass_np), acc.vec(x_np, mass_np))

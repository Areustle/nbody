import numpy as np
import matplotlib.pyplot as pt
from matplotlib.animation import FuncAnimation
from time import process_time


def equalize_step(seq, interval, maximum=0):
    """
    From a given time sequence return a subsequence whose entries closest
    match the inpu tinterval between elements.
    Used to smooth out and drop frames from animations an plots
    """

    if maximum == 0:
        maximum = seq[-1]
    goal = seq[0] + interval
    ret = [0]

    for j, t in enumerate(seq):
        if t > maximum:
            break
        if t >= goal:
            if (goal - seq[j - 1]) < (t - goal):
                ret.append(j - 1)
                goal = seq[j - 1] + interval
            else:
                ret.append(j)
                goal = t + interval
    return ret


def arg_time(seq, target):
    """
    find the index in a time sequence that is closest to target.
    """
    for j, t in enumerate(seq):
        if t >= target:
            return j


def classical_erk4(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2., y + h*k1*.5)
    k3 = f(t + h/2., y + h*k2*.5)
    k4 = f(t + h, y + h*k3)
    yn = y + (h/6.)*(k1 + 2.*k2 + 2.*k3 + k4)
    tn = t + h
    return yn, tn


def erk4(f, t, tU, y0, h=1e-5):
    '''
    Extended Runge Kutta 4 stage method for a system of first order
    differential equations
    Inputs:
    -------
    f:      System of differential equations being approximated
    t:      Initial time of the system (t=0)
    tU:     Ending time of the approximation.
    y0:     Initial conditions of the system
    h:      Step size.

    Returns:
    --------
    sol:    A matrix of y_n's determined by the scheme. Rows (first index)
            correspond to elements of the input vector. For example sol[i,:]
            represents how component y0[i] changes over time. Columns (second
            index) correspond to the result of new steps. For example sol[:,-1]
            represents the value y_n at the ending time of the simulation.
    ts:     A vector of time sequences. Each element is exactly h larger than
            the previous, starting at ts[0]=0.
    '''

    Dim = y0.size
    ts = [t]
    ys = [y0]
    y = y0

    while t < tU:
        yn, tn = classical_erk4(f, t, y, h)
        ys.append(yn)
        ts.append(tn)
        t = tn
        y = yn

    N = len(ys)
    sol = np.zeros([Dim, N])

    for j in range(N):
        for i in range(Dim):
            sol[i, j] = ys[j][i]

    return sol, ts


def q24(n=40, h=1e-5):
    """
    Fully interacting N-Body Problem
    """
    def f24(t, y):
        """
        inputs:
        t:  ignored
        y:  a flat vector containing x,y position & velocity for each planet

        returns:
        r:  an updated y. if y is y_n, r is y_{n+1}
        """

        N = y.size//4
        r = np.zeros(N*4)

        for j in range(N):
            j_asteroid = 4*j
            xj = y[j_asteroid:j_asteroid + 2]
            vjp = xj/(np.linalg.norm(xj)**3)
            r[j_asteroid] = y[j_asteroid + 2]
            r[j_asteroid + 1] = y[j_asteroid + 3]
            r[j_asteroid + 2] -= vjp[0]
            r[j_asteroid + 3] -= vjp[1]

            for i in range(j + 1, N):
                i_asteroid = 4*i
                xi = y[i_asteroid:i_asteroid + 2]
                normi = np.linalg.norm(xi - xj)
                if normi != 0:
                    vip = (1e-4/N)*((xi - xj)/(normi**3))
                    r[j_asteroid + 2] += vip[0]
                    r[j_asteroid + 3] += vip[1]
                    r[i_asteroid + 2] -= vip[0]
                    r[i_asteroid + 3] -= vip[1]

                else:
                    print("Maximum acceleration exceeded.")

        return r

    # Initial Conditions. Note Theta_j is left in "cluster" form. That is
    # the planets are not uniformly distributed on the unit circle.
    yinit = np.zeros(4*n)

    for j in range(n):
        i = 4*j
        theta_j = (j/(2*n*np.pi))
        omega_j = np.random.normal(0, 0.5)
        yinit[i] = np.cos(theta_j)
        yinit[i + 1] = np.sin(theta_j)
        yinit[i + 2] = 0 + omega_j* -np.sin(theta_j)
        yinit[i + 3] = 0 + omega_j*np.cos(theta_j)

    # Run the solver and take wall-clock reading.
    tstart = process_time()
    sol, ts = erk4(f24, 0, 0.2, yinit, h)
    tstop = process_time()

    print("ODE solver runtime: {} seconds. ".format((tstop - tstart)))

    # Animation of orbits
    fig, ax = pt.subplots()
    fig.set_tight_layout(True)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    xdata, ydata = [], []
    ax.plot(0, 0, 'ro')
    ln, = ax.plot([], [], 'bo', markersize=1, animated=True)

    def update(j):
        xdata = [sol[4*i, j] for i in range(n)]
        ydata = [sol[4*i + 1, j] for i in range(n)]
        ln.set_data(xdata, ydata)
        return ln,

    ani = FuncAnimation(fig, update, frames=equalize_step(ts, 0.01),
                        interval=50, blit=True)

    # ani.save('animations/areustle_loop_nbody_cluster.mp4', fps=30)

    pt.show()
    pt.clf()

    # Plot the time steps.
    fig, ax = pt.subplots(2, 3)
    fig.set_tight_layout(True)
    for k, time in enumerate([0, 0.1, 0.5, 1., 1.5, 2.]):
        ax.flat[k].set_xlim(-1.1, 1.1)
        ax.flat[k].set_ylim(-1.1, 1.1)
        ax.flat[k].plot(0, 0, 'ro')
        j = arg_time(ts, time)
        ax.flat[k].scatter([sol[4*i, j] for i in range(n)],
                           [sol[4*i + 1, j] for i in range(n)], s=0.5)
        ax.flat[k].set_xlabel("Time {0}".format(time))

    pt.show()
    pt.clf()


q24()

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation


def equalize_step(seq, interval, maximum=0):
    """
    From a given time sequence return a subsequence whose entries most nearly
    match the input interval between elements.  Used to smooth out and drop
    frames from animations and plots
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


def animate_n_body(sol, n, name=None):
    ts = sol['t']
    ys = sol['y']
    # print(ts.shape, ys.shape)

    # Animation of orbits
    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_xlim(-100.1, 100.1)
    ax.set_ylim(-100.1, 100.1)
    ax.set_zlim(-100.1, 100.1)
    xdata, ydata, zdata = [], [], []
    ln, = ax.plot([], [], [], 'bo', markersize=1, animated=True)

    def update(j):
        xdata = [ys[2*i, j] for i in range(n)]
        ydata = [ys[2*i + 1, j] for i in range(n)]
        zdata = [ys[2*i + 2, j] for i in range(n)]
        ln.set_data(xdata, ydata)
        ln.set_3d_properties(zdata)
        return ln,

    ani = FuncAnimation(fig, update, frames=equalize_step(ts, 0.1),
                        interval=50, blit=True)
    if name:
        ani.save(f'animations/{name}.mp4', fps=30)

    plt.show()
    plt.clf()

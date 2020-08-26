import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from threading import Thread
from time import time, sleep


def __func_iter(data_mat, del_x, del_y):
    new_data_mat = np.copy(data_mat)

    for i in range(1, data_mat.shape[0] - 1):
        for j in range(1, data_mat.shape[1] - 1):
            val = (del_y ** 2) * (data_mat[i + 1, j] + data_mat[i - 1, j])
            val += (del_x ** 2) * (data_mat[i, j + 1] + data_mat[i, j - 1])
            val /= 2 * (del_x ** 2 + del_y ** 2)
            new_data_mat[i, j] = val
    return new_data_mat


def solve_lp_rect(boundary_conds: tuple, Lx, Ly,
                  del_x=1, del_y=1, accuracy=1E-2):
    global data_mat, iter_n, acc  # remove this if not using animation
    """
    Solves laplacian eqn in a rect boundary.
    ---------------------------------------

    x, y: represents the first and second argument of the req fn.

    boundary_conds: a list of boundary conds (callable function)
    index 0: f(x, 0)    # function of x
    index 1: f(x, Ly)   # function of x
    index 2: f(0, y)    # function of y
    index 3: f(Lx, 0)   # function of y

    Lx, Ly: are the limits of the boundary.

    accu: accuracy of the solution. (=1E-2)
    del_x: delta x to use. (=0.1)
    del_y: delta y to use. (=0.1)
    """
    x_divs = np.arange(0, Lx, del_x)
    y_divs = np.arange(0, Ly, del_y)

    data_mat = np.zeros((x_divs.shape[0], y_divs.shape[0]))
    data_mat[:, 0] = np.vectorize(boundary_conds[0])(x_divs)
    data_mat[:, -1] = np.vectorize(boundary_conds[1])(x_divs)
    data_mat[0, :] = np.vectorize(boundary_conds[2])(y_divs)
    data_mat[-1, :] = np.vectorize(boundary_conds[3])(y_divs)

    acc = np.inf
    last_pritnted = time()
    iter_n = 0
    while (acc >= accuracy):
        new_data_mat = __func_iter(data_mat, del_x, del_y)
        acc = np.max(np.abs(new_data_mat - data_mat))
        data_mat = new_data_mat
        iter_n += 1
        sleep(sleep_after_draw)

    return data_mat


# Function related config
Lx = 100
Ly = 50
u_x0 = u_x_ly = lambda x: 10 * np.cos(2 * np.pi * x / Lx)
u_y0 = u_y_lx = lambda y: 10 * np.cos(2 * np.pi * y / Ly)

boundary_conds = [u_x0, u_x_ly, u_y0, u_y_lx]
del_x = 1
del_y = 1
accuracy = 1E-4

main_th = Thread(target=solve_lp_rect,
                 args=(boundary_conds, Lx, Ly, del_x, del_y, accuracy),
                 daemon=True)
main_th.start()
# same as:
# data_mat = solve_lp_rect(boundary_conds, Lx, Ly, del_x, del_y, accuracy)

# Plot related config
sleep_after_draw = 0  # 0 will skip (for smooth animation)
fig = plt.figure()
fig.set_size_inches(12, 6)

plot = plt.imshow(data_mat.T)
plt.colorbar(plot, orientation='vertical')
plt.gca().xaxis.tick_bottom()
plt.gca().invert_yaxis()


def update(*args):
    plot.set_data(data_mat.T)
    status = "Accuracy achieved: {:.2}% (After {} iteration)".\
        format(accuracy / acc * 100, iter_n)
    print(status, end='\r')
    plt.title(status)
    return [plot, ]


ani = FuncAnimation(fig, update, interval=167)
#ani.save('animation.gif', writer='imagemagick')
plt.show()

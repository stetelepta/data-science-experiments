from matplotlib import gridspec
from pylab import cm
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


def plot_predictions(probabilities, y_test, points_test, **params):
    # reshape each column to its 2D version
    probs_matrix = probabilities.reshape(params['grid_width'], params['grid_height'])

    # new figure
    plt.figure(figsize=(20, 10))

    # # set fig size
    # gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

    # # plot prediction
    # plt.subplot(gs[0])
    # plt.title("prediction 2D histogram", y=1.08)
    # plt.axis('off')

    # plt.imshow(probs_matrix)

    # set style
    sns.set_style("whitegrid")

    # prepare meshgrid
    x_mesh = np.arange(0, params['grid_width'], 1)
    y_mesh = np.arange(0, params['grid_height'], 1)

    # plot contour of prediction
    xx, yy = np.meshgrid(x_mesh, y_mesh)

    # probabilities
    z = probs_matrix

    # interpolate, so we can make nice plots
    f = interpolate.interp2d(x_mesh, y_mesh, z, kind='cubic')

    # prepare grid 10 times as large
    xx_new = np.arange(0, params['grid_width'], 0.001 * params['grid_width'])
    yy_new = np.arange(0, params['grid_height'], 0.001 * params['grid_height'])

    # interpolate z
    z_interpolated = f(xx_new, yy_new)

    cmap = cm.get_cmap('PiYG', 11)

    plt.contourf(xx_new, yy_new, z_interpolated.reshape(1000, 1000), cmap=cmap)
    plt.title("prediction: contour", y=1.08)
    ax = plt.gca()
    ax.axis([0, 10, 0, 10])
    ax.set_aspect('equal')
    plt.show()

    # # plot true values
    # plt.subplot(gs[2])
    # plt.title("true value: (%.2f, %.2f)" % (points_test[0], points_test[1]), y=1.08)
    # plt.axis('off')
    # plt.imshow(y_test.reshape(params['grid_width'], params['grid_height']))

    # with plt.style.context(('seaborn-darkgrid')):
    #     # plot sensor distances
    #     plt.subplot(gs[3])
    #     plot_sensors_distances(points_test, params['sensors'], params['grid_width'], params['grid_height'])


def plot_sensors_distances(point, sensors, grid_width, grid_height):
    # point: (2, 1) array with 2D true coordinates of point
    # sensors: (2, nr_sensors) array with sensor locations
    # grid_width, grid_width: for drawing the grid

    # reshape raveled sensors to (2, nr_sensors) matrix
    sensors = sensors.reshape(-1, 2).T

    # get current axis
    ax = plt.gca()

    # set margin for plotting grid
    margin = 0

    # setup axis
    ax.set_xticks(np.arange(0-margin, grid_width+margin, 1), minor=False)
    ax.set_yticks(np.arange(0-margin, grid_height+margin, 1), minor=False)
    ax.axis([0-margin, grid_width+margin, grid_height+margin, -margin])
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.xaxis.tick_top()

    ax.set_aspect('equal')

    # plot grid
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            grid_width,
            grid_height,
            fill=False,      # remove background
            color='darkgrey'
        )
    )

    p1_y = plt.Circle((point[0], point[1]), 0.2, color='darkviolet', zorder=2)
    ax.add_artist(p1_y)

    # plot title
    plt.title("(%.2f, %.2f)" % (point[0], point[1]), y=1.08)

    # plot sensors
    for i in range(0, sensors.shape[1]):
        s = plt.Circle((sensors[0, i], sensors[1, i]), 0.5, color='k', zorder=1)
        ax.add_artist(s)

    # plot distances
    for i in range(0, sensors.shape[1]):
        # calculate distance between sensor and points
        dx = point[0] - sensors[0, i]
        dy = point[1] - sensors[1, i]
        distance = np.sqrt(np.square(dx)+np.square(dy))

        # plot range around sensor
        r = plt.Circle((sensors[0, i], sensors[1, i]), distance, color='k', zorder=1, fill=False)
        ax.add_artist(r)

        # draw arrow from sensor to point
        ax.arrow(sensors[0, i], sensors[1, i], dx, dy, head_width=0, head_length=0, fc='k', ec='k', lw=0.1, zorder=1)

        ax.text(sensors[0, i] + dx/2, sensors[1, i] + dy/2, u'%.2f' % distance, fontsize=8)

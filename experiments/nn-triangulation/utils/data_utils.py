from scipy import ndimage
import numpy as np


def get_simulated_points(nr_points, minx=0, maxx=1, miny=0, maxy=1):
    """
    Input
    - nr_points: number of points to generate
    - minx, maxx, miny, maxy: points are generated in the range ([minx - maxx], [miny - maxy])

    Return (2, nr_points) array containing the random coordinates
    """

    # random values between 0 - 1
    points = np.random.rand(2, nr_points)

    # map x and y values between minx - maxx, miny - maxy
    points[0, :] = np.interp(points[0, :], [0, 1], [minx, maxx])
    points[1, :] = np.interp(points[1, :], [0, 1], [miny, maxy])

    return points


def get_distances(points, sensors):
    """
    Input
    - points: (2, nr_points) array with points
    - sensors: (2, nr_sensors) array with sensor locations
    Returns
    - distances: (nr_sensors, nr_points) array with distances between sensor and points
    """
    distances = np.zeros((sensors.shape[1], points.shape[1]))
    index = 0
    for s in sensors.T:
        distances[index, :] = np.linalg.norm(points - s.reshape(2, 1), axis=0, keepdims=True)
        index += 1
    return distances


def get_output_vector(points, grid_width, grid_height, sigma):
    """
    Input
    - points, (2, nr_points) array with points
    - grid_width, grid_height: size of the grid
    - sigma, determines the amount of gaussian noise that is added to the output vector
    Returns
    - (n_y, nr_points) matrix with column vectors containing the probabilities for each bin, n_y = grid_width * grid_height

    Important:
    Because want the output to conveniently match array indices, we flip the rows in the array.
    Point have their origin top-left, so a point p has column vector (x, y)
    The resulting index in the (grid_width x grid_height) matrix for point p is v[y, x]

    """
    nr_points = points.shape[1]
    n_y = grid_width * grid_height

    # a point (3.4, 2.1) will be in col=3, row=2
    bins = np.floor(points)

    # flip rows, so we get (row, col) indices
    bins = np.flip(bins, 0)

    # we use bins as indices, so cast to in
    bins = bins.astype(int)

    # initialize output matrices for each point
    output = np.zeros((nr_points, grid_height, grid_width))
    output_noise = np.zeros((nr_points, grid_height, grid_width))

    # initialize y as (n_y, nr_points) matrix
    y = np.zeros((n_y, nr_points))

    for i in range(0, points.shape[1]):
        # set output to 1 at bin index for each point
        output[i][tuple(bins[:, i])] = 1

        # add gaussian noise to output
        output_noise[i] = ndimage.filters.gaussian_filter(output[i], sigma)

        # create output vector
        y[:, i] = output_noise[i].ravel()

    return y


def prepare_dataset(transpose=True, **params):
    """
    Input (required):
      params['nr_points']  # nr of points to simulate
      params['grid_width']  # generates random points between (0, 0) and (grid_width, grid_height)
      params['grid_height']
      params['sensors']  # (1, 2n) vector with coordinates of the sensors (x1, y1, x2, y2, .. ,xn, yn)
      params['sigma']  # determines the amount of gaussian noise that is added to the output vector
    Input (optional):
      transpose: when True, return (nx, m) matrix (keras style), when False return (m, nx)

    Returns:
      - x: distances from points to sensors
      - y: (n_y, nr_points) matrix with column vectors containing the probabilities for each bin, n_y = grid_width * grid_height
      - points: coordinates for each simulated point
    """

    # get simulated points
    points = get_simulated_points(params['nr_points'], maxx=params['grid_width'], maxy=params['grid_height'])

    # get all sensor distances, reshape ravel
    x = get_distances(points, params['sensors'].reshape(-1, 2).T)

    # get output vector
    y = get_output_vector(points, params['grid_width'], params['grid_height'], params['sigma'])

    # normalize data
    x = x / np.sqrt(np.square(params['grid_width']) + np.square(params['grid_height']))

    if transpose:
        x = x.T
        y = y.T
        points = points.T

    return x, y, points

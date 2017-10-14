from collections import OrderedDict
import numpy as np

experiments = [
    OrderedDict([
        ('id', 'sensors-1'),
        ('epoch', 10000),  # nr epochs
        ('nr_points', 1000),  # nr test examples
        ('sensors', np.array([0, 0, 10, 0])),  # sensor locations [s1_x, s1_y, s2_x, s2_y .. ]
        ('sigma', 1),  # sigma for gaussian noise on output vector
        ('grid_width', 10),  # grid width
        ('grid_height', 10),  # grid height
        ('nh_1', 12),  # nr units in first hidden layer
        ('loss_function', 'categorical_crossentropy'),  # loss function
        ('hidden_activation', 'tanh'),  # activation for hidden layers
        ('output_activation', 'softmax'),  # final activation
        ('optimizer', 'adam')  # optimizer
    ]),
    OrderedDict([
        ('id', 'sensors-2'),
        ('epoch', 10000),  # nr epochs
        ('nr_points', 1000),  # nr test examples
        ('sensors', np.array([0, 0, 10, 10])),  # sensor locations [s1_x, s1_y, s2_x, s2_y .. ]
        ('sigma', 1),  # sigma for gaussian noise on output vector
        ('grid_width', 10),  # grid width
        ('grid_height', 10),  # grid height
        ('nh_1', 12),  # nr units in first hidden layer
        ('loss_function', 'categorical_crossentropy'),  # loss function
        ('hidden_activation', 'tanh'),  # activation for hidden layers
        ('output_activation', 'softmax'),  # final activation
        ('optimizer', 'adam')  # optimizer
    ]),
    OrderedDict([
        ('id', 'sensors-3'),
        ('epoch', 10000),  # nr epochs
        ('nr_points', 1000),  # nr test examples
        ('sensors', np.array([0, 0, 5, 10])),  # sensor locations [s1_x, s1_y, s2_x, s2_y .. ]
        ('sigma', 1),  # sigma for gaussian noise on output vector
        ('grid_width', 10),  # grid width
        ('grid_height', 10),  # grid height
        ('nh_1', 12),  # nr units in first hidden layer
        ('loss_function', 'categorical_crossentropy'),  # loss function
        ('hidden_activation', 'tanh'),  # activation for hidden layers
        ('output_activation', 'softmax'),  # final activation
        ('optimizer', 'adam')  # optimizer
    ]),
    OrderedDict([
        ('id', 'sensors-4'),
        ('epoch', 10000),  # nr epochs
        ('nr_points', 1000),  # nr test examples
        ('sensors', np.array([10, 5, 5, 10])),  # sensor locations [s1_x, s1_y, s2_x, s2_y .. ]
        ('sigma', 1),  # sigma for gaussian noise on output vector
        ('grid_width', 10),  # grid width
        ('grid_height', 10),  # grid height
        ('nh_1', 12),  # nr units in first hidden layer
        ('loss_function', 'categorical_crossentropy'),  # loss function
        ('hidden_activation', 'tanh'),  # activation for hidden layers
        ('output_activation', 'softmax'),  # final activation
        ('optimizer', 'adam')  # optimizer
    ]),
    OrderedDict([
        ('id', 'optimizer-1'),
        ('epoch', 10000),  # nr epochs
        ('nr_points', 1000),  # nr test examples
        ('sensors', np.array([0, 0, 10, 10])),  # sensor locations [s1_x, s1_y, s2_x, s2_y .. ]
        ('sigma', 1),  # sigma for gaussian noise on output vector
        ('grid_width', 10),  # grid width
        ('grid_height', 10),  # grid height
        ('nh_1', 12),  # nr units in first hidden layer
        ('loss_function', 'categorical_crossentropy'),  # loss function
        ('hidden_activation', 'tanh'),  # activation for hidden layers
        ('output_activation', 'softmax'),  # final activation
        ('optimizer', 'sgd')  # optimizer
    ]),
    OrderedDict([
        ('id', 'optimizer-2'),
        ('epoch', 10000),  # nr epochs
        ('nr_points', 1000),  # nr test examples
        ('sensors', np.array([0, 0, 10, 10])),  # sensor locations [s1_x, s1_y, s2_x, s2_y .. ]
        ('sigma', 1),  # sigma for gaussian noise on output vector
        ('grid_width', 10),  # grid width
        ('grid_height', 10),  # grid height
        ('nh_1', 12),  # nr units in first hidden layer
        ('loss_function', 'categorical_crossentropy'),  # loss function
        ('hidden_activation', 'tanh'),  # activation for hidden layers
        ('output_activation', 'softmax'),  # final activation
        ('optimizer', 'adam')  # optimizer
    ]),
    OrderedDict([
        ('id', 'epoch-1'),
        ('epoch', 20000),  # nr epochs
        ('nr_points', 1000),  # nr test examples
        ('sensors', np.array([0, 0, 10, 10])),  # sensor locations [s1_x, s1_y, s2_x, s2_y .. ]
        ('sigma', 1),  # sigma for gaussian noise on output vector
        ('grid_width', 10),  # grid width
        ('grid_height', 10),  # grid height
        ('nh_1', 12),  # nr units in first hidden layer
        ('loss_function', 'categorical_crossentropy'),  # loss function
        ('hidden_activation', 'tanh'),  # activation for hidden layers
        ('output_activation', 'softmax'),  # final activation
        ('optimizer', 'adam')  # optimizer
    ]),
    OrderedDict([
        ('id', 'epoch-2'),
        ('epoch', 50000),  # nr epochs
        ('nr_points', 1000),  # nr test examples
        ('sensors', np.array([0, 0, 10, 10])),  # sensor locations [s1_x, s1_y, s2_x, s2_y .. ]
        ('sigma', 1),  # sigma for gaussian noise on output vector
        ('grid_width', 10),  # grid width
        ('grid_height', 10),  # grid height
        ('nh_1', 12),  # nr units in first hidden layer
        ('loss_function', 'categorical_crossentropy'),  # loss function
        ('hidden_activation', 'tanh'),  # activation for hidden layers
        ('output_activation', 'softmax'),  # final activation
        ('optimizer', 'adam')  # optimizer
    ]),
]

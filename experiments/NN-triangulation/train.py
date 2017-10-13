from utils.data_utils import prepare_dataset
from utils.experiment_utils import save_experiment
from keras.models import Sequential
from keras.layers import Dense, Activation
from collections import OrderedDict
import numpy as np

# for reproducibility
np.random.seed(1)

# hyper parameters
hyper_params = OrderedDict([
    ('id', 1),
    ('epoch', 2000),  # nr epochs
    ('nr_points', 1000),  # nr test examples
    ('sensors', np.array([0, 5, 10, 5])),  # sensor locations [s1_x, s1_y, s2_x, s2_y .. ]
    ('sigma', 1),  # sigma for gaussian noise on output vector
    ('grid_width', 10),  # grid width
    ('grid_height', 10),  # grid height
    ('nh_1', 12),  # nr units in first hidden layer
    ('loss_function', 'categorical_crossentropy'),  # loss function
    ('hidden_activation', 'tanh'),  # activation for hidden layers
    ('output_activation', 'sigmoid'),  # final activation
    ('optimizer', 'sgd')  # optimizer
])

# prepare training set
x_train, y_train, points_train = prepare_dataset(**hyper_params)

# input and output size
n_y = hyper_params['grid_width'] * hyper_params['grid_height']

# create model
model = Sequential()
model.add(Dense(hyper_params['nh_1'], input_dim=x_train.shape[1], activation=hyper_params['hidden_activation']))
model.add(Dense(n_y, activation=hyper_params['output_activation']))

# Compile model
model.compile(loss=hyper_params['loss_function'], optimizer=hyper_params['optimizer'], metrics=['accuracy'])

# Fit the model, use 10 batches for the training set
model.fit(x_train, y_train, epochs=hyper_params['epoch'], batch_size=int(hyper_params['nr_points']/10))

# save model, hyper parameters and log experiment
save_experiment(model, hyper_params)

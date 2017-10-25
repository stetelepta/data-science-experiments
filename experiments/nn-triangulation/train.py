from utils.data_utils import prepare_dataset
from utils.experiment_utils import save_experiment, setup_experiments, save_losses
from keras.models import Sequential
from keras.layers import Dense, Activation
from collections import OrderedDict

import numpy as np

# for reproducibility
# np.random.seed(1)

# base experiment
base_experiment = OrderedDict([
    ('id', 'exp'),
    ('epoch', 10000),  # nr epochs
    ('nr_points', 10000),  # nr test examples
    ('sensors', np.array([0, 0, 10, 0])),  # sensor locations [s1_x, s1_y, s2_x, s2_y .. ]
    ('sigma', 0),  # sigma for gaussian noise on output vector
    ('grid_width', 10),  # grid width
    ('grid_height', 10),  # grid height
    ('nh_1', 10),  # nr units in first hidden layer
    ('loss_function', 'categorical_crossentropy'),  # loss function
    ('hidden_activation', 'tanh'),  # activation for hidden layers
    ('output_activation', 'softmax'),  # final activation
    ('optimizer', 'adam')  # optimizer
])


# what experiments do we want to run
exp_settings = [
    # {'param': 'nh_1', 'values': np.arange(1, 17)},                       # tune nr hidden layers
    # {'param': 'optimizer', 'values': ['sgd', 'adam']},                   # tune optimizers
    # {'param': 'sigma', 'values': [0, 0.5, 1, 2]},                        # tune sigma
    # {'param': 'hidden_activation', 'values': ['tanh', 'relu', 'elu', 'softplus', 'softsign']},          # tune activation for hidden layer
    # {'param': 'output_activation', 'values': ['sigmoid', 'softmax']},    # tune activation for output layer
    # {'param': 'optimizer', 'values': ['adam']},
    # {'param': 'nr_points', 'values': [100, 500, 1000, 2500, 5000, 10000]},
    {'param': 'epoch', 'values': [1000, 2500, 5000, 10000, 25000, 50000]}
]

experiments = setup_experiments(exp_settings, base_experiment)

# dict that will store losess for each experiment
losses = {}

for experiment in experiments:
    # prepare training set
    x_train, y_train, points_train = prepare_dataset(**experiment)

    # input and output size
    n_y = experiment['grid_width'] * experiment['grid_height']

    # create model
    model = Sequential()
    model.add(Dense(experiment['nh_1'], input_dim=x_train.shape[1], activation=experiment['hidden_activation']))
    # model.add(Dense(experiment['nh_1'], input_dim=experiment['nh_1'], activation=experiment['hidden_activation']))
    model.add(Dense(n_y, activation=experiment['output_activation']))

    # Compile model
    model.compile(loss=experiment['loss_function'], optimizer=experiment['optimizer'], metrics=['accuracy'])

    # Fit the model, use 10 batches for the training set
    history_callback = model.fit(x_train, y_train, epochs=experiment['epoch'], batch_size=int(experiment['nr_points']/10))

    # get loss history
    loss_history = history_callback.history["loss"]

    if experiment['name'] not in losses.keys():
        losses[experiment['name']] = {}
    # save loss for last iteration for this experiment
    losses[experiment['name']][experiment['value']] = loss_history[-1]

    # save model, hyper parameters and log experiment
    save_experiment(model, experiment, loss_history)

# save losses for all experiments
save_losses(losses)
# print(losses)

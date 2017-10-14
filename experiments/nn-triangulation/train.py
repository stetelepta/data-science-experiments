from utils.data_utils import prepare_dataset
from utils.experiment_utils import save_experiment
from experiments import experiments
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# for reproducibility
np.random.seed(1)

for hyper_params in experiments:
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
    history_callback = model.fit(x_train, y_train, epochs=hyper_params['epoch'], batch_size=int(hyper_params['nr_points']/10))

    # get loss history
    loss_history = history_callback.history["loss"]

    # save model, hyper parameters and log experiment
    save_experiment(model, hyper_params, loss_history)

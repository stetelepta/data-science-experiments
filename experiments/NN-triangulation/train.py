from data_utils import prepare_dataset
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# for reproducibility
np.random.seed(1)


def save_model(**params):
    # use experiment attributes as filename
    filename = ""
    for key in sorted(params):
        if key == "sensors":
            print(tuple(params[key]))
            filename += "sensors%s" % ([(s[0], s[1]) for s in tuple(params[key].T)])
        else:
            filename += "%s%s" % (key, str(params[key]))

    # save model
    model.save('models/%s.h5' % filename)  # save as HDF5 file


# hyper parameters
params = {
    'epoch': 10,  # nr epochs
    'nr_points': 100,  # nr test examples
    'sensors': np.array([[5, 5], [10, 10]]).T,  # sensor locations
    'sigma': 1,  # sigma for gaussian noise on output vector
    'grid_width': 10,  # grid width
    'grid_height': 10  # grid height
}

# prepare training set
x_train, y_train, points_train = prepare_dataset(**params)

# input and output size
n_y = params['grid_width'] * params['grid_height']

# create model
model = Sequential()
model.add(Dense(12, input_dim=x_train.shape[1], activation='tanh'))
model.add(Dense(n_y, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, epochs=params['epoch'], batch_size=int(params['nr_points']/10))

# save model
save_model(**params)

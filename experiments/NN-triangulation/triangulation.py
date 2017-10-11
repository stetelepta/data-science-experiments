from data_utils import *
from plot_utils import *
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# for reproducibility
np.random.seed(5)

# hyper parameters
epochs = 10000  # nr epochs
m_train = 1000  # training examples
m_test = 3  # test examples
sigma = 1  # sigma for gaussian noise on output vector
sensors = np.array([[0, 0], [10, 0], [0, 10]]).T  # sensor locations

# 
train = False
predict = True

# size of the grid
grid_width = 10
grid_height = 10


plt.figure(figsize=(10, 5))

# prepare training set
x_train, y_train, points_train = prepare_dataset(nr_points=m_train, sensors=sensors, grid_width=grid_width, grid_height=grid_height, sigma=sigma)

# prepare test set
x_test, y_test, points_test = prepare_dataset(nr_points=m_test, sensors=sensors, grid_width=grid_width, grid_height=grid_height, sigma=sigma)

# normalize data
x_test = x_test / np.sqrt(np.square(grid_width) + np.square(grid_height))
x_train = x_train / np.sqrt(np.square(grid_width) + np.square(grid_height))

# transpose data, because prepare_dataset returns shape (m, nx), and keras expects (nx, m)
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

# input and output size
n_y = grid_width * grid_height

# use experiment attributes as filename
filename = ""
for key in sorted(experiment):
    filename += "%s%s" % (key, experiment[key])

if train:
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=x_train.shape[1], activation='tanh'))
    model.add(Dense(n_y, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=int(nr_points_train/10))

    # save model
    model.save('models/%s.h5' % filename)  # save as HDF5 file

    # evaluate the model
    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

if predict:
    print("=== predict ===")
    model = load_model('models/epoch10000height10m1000n_h12sensors[[5 5]]sigma1width10.h5')

    # do prediction
    classes = model.predict(x_test, batch_size=m)

    print("=== plot predictions ===")
    # plot predictions
    plot_predictions(classes.T, y_test.T, grid_width, grid_height)

    suptitle = ""
    for key in sorted(experiment):
        suptitle += "%s: %s\n" % (key, experiment[key])
    plt.suptitle(suptitle, fontsize=10, horizontalalignment="left", x=0.02, y=0.95)

    fig = plt.gcf()

    plt.show()
    # fig.savefig("plots/%s.png" % filename)

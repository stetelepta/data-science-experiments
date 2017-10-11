from data_utils import prepare_dataset
from plot_utils import plot_predictions
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# for reproducibility
np.random.seed(1)

params = {
    'nr_points': 1,  # nr test examples
    'sensors': np.array([[5, 5]]).T,  # sensor locations
    'sigma': 1,  # sigma for gaussian noise on output vector
    'grid_width': 10,  # grid width
    'grid_height': 10  # grid height
}

# prepare test set
x_test, y_test, points_test = prepare_dataset(**params)

# load model from file
model = load_model('models/epoch10000height10m1000n_h12sensors[[5 5]]sigma1width10.h5')

# do prediction
probs = model.predict(x_test, batch_size=params['nr_points'])

# plot predictions
plot_predictions(probs, y_test, **params)

# show plot
plt.show()

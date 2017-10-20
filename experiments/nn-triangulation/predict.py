from utils.data_utils import prepare_dataset
from utils.plot_utils import plot_predictions, plot_sensors_distances
from utils.experiment_utils import save_experiment, load_experiment
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# for reproducibility
np.random.seed(2)

# load experiment
experiment_id = "one-sensors-1"

# load model and parameters
model, hyper_params = load_experiment(experiment_id)

# nr of points to generate for test set
hyper_params['nr_points'] = 1

# prepare test set
x_test, y_test, points_test = prepare_dataset(**hyper_params)

# do prediction
probs = model.predict(x_test)

for i in range(0, probs.shape[0]):
    # plot predictions
    plot_predictions(probs[i, :], y_test[i, :], points_test[i, :], **hyper_params)

    # save plot
    plt.gcf().savefig("plots/exp-%s-%d-probs.png" % (experiment_id, i), bbox_inches='tight')

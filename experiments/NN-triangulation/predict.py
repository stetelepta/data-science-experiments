from utils.data_utils import prepare_dataset
from utils.plot_utils import plot_predictions
from utils.experiment_utils import save_experiment, load_experiment
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# for reproducibility
np.random.seed(1)

# load experiment
experiment_id = 1

# load model and parameters
model, hyper_params = load_experiment(experiment_id)

# nr of points to generate for test set
hyper_params['nr_points'] = 5

# prepare test set
x_test, y_test, points_test = prepare_dataset(**hyper_params)

# do prediction
probs = model.predict(x_test)

# plot predictions
plot_predictions(probs, y_test, **hyper_params)

# show plot
# plt.show()

# save plot
plt.gcf().savefig("plots/exp-%s.png" % experiment_id)

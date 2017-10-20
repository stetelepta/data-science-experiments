import pickle
import datetime
import numpy as np
from keras.models import load_model


def setup_experiments(experiment_settings, base_experiment):
    # initialize list with experiments
    experiments = []

    # loop trough all experiments
    for p in experiment_settings:

        # create new dict for each value in values
        for value in p['values']:
            # copy base experiment
            experiment = base_experiment.copy()

            # set experiment name
            experiment['name'] = p['param']

            # set experiment id
            experiment['value'] = value

            # set experiment id
            experiment['id'] = "%s-%s" % (p['param'], value)

            # set parameters
            experiment[p['param']] = value

            # append experiment to the list
            experiments.append(experiment)

    return experiments


def load_experiment(id, folder="models"):
    # save hyper parameters as pickle
    hyper_params = pickle.load(open('%s/exp-%s.pickle' % (folder, id), 'rb'))

    # load model from file
    model = load_model('%s/exp-%s.h5' % (folder, id))

    return model, hyper_params


def save_losses(losess, folder="models"):

    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # save loss history
    pickle.dump(losess, open('%s/exp-losses-%s.pickle' % (folder, time), 'wb'))


def save_experiment(model, hyper_params, loss_history, folder="models"):
    # save hyper parameters as pickle file
    # save model as .h5 file
    # append entry for this experiment to experiment.txt

    # for logging date and time
    now = datetime.datetime.now()

    # initialize output
    output = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    # add all hyper parameters to output
    for key in hyper_params:
        output += " %s: %s;" % (key, str(hyper_params[key]))

    # save hyper parameters as pickle
    pickle.dump(hyper_params, open('%s/exp-%s.pickle' % (folder, hyper_params['id']), 'wb'))

    # save loss history
    pickle.dump(loss_history, open('%s/exp-%s-loss_history.pickle' % (folder, hyper_params['id']), 'wb'))

    # add final loss to output
    output += "loss: %f" % loss_history[-1]

    # append hyper parameters to experiments.txt
    with open("%s/experiments.txt" % folder, "a") as f:
        f.write("%s\n" % output)

    # save model as HDF5 file
    model.save('%s/exp-%s.h5' % (folder, hyper_params['id']))

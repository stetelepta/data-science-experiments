import pickle
import datetime
from keras.models import load_model


def load_experiment(id, folder="models"):
    # save hyper parameters as pickle
    hyper_params = pickle.load(open('%s/exp-%s.pickle' % (folder, id), 'rb'))

    # load model from file
    model = load_model('%s/exp-%s.h5' % (folder, id))

    return model, hyper_params


def save_experiment(model, hyper_params, folder="models"):
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

    # append hyper parameters to experiments.txt
    with open("%s/experiments.txt" % folder, "a") as f:
        f.write("%s\n" % output)

    # save model as HDF5 file
    model.save('%s/exp-%s.h5' % (folder, hyper_params['id']))

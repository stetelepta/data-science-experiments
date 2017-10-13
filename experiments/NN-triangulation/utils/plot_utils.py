import matplotlib.pyplot as plt


def plot_predictions(probabilities, Y_test, **params):
    # plot predictions

    m = probabilities.shape[0]

    for i in range(0, m):
        # reshape each column to its 2D version
        probs_matrix = probabilities[i, :].reshape(params['grid_width'], params['grid_height'])

        # plot probabilities
        plt.subplot(m, 2, 2*i+1)
        if i == 0:
            plt.title("probabilities")
        plt.axis('off')

        plt.imshow(probs_matrix)

        # plot true values
        plt.subplot(m, 2, 2*i+2)
        if i == 0:
            plt.title("true value")
        plt.axis('off')
        plt.imshow(Y_test[i, :].reshape(params['grid_width'], params['grid_height']))

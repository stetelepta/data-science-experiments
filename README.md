# data-science-experiments
Exploring data science concepts and techniques


Experiment 1: Triangulation with Neural networks
-----------------------

Predict the location of objects, based on simulated distances by a number of sensors. 

### Simulated dataset

This experiment uses simulation to generate a number (_m_) of points. The coordinates of (_n_) sensors are given as input to the simulation. 
For each point the euclidean distance between the points and the sensors is calculated and used as training data (X).

A 2D grid is used as output (Y), where the true location of the object is approximated in a _bin_, so the training goal is to predict the correct bin for a point. To reward the network for predictions near the correct bin, gaussian noise is added to the output vector.

_Dimensions_
* m: number of points
* n: number of sensors
* w: width of the 2D plane
* h: height of the 2D plane
* X: (m, n) matrix - for each point the euclidean distance between the points and the sensors is calculated and used as training data
* Y: (m, w x h) matrix - The output vector, is a 

### Installation
 
* Install the requirements using `pip install -r requirements.txt`.
  * Make sure you use Python 3.

Usage
-----------------------

* Adjust hyperparameters or add new experiments in `experiments.py`
* Run `python train` to run all experiments from `experiments.py`
   * model is saved as HD5 file: `models/exp-[experiment].hd5`
   * loss history is saved as pickle: `models/exp-[experiment]-loss_history.pickle`
   * hyper parameters are saved as pickle: `models/exp-[experiment].pickle`
   * an entry is with hyper parameters and final loss is added to `models/experiments.txt`
* Adjust `predict.py` to run a prediction for an experiment.
* Run `python predict.py`
   * plots with predicts results are saved as png: `plots/exp-[experiment-i]-probs.png` 
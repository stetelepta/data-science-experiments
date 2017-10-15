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
* w: width of the 2D grid
* h: height of the 2D grid
* X: (_m_, _n_) matrix
* Y: (m, _w_ x _h_) matrix

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
   
Results
-----------------------

### One sensor with a 10x10 grid
When using one sensor, the probabilities spread in a circle around the sensor.
* Sensor location top-left, cost after 10.000 epochs for 1.000 training examples: `3.514507`
![One sensor](https://github.com/stetelepta/data-science-experiments/blob/master/experiments/nn-triangulation/plots/exp-one-sensors-1-9-probs.png?raw=true)

* Sensor location center, cost after 10.000 epochs for 1.000 training examples: `4.108721`
![One sensor](https://github.com/stetelepta/data-science-experiments/blob/master/experiments/nn-triangulation/plots/exp-one-sensors-2-8-probs.png?raw=true)

### Two sensors with a 10x10 grid
When using two sensors, the probabilies are highest on the intersection of the two circles around the sensors. Note that there are still two possible locations where the distances match.
* Cost after 10.000 epochs for 1.000 training examples: `3.086349`

![Two sensors](https://github.com/stetelepta/data-science-experiments/blob/master/experiments/nn-triangulation/plots/exp-sensors-2-9-probs.png?raw=true)

### Three sensors with a 10x10 grid
When using three sensors, there is just one possible location left where the object can be located. This location is correctly predicted.
* Cost after 10.000 epochs for 1.000 training examples: `2.589003`

![Three sensors](https://raw.githubusercontent.com/stetelepta/data-science-experiments/master/experiments/nn-triangulation/plots/exp-three-sensors-1-2-probs.png)

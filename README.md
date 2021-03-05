## Machine Learning: Decision Trees


### Data

The ``data/`` directory contains all the datasets.

The primary datasets are:
- ``iris.dat`` (for part 1)
- ``housing.csv`` (for part 2)


### Codes

- ``part1_nn_lib.py``

	* Contains  ``Layer`` abstract class
    * Contain ``MSELossLayer``, ``CrossEntropyLossLayer`` class (loss functions) 
    * Contains ``SigmoidLayer``, ``ReluLayer``, ``LinearLayer`` class (activation functions) 
    * Contains  ``MultiLayerNetwork``, ``Preprocessor``, ``Trainer`` class
    * Builds a neural network from scratch (w/out library)
    

- ``part2_house_value_regression.py``

    * Contains ``Net``, ``Regressor`` class
    * Builds a neural network with three hidden layers using ``Pytorch``
    * Contains function for evaluation and hyper-parameters tunning
    * Contains function for plotting loss vs. epoch

### Instructions

Simply run the ``part1_nn_lib.py`` or ``part2_house_value_regression.py``.





import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self, input_size, output_size, n_1, n_2, n_3, dropout_p):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, n_1)
        self.fc2 = nn.Linear(n_1, n_2)
        self.fc3 = nn.Linear(n_2, n_3)
        self.fc4 = nn.Linear(n_3, output_size)
        self.dropout = nn.Dropout(dropout_p, inplace=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.fc4(x)

        return x


class Regressor(BaseEstimator, ClassifierMixin):

    def __init__(self, x, nb_epoch=100, batch_size=32, learning_rate=0.001, n_1=32, n_2=16, n_3=8, dropout_p=0):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Parameters for re-applying in _preprocessing()
        self.one_hot = preprocessing.LabelBinarizer()
        self.xScaler = preprocessing.MinMaxScaler()
        self.yScaler = preprocessing.MinMaxScaler()

        # initialising parameters
        self.x = x
        if x is not None:
            X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.losses = []
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        self.dropout_p = dropout_p

        self.net = Net(self.input_size, self.output_size, self.n_1, self.n_2, self.n_3, self.dropout_p)
        self.optimizer = None
        self.criterion = nn.MSELoss()

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        textual_attr = "ocean_proximity"

        # handle textual values
        if training:
            self.one_hot.fit(x[textual_attr])
        lb_one_hot_values = self.one_hot.transform(x[textual_attr])

        # handle NaN values
        x_numerical = x.loc[:, x.columns != textual_attr]
        values = {i: x_numerical[i].mean(axis=0) for i in x_numerical.keys()}
        x = x.fillna(value=values)

        # remove the textual data
        x = x.drop(textual_attr, axis=1)

        # Normalisation on X
        if training:
            self.xScaler.fit(x)
        x = self.xScaler.transform(x)

        # Normalisation on y
        if y is not None:
            if training:
                self.yScaler.fit(y)
            y = self.yScaler.transform(y)

        # apply one-hot-encoding
        x = np.concatenate((x, lb_one_hot_values), axis=1)

        if y is not None:
            y = torch.tensor(y).float()
        x = torch.tensor(x).float()

        # Return preprocessed x and y, return None for y if it was None
        return x, (y if isinstance(y, torch.Tensor) else None)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget

        self.net.train()  # set training mode
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.losses = []  # reset everytime it trains
        dataset = torch.utils.data.TensorDataset(X, Y)

        for epoch in range(1, self.nb_epoch + 1):

            X, Y = sklearn.utils.shuffle(X, Y)
            train_set = torch.utils.data.DataLoader(dataset,
                                                    batch_size=self.batch_size, shuffle=True)

            total_loss_per_batch = 0

            for input, target in train_set:
                self.optimizer.zero_grad()

                output = self.net(input)

                loss = self.criterion(output, target)

                loss.backward()

                self.optimizer.step()

                total_loss_per_batch += loss.item()  # get python number from tensor

            print(f"loss(epoch {epoch}) {total_loss_per_batch}")
            self.losses += total_loss_per_batch,

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, _ = self._preprocessor(x, training=False)
        self.net.eval()  # set evaluation mode

        with torch.no_grad():
            y_pred = self.net(X)
        denormalized_y_pred = self.yScaler.inverse_transform(y_pred)
        return denormalized_y_pred

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y_true):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # X, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
        y_pred = self.predict(x)
        mse = mean_squared_error(y_true, y_pred)
        return np.math.sqrt(mse)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def plotLoss(self):  # Model's Loss vs. Epoch
        plt.plot(list(range(1, self.nb_epoch + 1)), self.losses)
        plt.title("Model's Loss vs. Epoch")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()


def save_regressor(trained_model):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x, y):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    parameters = [{'nb_epoch': range(100, 1000),
                   'batch_size': [16, 32, 64, 128],
                   'learning_rate': [0.002, 0.001, 0.0005],
                   'n_1': range(7, 80),
                   'n_2': range(7, 50),
                   'n_3': range(7, 30),
                   'dropout_p': [0, 0.1, 0.15]
                   }]

    # setting up the search
    search = RandomizedSearchCV(
        Regressor(x),
        param_distributions=parameters,
        cv=2,
        n_iter=1,
        scoring="neg_mean_squared_error",
    )

    search.fit(x, y)

    return search.best_estimator_, search.best_params_

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    X = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    x_train, x_val, y_train, y_val = \
        train_test_split(X, y, test_size=0.2)

    # select the best models and parameters and then train the model
    best_regressor, best_hyper_params = RegressorHyperParameterSearch(x_train, y_train)
    print('Best combinations of hyper-parameters: \n{}:\n'.format(best_hyper_params))
    regressor = best_regressor
    regressor.fit(x_train, y_train)
    # save_regressor(regressor)

    # regressor.plotLoss()  # plot graph of loss vs. epoch

    # Regressor error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error on training set: {}\n".format(error))

    error = regressor.score(x_val, y_val)
    print("\nRegressor error on validation set: {}\n".format(error))


if __name__ == "__main__":
    example_main()

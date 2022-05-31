import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
        
#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

import seaborn as sns #sets up styles and gives us more plotting options


class batch_gradient_descent:
    """
    Linear Regression using Batch Gradient Descent.
    Parameters
    -----------
    learn_rate : float
        Learning Rate. 
        Default: 0.0001
    
    iterations : int
        Epochs / number of passes on the training set.
        Default: 1000
    """

    def __init__(self, learn_rate = 0.0001, iterations = 1000):
        self.learn_rate = learn_rate
        self.iterations = iterations

    def fit(self, inputs, outputs):
        """Fit the training data
        Parameters
        ----------
        inputs : numpy-array, shape = [number of samples, number of features]
        outputs : numpy-array, shape = [number of samples, number of target values]
        
        Returns
        -------
        self: object
        """
        thetas = np.ones((2,1))

        for _ in range(self.iterations):
            y_predictions = np.dot(inputs, thetas)
            residual_values = outputs - y_predictions
            gradient_vector = np.dot(inputs.T, residual_values)
            thetas += self.learn_rate/outputs.shape[0]*gradient_vector

        self.thetas = thetas
        return self

    def predict(self, inputs):
        """
        Predict values based on theta.
        Parameters
        ----------
        inputs: numpy-array containing values from which to predict.

        Returns
        -------
        Predicted value(s)
        """

        return np.dot(inputs, self.thetas)
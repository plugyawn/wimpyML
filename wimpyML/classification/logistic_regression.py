from operator import index
import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd
from torch import dsmm, sigmoid #lets us handle data as dataframes
        
#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

import seaborn as sns #sets up styles and gives us more plotting options

from wimpyML.tools.helper import common_functions

class binary_classification:
    """Logistic Regression for binary classification.
    Parameters
    ----------
    learn_rate : float
    Learning rate, checks how much peturbation is tolerated when training model.
    Too low will take more iterations to estimate weights, too high will cause
    too much peturbation for a good estimation.
    Default : 0.01

    iterations : int
    Number of iterations passes to go through the system.
    Default : 10e4
    """


    def __init__(self, learn_rate = 0.01, iterations = 10e4):
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
        # try: outputs.shape[1]
        # except: outputs = outputs.reshape((outputs.shape[0],1))
        inputs = np.concatenate((np.ones((inputs.shape[0],1)),inputs), axis = 1)

        try: self.thetas = self.thetas 
        except: self.thetas = np.zeros((inputs.shape[1], 1))

        for _ in range(int(self.iterations)):
            
            output_predictions = common_functions.sigmoid(np.dot(inputs, self.thetas))
            residual_values = outputs - output_predictions
            gradient_vector = np.dot(inputs.T, residual_values)
            self.thetas += self.learn_rate/inputs.shape[0]*gradient_vector
            print(_)
        return self
        
    def classify(self, inputs):
        """Classify based on current hypothesis.
        Parameters
        ----------
        inputs : numpy-array
        Pass the inputs on the basis of which outputs are desired.
        
        Output
        ------
        outputs : numpy-array
        Predictions based on the hypothesis of the class object.
        """
        return np.round(common_functions.sigmoid(np.dot(inputs, self.thetas)))
from operator import index
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
        try: inputs.insert(1, "C", np.ones(inputs.shape[0]))
        except: pass
        try:
            thetas = self.thetas
        except:
            thetas = np.ones((inputs.shape[1], 1))

        for _ in range(self.iterations):
            output_predictions = np.dot(inputs, thetas)
            residual_values = outputs - output_predictions
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
        outputs : numpy-array
        Predicted value(s)
        """

        return np.dot(inputs, self.thetas)

class stochastic_gradient_descent:
    """
    Linear Regression with Stochastic Gradient Descent.
    Parameters
    ----------
    learn_rate : float
        Learning Rate
        Default: 0.0001
        
    iterations: int
        Number of passes to go through the system
        Default: 1000
    """

    def __init__(self, learn_rate = 0.0001, iterations=10000):
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
        inputs.insert(1, "C", np.ones(inputs.shape[0]))
        thetas = np.ones((2,1))
        batch_iterations = self.iterations//outputs.shape[0]

        batch_grad = batch_gradient_descent(learn_rate=self.learn_rate ,iterations=batch_iterations)
        batch_grad.fit(inputs=inputs, outputs=outputs)

        self.iterations -= batch_iterations*outputs.shape[0]
        inputs = inputs.iloc[0:self.iterations,]
        outputs = outputs.iloc[0:self.iterations,]

        batch_grad.iterations = 1
        batch_grad.fit(inputs=inputs, outputs=outputs)

        self.thetas = batch_grad.thetas
        return self

    def predict(self, inputs):
        """
        Predict next value based on current hypothesis.
        Parameters
        ----------
        inputs : numpy-array
        Pass the inputs on the basis of which outputs are desired.
        
        Output
        ------
        outputs : numpy-array
        Predictions based on the hypothesis of the class object.
        """

        return np.dot(inputs, self.thetas)

class bayesian_linear_regression:
    """
    Bayesian Linear Regression as opposed to Frequentist Linear Regression.
    """
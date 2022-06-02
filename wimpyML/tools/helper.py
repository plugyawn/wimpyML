from operator import index
from tokenize import String
from typing import Any
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

class common_functions:
    """
    Common functions that get used often. Includes sigmoids, loss functions, etc.
    """

    def sigmoid(parameter: float) -> float:
        return 1 / (1 + np.exp(-parameter))

    def mean_square_error(sample, predictions):
        """Mean square error, as used prominently in Linear Regression.
        Parameters
        ----------
        sample : array-like, shape = [sample size, number of features]
        predictions : array-like, shape = [sample size, number of features]
        
        Output
        ------
        error : float, the mean-square error of predictions wrt the sample."""
        return 1/(sample.shape[0])*(np.sum(sample-predictions))**2


class dataset_access:
    """
    Datasets included in the library to test various kinds of regressions.
    """

    def dataset_load(dataset_name: String) -> Any:
        """
        Load datasets based on given parameters.
        Parameters
        ----------
        dataset_name : String, name of the dataset that we are accessing.
        
        Available Datasets
        ------------------
        "simple_linear_regression" : simple linear regression dataset from towardsdatascience.com
        "simple_binary_classification" : simple logistic regression dataset sampled from the iris dataset
        """
        if dataset_name == "simple_linear_regression":

            dataset = pd.read_csv("./datasets/simple_linear_regression.csv")

            inputs = dataset.copy()
            inputs = inputs.drop("Y", axis = 1)

            outputs = dataset.copy().drop(["X"], axis = 1)

            return (inputs, outputs)

        if dataset_name == "simple_binary_classification":

            dataset = pd.read_csv("./datasets/simple_binary_classification.csv").drop(["REM"], axis = 1)
            inputs = dataset.copy()
            inputs = dataset.drop("Y", axis = 1)

            outputs = dataset.copy().drop(["X1", "X2"], axis = 1)
            
            return (inputs, outputs)


            

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
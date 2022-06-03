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

class plotter:
    """Functions for convenient plotting."""

    def plot_func(self, func, x_end = 1, x_start = 0, precision = 100):
        """Basic function plotter based on seaborn.
        Parameters
        ----------
        func : define function and pass as argument.

        x_start : origin for the X axis.
        default = 0

        x_end : end-point for X axis on graph.
        default = 0

        precision : number of samples to graph over.
        default = 100
        """
        try:
            x = np.linspace(x_start, x_end, precision)
            y = func(x)
            sns.lineplot(x = x, y = y)
            plt.show()
        except:
            print("Value Error: Enter function that takes in one variable and returns another.")
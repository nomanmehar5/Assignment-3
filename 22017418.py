# -*- coding: utf-8 -*-
"""
This code examines the total population and there impact on CO2 Emission.
Using Kmeans clustering for categorization of total population and 
employing curve fitting models for projecting CO2 Emission data.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import scipy.optimize as opt
import itertools as iter
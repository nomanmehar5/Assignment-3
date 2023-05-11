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

#Data files
Total_population = 'API_SP.POP.TOTL_DS2_en_csv_v2_5436324.csv'
CO2Emissions = "API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5358914.csv"

def read(filename):
    """ 
    this function reads an CSV file and returns the original dataframe 
    and its transposed version
    """
        
    data = pd.read_csv(filename, skiprows=4)
    data.drop(data.columns[[1, 2, 3, 66]], axis=1, inplace=True)
    return data, data.transpose()

"""
Reading a Total Population file where O_data is the original format and 
T_data is the transposed format

"""

O_data, T_data = read(Total_population)

#Create a year1 and year2 variable for specific countries
year1 = '1991'

year2 = '2021'

#Get the required data for the clustering and drop missing values
Odata_list = O_data.loc[O_data.index,['Country Name', year1, year2]].dropna()

#Plot the data
plt.figure()

Odata_list.plot(year1, year2, kind='scatter', color='blue', label='Population')

plt.title('Total Population', fontsize=16)
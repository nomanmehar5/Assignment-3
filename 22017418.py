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

#Convert the dataframe to an array and store into Od_arr
Od_arr = Odata_list[[year1, year2]].values

#print(Od_arr)

#Using Elbow Method to get the best number of clusters
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(Odata_list[['1991','2021']])
    sse.append(km.inertia_)
    
print(sse)

#Plot the Elbow plot
plt.figure()
plt.plot(k_rng, sse)
plt.xlabel('K', fontweight='bold', fontsize=14)
plt.ylabel('SSE', fontweight='bold', fontsize=14)
plt.title('Elbow Method to get required cluster value', fontweight='bold',
          fontsize=14)
plt.show()

#Normalizing the data with MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(Odata_list[['1991']])
Odata_list['1991'] = scaler.transform(Odata_list[['1991']])
scaler.fit(Odata_list[['2021']])
Odata_list['2021'] = scaler.transform(Odata_list[['2021']])

#Getting the values of cluster
km = KMeans(n_clusters=3)
y_pred = km.fit_predict(Odata_list[['1991','2021']])
print(y_pred)

#add a new column Cluster in original dataframe
Odata_list['cluster'] = y_pred

print(Odata_list)

#Finding the cluster center
cluster_cent = km.cluster_centers_
print(cluster_cent)

#Creating new dataframes for all three clusters
df0 = Odata_list[Odata_list.cluster == 0]
df1 = Odata_list[Odata_list.cluster == 1]
df2 = Odata_list[Odata_list.cluster == 2]

#Plot the cluster plot with marker as a star at the center of every cluster
plt.figure()
plt.scatter(df0['1991'], df0['2021'], color='green', label='cluster 0')
plt.scatter(df1['1991'], df1['2021'], color='blue', label='cluster 1')
plt.scatter(df2['1991'], df2['2021'], color='red', label='cluster 2')
plt.scatter(cluster_cent[:,0], cluster_cent[:,1],
            color='purple', marker='*', s=250, label='centroid')
plt.xlabel('1991', fontweight='bold', fontsize=14)
plt.ylabel('2021', fontweight='bold', fontsize=14)
plt.legend(fontsize=14)
plt.title('Total Population', fontweight='bold', fontsize=14)
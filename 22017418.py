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
plt.figure(figsize=(10, 6))

Odata_list.plot(year1, year2, kind='scatter', color='blue', label='Population')

plt.legend(fontsize=20)

plt.title('Total Population', fontsize=20)

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
plt.figure(figsize=(10, 6))
plt.plot(k_rng, sse)
plt.xlabel('K', fontweight='bold', fontsize=20)
plt.ylabel('SSE', fontweight='bold', fontsize=20)
plt.title('Elbow Method to get required cluster value', fontweight='bold',
          fontsize=20)
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

#Add a new column Cluster in original dataframe
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
plt.figure(figsize=(10, 6))
plt.scatter(df0['1991'], df0['2021'], color='green', label='cluster 0')
plt.scatter(df1['1991'], df1['2021'], color='blue', label='cluster 1')
plt.scatter(df2['1991'], df2['2021'], color='red', label='cluster 2')
plt.scatter(cluster_cent[:,0], cluster_cent[:,1],
            color='purple', marker='*', s=250, label='centroid')
plt.xlabel('1991', fontweight='bold', fontsize=20)
plt.ylabel('2021', fontweight='bold', fontsize=20)
plt.legend(fontsize=20)
plt.title('Total Population', fontweight='bold', fontsize=20)

"""

Curve Fitting Solutions

"""

#Reading the Total population file from the world bank format
O_data, T_data = read(Total_population)

print(T_data)

#Rename the transposed data columns for population
df2 = T_data.rename(columns=T_data.iloc[0])

#Drop the country name
df2 = df2.drop(index=df2.index[0], axis=0)
df2['Year'] = df2.index

print(df2)

#Fitting for China's population data
df_fit = df2[['Year', 'China']].apply(pd.to_numeric, errors='coerce')

#Logistic function for curve fitting and forecasting the Total Population
def logistic(t, n0, g, t0):
    """ 
    Calculates the logistic growth of a population.
    
    Parameters:
        t = The current time.
        n0 = The initial population.
        g = The growth rate.
        t0 = The inflection point.
        
    """
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

#Fits the logistic data
param_ch, covar_ch = opt.curve_fit(logistic, df_fit['Year'], df_fit['China'],
                                   p0=(3e12, 0.03, 2041))

""""""
#Error ranges calculation
def err_ranges(x, func, param, sigma):
    """
    Calculates the error ranges for a given function and its parameters.
    
    Parameters:
        x = The input value for the function.
        func = The function for which the error ranges will be calculated.
        param = The parameters for the function.
        sigma = The standard deviation of the data.
        
    """
    
    #Initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    #Create a list of tuples of upper and lower limits for parameters
    uplow = []
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    #Calculate the upper and lower limits
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper

#Calculating the standard deviation
sigma_ch = np.sqrt(np.diag(covar_ch))

#Creating a new column with the fit data
df_fit['fit'] = logistic(df_fit['Year'], *param_ch)

#Forecast for the next 20 years
year = np.arange(1960, 2041)

forecast = logistic(year, *param_ch)

#Calculates the error ranges
low_ch, up_ch = err_ranges(year, logistic, param_ch, sigma_ch)

#Plotting China's Total Population with Forecast and Confidence range
plt.figure(dpi=600)
plt.plot(df_fit["Year"], df_fit["China"], label="Population", c='purple')
plt.plot(year, forecast, label="Forecast", c='red')
plt.fill_between(year, low_ch, up_ch, color="orange", alpha=0.7, 
                 label='Confidence Range')
plt.xlabel("Year", fontweight='bold', fontsize=14)
plt.ylabel("Population",fontweight='bold', fontsize=14)
plt.legend(fontsize=14)
plt.title('China', fontweight='bold', fontsize=14)
plt.show()

#Prints the error ranges
print(err_ranges(2041, logistic, param_ch, sigma_ch))

#Fitting the United State's Population data
us = df2[['Year', 'United States']].apply(pd.to_numeric, errors='coerce')

#Fits the US logistic data
param_us, covar_us = opt.curve_fit(logistic, us['Year'], us['United States'], 
                                   p0=(3e12, 0.03, 2041))

#Calculates the standard deviation for United States data
sigma_us = np.sqrt(np.diag(covar_us))

#Forecast for the next 20 years
forecast_us = logistic(year, *param_us)

#Calculate error ranges
low_us, up_us = err_ranges(year, logistic, param_us, sigma_us)

#Plotting United State's Total Population with Forecast and Confidence range
plt.style.use('seaborn')
plt.figure(dpi=600)
plt.plot(us["Year"], us["United States"],
         label="Population")
plt.plot(year, forecast_us, label="Forecast", c='red')
plt.fill_between(year, low_us, up_us, color="orange", alpha=0.7, 
                 label="Confidence Range")
plt.xlabel("Year", fontweight='bold', fontsize=14)
plt.ylabel("Population", fontweight='bold', fontsize=14)
plt.legend(loc='upper left', fontsize=14)
plt.title('US Population', fontweight='bold', fontsize=14)
plt.show()

#Reading the CO2 Emissions file from the world bank format
CO2, CO2T = read(CO2Emissions)

#Drop the specific columns with null values
CO2 = CO2.drop(CO2.loc[:, '1960':'1989'].columns, axis=1)
CO2 = CO2.drop(['2020', '2021'], axis=1)

print(CO2)

#Transpose the data
CO2T = CO2.transpose()

print(CO2T)

#Rename the columns
g = CO2T.rename(columns=CO2T.iloc[0])

#Drop the country name
g = g.drop(index=g.index[0], axis=0)

g['Year'] = g.index

print(g)

#Fitting the data
g_chi = g[['Year', 'China']].apply(pd.to_numeric, errors='coerce')

print(g_chi)

# poly function for forecasting CO2 Emission
def poly(x, a, b, c):
    
    """ 
    Calculates the value of a polynomial function of the form ax^2 + bx + c.
    
    Parameters:
        x = The input value for the polynomial function.
        a = The coefficient of x^2 in the polynomial.
        b = The coefficient of x in the polynomial.
        c = The constant term in the polynomial.
      
    """
    return a*x**2 + b*x + c

def get_error_estimates(x, y, degree):
    
    """
    Calculates the error estimates of a polynomial function.
   
    Parameters:
       x = The x-values of the data points.
       y = The y-values of the data points.
       degree = The degree of the polynomial.
       
    """
      
    coefficients = np.polyfit(x, y, degree)
    y_estimate = np.polyval(coefficients, x)
    residuals = y - y_estimate
 
    return np.std(residuals)

#Fits the linear data
param_cg, cov_cg = opt.curve_fit(poly, g_chi['Year'], g_chi['China'] )

#Calculates the standard deviation
sigma_cg = np.sqrt(np.diag(cov_cg))

#Creates a new column for the fit figures
g_chi['fit'] = poly(g_chi['Year'], *param_cg)

year = np.arange(1990, 2041)

#Forecasting the fit figures
forecast_cg = poly(year, *param_cg)

#Error estimates
error_cg = get_error_estimates(g_chi['China'], g_chi['fit'], 2)
print('\n Error Estimates for China CO2 Emissions:\n', error_cg)

#Plotting the China CO2 Emissions data with Forecast
plt.figure()
plt.plot(g_chi["Year"], g_chi["China"], label="CO2 Emissions", c='purple')
plt.plot(year, forecast_cg, label="Forecast", c='red')
plt.xlabel("Year", fontweight='bold',fontsize=14)
plt.ylabel("CO2 Emissions", fontweight='bold', fontsize=14)
plt.legend(fontsize=14)
plt.title('China', fontweight='bold', fontsize=14)
plt.show()


#Fitting the data
g_us = g[['Year', 'United States']].apply(pd.to_numeric, errors='coerce')

#Fits the linear data
param_usg, cov_usg = opt.curve_fit(poly, g_us['Year'], g_us['United States'])

#Calculates the standard deviation
sigma_usg = np.sqrt(np.diag(cov_usg))

#Creates a column for the fit data
g_us['fit'] = poly(g_us['Year'], *param_usg)

#Forecasting for the next 20 years
forecast_usg = poly(year, *param_usg)

#Error estimates
error_us = get_error_estimates(g_us['United States'], g_us['fit'], 2)

print('\n Error Estimates for US CO2 Emissions:\n', error_us)

#Plotting the United States CO2 Emissions data with Forecast
plt.figure()
plt.plot(g_us["Year"], g_us["United States"], label="CO2 Emissions")
plt.plot(year, forecast_usg, label="Forecast", c='red')
plt.xlabel("Year", fontweight='bold', fontsize=14)
plt.ylabel("CO2 Emissions", fontweight='bold', fontsize=14)
plt.legend(fontsize=14)
plt.title('United States', fontweight='bold', fontsize=14)
plt.show()
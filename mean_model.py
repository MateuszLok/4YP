__author__ = 'Mat'
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import GP_functions_2 as gp
from pyDOE import lhs
from math import fabs
from math import log

# ----------------------------------------------------------------------------
#Data input
sample_size =1260
volatility_observed = gp.get_volatility(sample_size)
time_vector = gp.get_time_vector(volatility_observed)
print time_vector
print volatility_observed
new_x = np.array([0.2])

average=sum(volatility_observed)/len(volatility_observed)
print average
for index in range(0,len(volatility_observed)):
    volatility_observed[index]-=average

print volatility_observed

forecast_period=252
forecast_volatility = gp.get_volatility(sample_size+forecast_period)
for index in range(0,len(forecast_volatility)):
    forecast_volatility[index]-=average

print forecast_volatility[1260:]

sum_errors = [0,0,0,0]

x=sample_size+forecast_period

H_periods=[1,5,21,252]
for index1 in range(len(H_periods)):
    for index in range(sample_size,sample_size+H_periods[index1]):
        sum_errors[index1]+=fabs(forecast_volatility[index])**2

print sum_errors
error=[0,0,0,0]
for index1 in range(len(H_periods)):
    error[index1] = (sum_errors[index1]/H_periods[index1])**0.5

print error
__author__ = 'Mat'
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import GP_functions_2 as gp
from pyDOE import lhs
from math import fabs
from math import log


sample_size =60
jitter=0
monthly_volatility_observed = gp.get_monthly_data(sample_size,'average')
time_vector = gp.get_time_vector(monthly_volatility_observed)
print time_vector
print monthly_volatility_observed
new_x = np.array([0.2])

average=sum(monthly_volatility_observed)/len(monthly_volatility_observed)
print average
for index in range(0,len(monthly_volatility_observed)):
    monthly_volatility_observed[index]-=average

print monthly_volatility_observed
forecast_period=12
forecast_volatility = gp.get_monthly_data(sample_size+forecast_period,'average')

for index in range(0,len(forecast_volatility)):
    forecast_volatility[index]-=average


sum_errors = [0,0,0,0]

x=sample_size+forecast_period

H_periods=[1,3,6,12]
for index1 in range(len(H_periods)):
    for index in range(sample_size,sample_size+H_periods[index1]):
        print forecast_volatility[index]
        sum_errors[index1]+=fabs(forecast_volatility[index])**2

print sum_errors
error=[0,0,0,0]
for index1 in range(len(H_periods)):
    error[index1] = (sum_errors[index1]/H_periods[index1])**0.5

print error

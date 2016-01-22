__author__ = 'Mat'
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import GP_functions_sparse as gps
from pyDOE import lhs
from math import fabs

# ----------------------------------------------------------------------------
#Data input
M=50
sample_size=500


#Latin Hypercube Initlialziation
print 'Starting...'
latin_hypercube_values = lhs(M+3, samples=2)
latin_hypercube_values=latin_hypercube_values*M

#Optimization part
result=np.zeros((len(latin_hypercube_values),M+3))
for number in range(0,len(latin_hypercube_values)):
    print number
    wynik= optimize.minimize(gps.function_to_minimize, latin_hypercube_values[number], args=(M,sample_size), method='BFGS')
    result[number]=wynik['x']

likelihood=np.zeros((len(latin_hypercube_values),1))
for number in range(0,len(latin_hypercube_values)):
    likelihood[number] = gps.function_to_minimize(result[number],M,sample_size)
min_index = np.argmin(likelihood)
print likelihood
print min_index
print result[min_index]
hyperparameters = result[min_index]




#Finding values of K matrices for new values of x
K = gp.find_K(time_vector,hyperparameters,'normal')
K_2stars_estimate = gp.find_K_2stars(new_x,hyperparameters,'normal')
K_inv = np.linalg.inv(K)

#--------------------------------

#Vector of new x values
new_values = np.arange(0,900,0.20001)

#Initialise matrices to store estimated values of y(volatility) and variance
estimated_values_y = []
estimated_variance_y = []

#Initialise variable to store size of the variance and y vector
estimated_variance_size = 0

#Find y and variance for all 'new values'
for number in new_values:
    K_star_estimate = gp.find_K_star(time_vector,number,hyperparameters,'normal')
    X_estimate = np.dot(K_star_estimate,K_inv)
    estimated_values_y.append((np.dot(X_estimate,volatility_observed).tolist()))
    K_star_trans_estimate = K_star_estimate.transpose()
    temp = (K_2stars_estimate-np.dot(K_star_estimate,np.dot(K_inv,K_star_trans_estimate)))
    #To list to get rid of matrix representation
    estimated_variance_y.append(temp.tolist())
    estimated_variance_y[estimated_variance_size]=1.96*(fabs(temp)**0.5)
    estimated_variance_size+=1


new_estimated_values_y = []
for number in range(0,len(estimated_values_y)):
    new_estimated_values_y.append(estimated_values_y[number][0])

new_estimated_variance_y = estimated_variance_y

new_estimated_variance_y_1 = []
new_estimated_variance_y_2 = []
for number in range(0,len(new_estimated_variance_y)):
    new_estimated_variance_y_1.append(new_estimated_values_y[number]+new_estimated_variance_y[number])
    new_estimated_variance_y_2.append(new_estimated_values_y[number]-new_estimated_variance_y[number])





#Plotting one new value
plt.fill_between(new_values, new_estimated_variance_y_2,new_estimated_variance_y_1,alpha=0.5)
plt.plot(new_values,new_estimated_values_y, 'g-')
plt.plot(time_vector, volatility_observed, 'r.', markersize = 5)
#plt.plot(gp.get_time_vector(gp.get_volatility('all')), gp.get_volatility('all'), 'r.', markersize = 5)
#plt.axis([min(sample_vector)-0.5, max(sample_vector)+0.5, min(sample_y)-0.5, max(sample_y)+0.5])
plt.axis([0,len(time_vector)+50,-0.5,0.5])


#Error calculation
forecast_period = 200
forecast_volatility = gp.get_volatility(sample_size+forecast_period)
print len(forecast_volatility)

forecast_values = np.arange(sample_size,sample_size+forecast_period,1)

#Initialise matrices to store estimated values of y(volatility) and variance
estimated_values_y_forecast = []

#Find y and variance for all 'new values'
for number in forecast_values:
    K_star_estimate = gp.find_K_star(time_vector,number,hyperparameters,'normal')
    X_estimate = np.dot(K_star_estimate,K_inv)
    estimated_values_y_forecast.append((np.dot(X_estimate,volatility_observed).tolist()))

new_estimated_values_y_forecast = []
for number in range(0,len(estimated_values_y_forecast)):
    new_estimated_values_y_forecast.append(estimated_values_y_forecast[number][0])

print new_estimated_values_y_forecast

print len(new_estimated_values_y_forecast)

for index in range(0,len(forecast_volatility)):
    forecast_volatility[index]-=average

sum_errors = 0

x=sample_size+forecast_period

for index in range(sample_size,sample_size+forecast_period):
    sum_errors+=fabs(forecast_volatility[index]-new_estimated_values_y_forecast[index-sample_size])**2

error = sum_errors/forecast_period

print error




plt.show()







import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import GP_functions_2 as gp
from pyDOE import lhs
from math import fabs

# ----------------------------------------------------------------------------
#Data input
volatility_observed = gp.get_volatility(96)
time_vector = gp.get_time_vector(volatility_observed)
print time_vector
print volatility_observed
new_x = np.array([0.2])


#Latin Hypercube Initlialziation
print 'Starting...'
latin_hypercube_values = lhs(5, samples=2)
latin_hypercube_values=latin_hypercube_values*5

#Optimization part
result=np.zeros((len(latin_hypercube_values),5))
for number in range(0,len(latin_hypercube_values)):
    print number
    wynik= optimize.minimize(gp.function_to_minimize_volatility, latin_hypercube_values[number],method='BFGS')
    result[number]=wynik['x']

likelihood=np.zeros((len(latin_hypercube_values),1))
for number in range(0,len(latin_hypercube_values)):
    likelihood[number] = gp.function_to_minimize_volatility(result[number])
min_index = np.argmin(likelihood)
print likelihood
print min_index
print result[min_index]
hyperparameters = result[min_index]




#Finding values of K matrices for new values of x
K = gp.find_K(time_vector,hyperparameters)
K_2stars_estimate = gp.find_K_2stars(new_x,hyperparameters)
K_inv = np.linalg.inv(K)

#--------------------------------

#Vector of new x values
new_values = np.arange(0,300,0.201)

#Initialise matrices to store estimated values of y(volatility) and variance
estimated_values_y = []
estimated_variance_y = []

#Initialise variable to store size of the variance and y vector
estimated_variance_size = 0

#Find y and variance for all 'new values'
for number in new_values:
    K_star_estimate = gp.find_K_star(time_vector,number,hyperparameters)
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
print new_estimated_variance_y
for number in range(0,len(new_estimated_variance_y)):
    new_estimated_variance_y_1.append(new_estimated_values_y[number]+new_estimated_variance_y[number])
    new_estimated_variance_y_2.append(new_estimated_values_y[number]-new_estimated_variance_y[number])
print new_estimated_variance_y_1
print new_estimated_variance_y_2

#Plotting one new value
plt.fill_between(new_values, new_estimated_variance_y_2,new_estimated_variance_y_1,alpha=0.5)
plt.plot(new_values,new_estimated_values_y, 'g-')
plt.plot(time_vector, volatility_observed, 'r.', markersize = 5)
plt.plot(gp.get_time_vector(gp.get_volatility('all')), gp.get_volatility('all'), 'r.', markersize = 5)
#plt.axis([min(sample_vector)-0.5, max(sample_vector)+0.5, min(sample_y)-0.5, max(sample_y)+0.5])
plt.axis([0,len(time_vector)+50,-3,-1])
plt.show()


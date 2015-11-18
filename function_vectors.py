__author__ = 'Mat'


from math import exp
from math import log10
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import GP_functions as gp
from pyDOE import lhs

# ----------------------------------------------------------------------------


#Example of x and y vectors
sample_vector = gp.get_x_values_2d()
sample_y = gp.get_y_values_2d()
new_x = np.array([0.2])
print 'Starting...'
latin_hypercube_values = lhs(3, samples=100)
latin_hypercube_values=latin_hypercube_values

#Optimization part
result=np.zeros((len(latin_hypercube_values),3))
for number in range(0,len(latin_hypercube_values)):
    wynik= optimize.minimize(gp.function_to_minimize, latin_hypercube_values[number],method='BFGS')
    result[number]=wynik['x']

likelihood=np.zeros((len(latin_hypercube_values),1))
for number in range(0,len(latin_hypercube_values)):
    likelihood[number] = gp.function_to_minimize(result[number])
min_index = np.argmin(likelihood)
print result[min_index]
sigma_f = result[min_index][0]
sigma_n = result[min_index][1]
length = result[min_index][2]

"""sigma_f = 1.27
sigma_n = 0.3
length = 1"""



#Finding values of K matrices for new values of x
K = gp.find_K(sample_vector,sigma_f,sigma_n,length)
K_star = gp.find_K_star(sample_vector,new_x,sigma_f,sigma_n,length)
K_2stars = gp.find_K_2stars(new_x,sigma_f,sigma_n,length)

#Find y*
K_inv = np.linalg.inv(K)
X = np.dot(K_star,K_inv)
y_star = np.dot(X, sample_y)

#Find variance of y*
K_star_trans = K_star.transpose()
y_star_var = K_2stars-np.dot(K_star,np.dot(K_inv,K_star_trans))

#--------------------------------

#Vector of new x values
new_values = np.arange(-5,5,0.001)
estimated_values_y = []
estimated_variance_y = []
for number in new_values:
    K_star_estimate = gp.find_K_star(sample_vector,number,sigma_f,sigma_n,length)
    K_2stars_estimate = gp.find_K_2stars(number,sigma_f,sigma_n,length)
    X_estimate = np.dot(K_star_estimate,K_inv)
    # Without conversion to float we have a list of arrays
    estimated_values_y.append((np.dot(X_estimate,sample_y).tolist()))
    K_star_trans_estimate = K_star_estimate.transpose()
    estimated_variance_y.append(((1.96*(K_2stars_estimate-np.dot(K_star_estimate,np.dot(K_inv,K_star_trans_estimate)))**0.5)).tolist())

new_estimated_values_y = []
for number in range(0,len(estimated_values_y)):
    new_estimated_values_y.append(estimated_values_y[number][0][0])

new_estimated_variance_y = []
for number in range(0,len(estimated_variance_y)):
    new_estimated_variance_y.append(estimated_variance_y[number][0][0])

#Plotting estimated curve
plt.errorbar(new_values,new_estimated_values_y, yerr=new_estimated_variance_y, capsize=0)
#Plotting one new value
plt.plot(sample_vector, sample_y, 'ro')
#plt.axis([min(sample_vector)-0.5, max(sample_vector)+0.5, min(sample_y)-0.5, max(sample_y)+0.5])
plt.axis([-4,4,-4,4])
#plt.errorbar(sample_vector, sample_y, yerr=0.3, fmt='ro', capsize=0)
#plt.plot([0.2],[y_star], 'go')
plt.show()


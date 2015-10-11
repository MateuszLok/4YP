__author__ = 'Mat'


from math import exp
from math import log10
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import GP_functions as gp

# ----------------------------------------------------------------------------

#Example of x and y vectors
sample_vector = [ -1.5, -1.00, -0.75, -0.40, -0.25, 0.00]
sample_y = [-1.7, -1.2, -0.35, 0.1, 0.5, 0.75]

#Finding values of K matrices for new values of x
K = gp.find_K(sample_vector,1.27,0.3,1.0)
K_star = gp.find_K_star(sample_vector,0.2,1.27,0.3,1.0)
K_2stars = gp.find_K_2stars(0.2,1.27,0.3,1.0)

#Find y*
K_inv = np.linalg.inv(K)
X = np.dot(K_star,K_inv)
y_star = np.dot(X, sample_y)

#Find variance of y*
K_star_trans = K_star.transpose()
y_star_var = K_2stars-np.dot(K_star,np.dot(K_inv,K_star_trans))


#--------------------------------

#Optimization part

result = optimize.minimize(gp.function_to_minimize, [3,0.5,3])

wynik = result['x']



#--------------------------------

#Vector of new x values
new_values = np.arange(-2,0.5,0.001)
estimated_values_y = []
estimated_variance_y = []
for number in new_values:
    K_star_estimate = gp.find_K_star(sample_vector,number,1.27,0.3,1.0)
    K_2stars_estimate = gp.find_K_2stars(number,1.27,0.3,1.27)
    X_estimate = np.dot(K_star_estimate,K_inv)
    # Without conversion to float we have a list of arrays
    estimated_values_y.append(float(np.dot(X_estimate,sample_y)))

    K_star_trans_estimate = K_star_estimate.transpose()
    estimated_variance_y.append((float(1.96*(K_2stars_estimate-np.dot(K_star_estimate,np.dot(K_inv,K_star_trans_estimate))**0.5))))


#Plotting estimated curve
plt.plot(new_values,estimated_values_y)
plt.errorbar(new_values,estimated_values_y, yerr=estimated_variance_y, capsize=0)

#Plotting one new value
plt.plot(sample_vector, sample_y, 'ro')
#plt.axis([min(sample_vector)-0.5, max(sample_vector)+0.5, min(sample_y)-0.5, max(sample_y)+0.5])
plt.axis([-1.7,0.4,-2.6,1.7])
plt.errorbar(sample_vector, sample_y, yerr=0.3, fmt='ro', capsize=0)
plt.plot([0.2],[y_star], 'go')
plt.show()


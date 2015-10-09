__author__ = 'Mat'

from math import exp
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------

 
#Evaluates the covariance function 'k' for 2 values of x
def calculate_k(x1,x2):
    sigma_y = 1.27
    sigma_n = 0.3
    if x1 ==x2:
        return (sigma_y ** 2) * exp(-0.5 * (x1-x2) ** 2 ) + sigma_n ** 2
    else:
        return (sigma_y ** 2) * exp(-0.5 * (x1-x2) ** 2 )

#Evaluates matrix K
def find_K(vector_X):
    outcome = np.zeros((len(vector_X),len(vector_X)))
    for i in range(0, len(vector_X)):
        for j in range(0,len(vector_X)):
            outcome[i][j] = (calculate_k(vector_X[i],vector_X[j]))
    return outcome

#Evaluates matrix K*
def find_K_star(vector,x):
    outcome = np.zeros((1, len(vector)))
    for i in range(0,len(vector)):
        outcome[0][i] = calculate_k(x,vector[i])
    return outcome

#Evaluates matrix K**
def find_K_2stars(x):
    return calculate_k(x,x)


# ----------------------------------------------------------------------------


#Example of x and y vectors
sample_vector = [ -1.5, -1.00, -0.75, -0.40, -0.25, 0.00]
sample_y = [-1.7, -1.2, -0.35, 0.1, 0.5, 0.75]

#Finding values of K matrices for new values of x
K = find_K(sample_vector)
K_star = find_K_star(sample_vector,0.2)
K_2stars = find_K_2stars(0.2)

#Find y*
K_inv = np.linalg.inv(K)
X = np.dot(K_star,K_inv)
y_star = np.dot(X, sample_y)
print y_star

#Find variance of y*
K_star_trans = K_star.transpose()
y_star_var = K_2stars-np.dot(K_star,np.dot(K_inv,K_star_trans))
print y_star_var

#--------------------------------

#Vector of new x values
new_values = np.arange(-2,0.5,0.001)
estimated_values_y = []
estimated_variance_y = []
for number in new_values:
    K_star_estimate = find_K_star(sample_vector,number)
    K_2stars_estimate = find_K_2stars(number)
    X_estimate = np.dot(K_star_estimate,K_inv)
    # Without conversion to float we have a list of arrays
    estimated_values_y.append(float(np.dot(X_estimate,sample_y)))

    K_star_trans_estimate = K_star_estimate.transpose()
    estimated_variance_y.append((float(1.96*(K_2stars_estimate-np.dot(K_star_estimate,np.dot(K_inv,K_star_trans_estimate))**0.5))))

print estimated_variance_y



#Plotting estimated curve
plt.plot(new_values,estimated_values_y,)
plt.errorbar(new_values,estimated_values_y, yerr=estimated_variance_y, capsize=0)

#Plotting one new value
plt.plot(sample_vector, sample_y, 'ro')
#plt.axis([min(sample_vector)-0.5, max(sample_vector)+0.5, min(sample_y)-0.5, max(sample_y)+0.5])
plt.axis([-1.7,0.4,-2.6,1.7])
plt.errorbar(sample_vector, sample_y, yerr=0.3, fmt='ro', capsize=0)
plt.plot([0.2],[y_star], 'go')
plt.show()


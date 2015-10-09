__author__ = 'Mat'

from math import exp
import numpy as np
import matplotlib.pyplot as plt
 
#Evaluates the covariance function 'k' for 2 values of x
def calculate_k(x1,x2):
    sigma_y = 1.27
    sigma_n = 0.3
    if x1 ==x2:
        return (sigma_y ** 2) * exp(-0.5 * (x1-x2) ** 2 ) + sigma_n **2
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


#Example of x and y vectors
sample_vector = [-2.0, -1.5, -1.00, -0.75, -0.40, -0.25, 0.00]
sample_y = [-2.2, -1.7, -1.2, -0.35, 0.1, 0.5, 0.75]

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

plt.plot(sample_vector, sample_y, 'ro')
plt.axis([min(sample_vector)-0.5, max(sample_vector)+0.5, min(sample_y)-0.5, max(sample_y)+0.5])
plt.plot([0.2],[y_star], 'go')
plt.show()


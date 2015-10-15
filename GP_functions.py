__author__ = 'Mat'

from math import exp
import numpy as np
from math import log10

#Define input data
"""def get_x_values():
    return np.array([-1.5, -1.00, -0.75, -0.40, -0.25, 0.00])

def get_y_values():
    return np.array([-1.7, -1.25, -0.35, 0.1, 0.5, 0.75])"""

"""def get_x_values():
    return np.array(([-1.5],[-1.00],[-0.75],[-0.40],[-0.25],[0.0]),dtype=float)

def get_y_values():
    return np.array(([-1.7], [-1.25], [-0.35], [0.1], [0.5], [0.75]),dtype=float)"""


def get_x_values():
    return np.array(([-1.5,1],[-1.00,1],[-0.75,1],[-0.40,1],[-0.25,1],[0.0,1]),dtype=float)

def get_y_values():
    return np.array(([-1.7], [-1.25], [-0.35], [0.1], [0.5], [0.75]),dtype=float)




#Calculate covariance for two values
#Non-vector format
def calculate_kk(x1,x2,f,n,ll):
    sigma_f = f
    sigma_n = n
    l=ll
    if x1 == x2:
        return (sigma_f ** 2) * exp(-(x1-x2) ** 2 * (2 * l ** 2) ** -1) + sigma_n ** 2
    else:
        return (sigma_f ** 2) * exp(-(x1-x2) ** 2 * (2 * l ** 2) ** -1 )

#Vector format
def calculate_k(x1,x2,f,n,ll):
    sigma_f = f
    sigma_n = n
    l=ll
    difference = x1-x2
    intermediate = -np.dot(difference.transpose(),difference) * (2 * l ** 2) ** -1
    if (x1==x2).all():
        return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2
    else:
        return (sigma_f ** 2) * exp(intermediate)


#Evaluates matrix K - covariance matrix
#Vector format
def find_K(vector_X, f, n, l):
    sigma_f = f
    l = l
    sigma_n = n
    outcome = np.zeros((len(vector_X),len(vector_X)))
    for i in range(0, len(vector_X)):
        for j in range(0,len(vector_X)):
            outcome[i][j] = (calculate_k(vector_X[i,:],vector_X[j,:],sigma_f,sigma_n,l))
    return outcome

#Non-Vector Format
def find_KK(vector_X, f, n, l):
    sigma_f = f
    l = l
    sigma_n = n
    outcome = np.zeros((len(vector_X),len(vector_X)))
    for i in range(0, len(vector_X)):
        for j in range(0,len(vector_X)):
            outcome[i][j] = (calculate_k(vector_X[i],vector_X[j],sigma_f,sigma_n,l))
    return outcome

#Evaluates matrix K*
#Non-vector format
def find_K_starr(vector,x,f,n,l):
    sigma_f = f
    l = l
    sigma_n = n
    outcome = np.zeros((1, len(vector)))
    for i in range(0,len(vector)):
        outcome[0][i] = calculate_k(x,vector[i],sigma_f, sigma_n,l)
    return outcome

#Vector format
def find_K_star(vector,x,f,n,l):
    sigma_f = f
    l = l
    sigma_n = n
    outcome = np.zeros((1, len(vector)))
    for i in range(0,len(vector)):
        outcome[0][i] = calculate_k(x,vector[i,:],sigma_f, sigma_n,l)
    return outcome


#Evaluates matrix K**
def find_K_2stars(x,f,n,l):
    sigma_f = f
    l = l
    sigma_n = n
    return calculate_k(x,x,sigma_f,sigma_n,l)


#Function to optimize


def function_to_minimize(input_data):
    f = input_data[0]
    n = input_data[1]
    l = input_data[2]
    vector_x= get_x_values()
    sample_y_opt = np.array(get_y_values())
    sample_y_trans_opt = sample_y_opt.transpose()
    return 0.5 * np.dot(sample_y_trans_opt, np.dot(np.linalg.inv(find_K(vector_x,f,n,l)), sample_y_opt)) +0.5 * log10(np.linalg.det(find_K(vector_x,f,n,l))) + log10(6.28)

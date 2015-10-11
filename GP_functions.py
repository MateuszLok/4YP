__author__ = 'Mat'

from math import exp
import numpy as np
from math import log10
import matplotlib.pyplot as plt
import scipy.optimize as optimize

#Write
def calculate_k(x1,x2,f,n,l):
    sigma_f = f
    sigma_n = n
    if x1 == x2:
        return (sigma_f ** 2) * exp(-(x1-x2) ** 2 * (2 * l ** 2) ** -1) + sigma_n ** 2
    else:
        return (sigma_f ** 2) * exp(-(x1-x2) ** 2 * (2 * l ** 2) ** -1 )

#Evaluates matrix K
def find_K(vector_X, f, n, l):
    sigma_f = f
    l = l
    sigma_n = n
    outcome = np.zeros((len(vector_X),len(vector_X)))
    for i in range(0, len(vector_X)):
        for j in range(0,len(vector_X)):
            outcome[i][j] = (calculate_k(vector_X[i],vector_X[j],sigma_f,sigma_n,l))
    return outcome

#Evaluates matrix K*
def find_K_star(vector,x,f,n,l):
    sigma_f = f
    l = l
    sigma_n = n
    outcome = np.zeros((1, len(vector)))
    for i in range(0,len(vector)):
        outcome[0][i] = calculate_k(x,vector[i],sigma_f, sigma_n,l)
    return outcome

#Evaluates matrix K**
def find_K_2stars(x,f,n,l):
    sigma_f = f
    l = l
    sigma_n = n
    return calculate_k(x,x,sigma_f,sigma_n,l)


#----------------Optimization
def function_to_minimize(input_data):
    f = input_data[0]
    n = input_data[1]
    l = input_data[2]
    vector_x= [ -1.5, -1.00, -0.75, -0.40, -0.25, 0.00]
    sample_y_opt = np.array([-1.7, -1.2, -0.35, 0.1, 0.5, 0.75])
    sample_y_trans_opt = sample_y_opt.transpose()
    return 0.5 * np.dot(sample_y_trans_opt, np.dot(np.linalg.inv(find_K(vector_x,f,n,l)), sample_y_opt)) +0.5 * log10(np.linalg.det(find_K(vector_x,f,n,l))) + log10(6.28)

__author__ = 'Mat'

from math import exp
import numpy as np
from math import log10
from math import fabs
import json


#Fetch actual market data from a JSON file
def get_volatility():
    json_data = open('oil_futures.json').read()
    data = json.loads(json_data)
    days = len(data["dataset"]["data"])
    volatility = np.zeros((1,days))
    for number in range(0,days):
        #print data["dataset"]["data"][number][0]
        max = data["dataset"]["data"][number][2]
        min = data["dataset"]["data"][number][3]
        if type(max) != float or type(min)!= float or min ==0  or max==0:
            volatility[0][number]=volatility[0][number-1]
        elif (log10(max)-log10(min)) == 0:
            volatility[0][number]=volatility[0][number-1]
        else:
            volatility[0][number]=log10(log10(max)-log10(min))

    return volatility[0]

def get_time_vector(x):
    return np.array((range(0,len(x),1)))


#Define sample input vectors
def get_x_values_2d():
    return np.array(([-1.5],[-1.49],[-1.00],[-0.75],[-0.40],[-0.25],[-0.24],[0.0],[0.1]),dtype=float)

def get_y_values_2d():
    return np.array(([-1.7], [-1.69], [-1.25], [-0.35], [0.1], [0.5],[0.51], [0.75],[0.82]),dtype=float)


def get_x_values_3d():
    return np.array(([-1.5,-1],[-1.00,-1],[-0.75,-1],[-0.40,-1],[-0.25,-1],[0.0,-1.0]),dtype=float)

def get_y_values_3d():
    return np.array(([-1.5], [-1.25], [-0.35], [0.1], [0.5], [0.75]),dtype=float)

def get_x_values_4d():
    return np.array(([-1.5,-1,1],[-1.00,-1.2,1],[-0.75,-0,1],[-0.40,0.5,1],[-0.25,-0.25,1],[0.0,-1.0,1]),dtype=float)

def get_y_values_4d():
    return np.array(([-1.5], [-1.25], [-0.35], [0.1], [0.5], [0.75]),dtype=float)



#Calculate covariance for two values
def calculate_kk(x1,x2,sigma_f,sigma_n,l):
    difference = x1-x2
    l=exp(l)
    sigma_f=exp(sigma_f)
    sigma_n=exp(sigma_n)
#3D
    if x1.shape == (2,):
        intermediate = -0.5*np.dot(np.dot(difference.transpose(),np.linalg.inv(l)),difference)
        #Compares if all entries in each vector are the same, hence whether the vectors are equal
        if (x1==x2).all():
            return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2
        else:
            return (sigma_f ** 2) * exp(intermediate)
#2D
    else:
        intermediate = -np.dot(difference.transpose(),difference) * (2 * l ** 2) ** -1
        if (x1==x2).all():
            return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2
        else:
            return (sigma_f ** 2) * exp(intermediate)

#Not squared
def calculate_k(x1,x2,sigma_f,sigma_n,l):
    difference = x1-x2
    #print sigma_n
    l=exp(l)
    sigma_f=exp(sigma_f)
    sigma_n=exp(sigma_n)
    intermediate = -fabs(difference) * (l **-1)
    if (x1==x2).all():
        return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2
    else:
        return (sigma_f ** 2) * exp(intermediate)



#Evaluates matrix K - covariance matrix
#Vector format
def find_K(vector_X, sigma_f, sigma_n, l):
    outcome = np.zeros((len(vector_X),len(vector_X)))
    for i in range(0, len(vector_X)):
        for j in range(i,len(vector_X)):
            temp = (calculate_k(vector_X[i],vector_X[j],sigma_f,sigma_n,l))
            outcome[i][j] = temp
            outcome[j][i] = temp

    return outcome

#Evaluates matrix K*
def find_K_star(vector,x,sigma_f,sigma_n,l):
    outcome = np.zeros((1, len(vector)))
    for i in range(0,len(vector)):
        outcome[0][i] = calculate_k(x,vector[i],sigma_f, sigma_n,l)
    return outcome

#Evaluates matrix K**
def find_K_2stars(x,sigma_f,sigma_n,l):
    return calculate_k(x,x,sigma_f,sigma_n,l)

#Function to optimize
def function_to_minimize(input_data):
    #print len(input_data)
    if len(input_data)!=1:
        #print input_data
        f = input_data[0]
        n = input_data[1]
        l = input_data[2]
    else:
        print 'lol'
        f = input_data[0][0]
        n = input_data[0][1]
        l = input_data[0][2]

    vector_x= get_x_values_2d()
    sample_y_opt = np.array(get_y_values_2d())
    sample_y_trans_opt = sample_y_opt.transpose()
    chol=np.linalg.cholesky(find_K(vector_x,f,n,l))
    transient = 2*cholesky_det(chol)
    chol_trans = chol.transpose()
    K_inverse = np.dot(np.linalg.inv(chol_trans),np.linalg.inv(chol))
    return (0.5 * np.dot(sample_y_trans_opt, np.dot(K_inverse, sample_y_opt)) +0.5 * transient + log10(6.28))[0]

def function_to_minimize_3d(input_data):
    f = input_data[0]
    n = input_data[1]
    l1= input_data[2]
    l2 = input_data[3]
    l = np.array(([l1,0],[0,l2]),dtype=float)

    vector_x= get_x_values_3d()
    sample_y_opt = np.array(get_y_values_3d())
    sample_y_trans_opt = sample_y_opt.transpose()
    return 0.5 * np.dot(sample_y_trans_opt, np.dot(np.linalg.inv(find_K(vector_x,f,n,l)), sample_y_opt)) \
           +0.5 * log10(fabs(np.linalg.det(find_K(vector_x,f,n,l)))) + log10(6.28)

def function_to_minimize_volatility(input_data):
    if len(input_data)!=1:
        f = input_data[0]
        n = input_data[2]
        l = input_data[1]
    else:
        f = input_data[0][0]
        n = input_data[0][2]
        l = input_data[0][1]

    vector_x= get_time_vector(get_volatility())
    sample_y_opt = np.array(get_volatility())
    sample_y_trans_opt = sample_y_opt.transpose()
    chol=np.linalg.cholesky(find_K(vector_x,f,n,l))
    transient = 2*cholesky_det(chol)
    chol_trans = chol.transpose()
    K_inverse = np.dot(np.linalg.inv(chol_trans),np.linalg.inv(chol))
    return (0.5 * np.dot(sample_y_trans_opt, np.dot(K_inverse, sample_y_opt)) +0.5 * transient + log10(6.28))

def function_to_minimize_volatility_nonoise(input_data):
    if len(input_data)!=1:
        f = input_data[0]
        l = input_data[1]
        n=0.0001
    else:
        f = input_data[0][0]
        l = input_data[0][1]
        n=0.0001

    vector_x= get_time_vector(get_volatility())
    sample_y_opt = np.array(get_volatility())
    sample_y_trans_opt = sample_y_opt.transpose()
    chol=np.linalg.cholesky(find_K(vector_x,f,n,l))
    transient = 2*cholesky_det(chol)
    chol_trans = chol.transpose()
    K_inverse = np.dot(np.linalg.inv(chol_trans),np.linalg.inv(chol))
    return (0.5 * np.dot(sample_y_trans_opt, np.dot(K_inverse, sample_y_opt)) +0.5 * transient + log10(6.28))

def cholesky_det(R):
    sum=0
    for number in range(0,len(R)):
        sum=sum+log10(R[number,number])
    return sum


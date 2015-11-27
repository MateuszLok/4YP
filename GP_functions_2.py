__author__ = 'Mat'

from math import exp
import numpy as np
from math import log10
from math import fabs
import json


#Fetch actual market data from a JSON file
def get_volatility(x):
    json_data = open('oil_futures.json').read()
    data = json.loads(json_data)
    days = len(data["dataset"]["data"])
    volatility = np.zeros((1,days))
    for number in range(0,days):
        max = data["dataset"]["data"][days-number-1][2]
        min = data["dataset"]["data"][days-number-1][3]
        if type(max) != float or type(min)!= float or min ==0  or max==0:
            volatility[0][number]=volatility[0][number-1]
        elif (log10(max)-log10(min)) == 0:
            volatility[0][number]=volatility[0][number-1]
        else:
            volatility[0][number]=log10(log10(max)-log10(min))
    if x=='all':
        return volatility[0]
    else:
        return volatility[0][0:x]

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
def calculate_kk(x1,x2,theta):
    difference = x1-x2
    l=exp(theta[1])
    sigma_f=exp(theta[0])
    sigma_n=exp(theta[2])
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
def calculate_kk(x1,x2,theta):
    difference = x1-x2
    #print sigma_n
    l=exp(theta[0])
    sigma_f=exp(theta[1])
    sigma_n=exp(theta[2])
    intermediate = -fabs(difference) * (l **-1)
    if (x1==x2).all():
        return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2
    else:
        return (sigma_f ** 2) * exp(intermediate)

#As in the paper
def calculate_k(x1,x2,theta):
    difference = x1-x2
    l1=exp(theta[1])
    sigma_f1=exp(theta[0])
    l2=exp(theta[2])
    sigma_f2=(theta[3])
    sigma_n=theta([4])
    intermediate1 = -fabs(difference) * (l1 **-1)
    intermediate2 = -fabs(difference) * (l2 **-1)

    if (x1==x2).all():
        return (sigma_f1 ** 2) * exp(intermediate1)+ (sigma_f2 ** 2) * exp(intermediate2) + sigma_n ** 2
    else:
        return (sigma_f1 ** 2) * exp(intermediate2) + (sigma_f2 ** 2) * exp(intermediate2)


#Evaluates matrix K - covariance matrix
#Vector format
def find_K(vector_X, theta):
    outcome = np.zeros((len(vector_X),len(vector_X)))
    for i in range(0, len(vector_X)):
        for j in range(i,len(vector_X)):
            temp = (calculate_k(vector_X[i],vector_X[j],theta))
            outcome[i][j] = temp
            outcome[j][i] = temp

    return outcome

#Evaluates matrix K*
def find_K_star(vector,x,theta):
    outcome = np.zeros((1, len(vector)))
    for i in range(0,len(vector)):
        outcome[0][i] = calculate_k(x,vector[i],theta)
    return outcome

#Evaluates matrix K**
def find_K_2stars(x,theta):
    return calculate_k(x,x,theta)

#Function to optimize
def function_to_minimize(input_data):
    vector_x= get_x_values_2d()
    sample_y_opt = np.array(get_y_values_2d())
    sample_y_trans_opt = sample_y_opt.transpose()
    chol=np.linalg.cholesky(find_K(vector_x,input_data))
    transient = 2*cholesky_det(chol)
    chol_trans = chol.transpose()
    K_inverse = np.dot(np.linalg.inv(chol_trans),np.linalg.inv(chol))
    return (0.5 * np.dot(sample_y_trans_opt, np.dot(K_inverse, sample_y_opt)) +0.5 * transient + log10(6.28))[0]

def function_to_minimize_volatility(input_data):
    vector_x= get_time_vector(get_volatility(96))
    sample_y_opt = np.array(get_volatility(96))
    sample_y_trans_opt = sample_y_opt.transpose()
    chol=np.linalg.cholesky(find_K(vector_x,input_data))
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


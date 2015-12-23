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

#Simple case for testing derivative
def calculate_k(x1,x2,theta,type):
    if type=="normal":
        difference = x1-x2
        #print sigma_n
        l=exp(theta[0])
        sigma_f=exp(theta[1])
        sigma_n=0.001
        intermediate = -fabs(difference) * (l **-1)
        if (x1==x2).all():
            return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2
        else:
            return (sigma_f ** 2) * exp(intermediate)
    elif type=="der_l":
        difference = x1-x2
        #print sigma_n
        l=exp(theta[0])
        sigma_f=exp(theta[1])
        sigma_n=0.001
        intermediate = -fabs(difference) * (l **-1)
        return (sigma_f ** 2) *fabs(difference)*(l)**-2 * exp(intermediate)
    elif type=="der_f":
        difference = x1-x2
        #print sigma_n
        l=exp(theta[0])
        sigma_f=exp(theta[1])
        sigma_n=0.001
        intermediate = -fabs(difference) * (l **-1)
        return 2*sigma_f*exp(intermediate)


#As in the paper
def calculate_kk(x1,x2,theta):
    difference = x1-x2
    l1=exp(theta[1])
    sigma_f1=exp(theta[0])
    l2=exp(theta[2])
    sigma_f2=exp(theta[3])
    sigma_n=exp(theta[4])
    intermediate1 = -fabs(difference) * (l1 **-1)
    intermediate2 = -fabs(difference) * (l2 **-1)

    if (x1==x2).all():
        return (sigma_f1 ** 2) * exp(intermediate1)+ (sigma_f2 ** 2) * exp(intermediate2) + sigma_n ** 2
    else:
        return (sigma_f1 ** 2) * exp(intermediate2) + (sigma_f2 ** 2) * exp(intermediate2)


#Evaluates matrix K - covariance matrix
#Vector format
def find_K(vector_X, theta,type):
    outcome = np.zeros((len(vector_X),len(vector_X)))
    for i in range(0, len(vector_X)):
        for j in range(i,len(vector_X)):
            temp = (calculate_k(vector_X[i],vector_X[j],theta,type))
            outcome[i][j] = temp
            outcome[j][i] = temp

    return outcome


#Evaluates matrix K*
def find_K_star(vector,x,theta,type):
    outcome = np.zeros((1, len(vector)))
    for i in range(0,len(vector)):
        outcome[0][i] = calculate_k(x,vector[i],theta,type)
    return outcome

#Evaluates matrix K**
def find_K_2stars(x,theta,type):
    return calculate_k(x,x,theta,type)

#Function to optimize
def function_to_minimize_volatility(input_data):
    vector_x= get_time_vector(get_volatility(90))
    sample_y_opt = np.array(get_volatility(90))
    sample_y_trans_opt = sample_y_opt.transpose()
    chol=np.linalg.cholesky(find_K(vector_x,input_data,'normal'))
    transient = 2*cholesky_det(chol)
    chol_trans = chol.transpose()
    K_inverse = np.dot(np.linalg.inv(chol_trans),np.linalg.inv(chol))
    return (0.5 * np.dot(sample_y_trans_opt, np.dot(K_inverse, sample_y_opt)) +0.5 * transient + log10(6.28))


def cholesky_det(R):
    sum=0
    for number in range(0,len(R)):
        sum=sum+log10(R[number,number])
    return sum


def K_inverse_function(input_data,l):
    vector_x= get_time_vector(get_volatility(l))
    sample_y_opt = np.array(get_volatility(l))
    sample_y_trans_opt = sample_y_opt.transpose()
    chol=np.linalg.cholesky(find_K(vector_x,input_data,'normal'))
    transient = 2*cholesky_det(chol)
    chol_trans = chol.transpose()
    K_inverse = np.dot(np.linalg.inv(chol_trans),np.linalg.inv(chol))
    return K_inverse

def jacobian_of_likelihood(input_data):
    length=90
    vector_x= get_time_vector(get_volatility(length))
    K_inv = K_inverse_function(input_data,length)
    sample_y_opt = np.array(get_volatility(length))
    sample_y_trans_opt = sample_y_opt.transpose()
    ytKinv=np.dot(sample_y_trans_opt,K_inv)
    Kinvy=np.dot(K_inv,sample_y_opt)
    Kf=find_K(vector_x,input_data,'der_f')
    Kl=find_K(vector_x,input_data,'der_l')
    Kf=-0.5*np.dot(ytKinv,np.dot(Kf,Kinvy))+0.5*np.trace(np.dot(K_inv,Kf))
    Kl=-0.5*np.dot(ytKinv,np.dot(Kl,Kinvy))+0.5*np.trace(np.dot(K_inv,Kl))
    return np.array([Kl,Kf])



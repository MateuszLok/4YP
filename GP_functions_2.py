__author__ = 'Mat'

from math import exp
import numpy as np
from math import log10
from math import fabs
import json
from math import sin
from math import log
import scipy as sp


#Fetch actual market data from a JSON file
def get_volatility(x):
    json_data = open('apple2005.json').read()
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
            volatility[0][number]=log(log(max)-log(min))
    if x=='all':
        return volatility[0]
    else:
        return volatility[0][0:x]

def get_time_vector(x):
    return np.array((range(0,len(x),1)))

#------------------------------------------------------------------------------

#Calculate covariance for two values
def calculate_kk(x1,x2,theta,type,j):
    if type=='normal':
        difference = x1-x2
        l=(theta[0])
        sigma_f=(theta[1])
        sigma_n=(theta[2])
        intermediate = -np.dot(difference.transpose(),difference) * (2 * l ** 2) ** -1
        if (x1==x2).all():
            return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2 + j
        else:
            return (sigma_f ** 2) * exp(intermediate)


#Matern 3/2

def calculate_k(x1,x2,theta,type,j):
    if type=='normal':
        difference = x1-x2
        l=(theta[0])
        sigma_f=(theta[1])
        sigma_n=(theta[2])
        intermediate = -(3)**(0.5)*fabs(difference)*(l **-2)
        if (x1==x2).all():
            return (sigma_f ** 2)*(1+(3)**(0.5)*fabs(difference)*(l **-2)) * exp(intermediate) + sigma_n ** 2 + j
        else:
            return (sigma_f ** 2)*(1+(3)**(0.5)*fabs(difference)*(l **-2)) * exp(intermediate)

#Matern 5/2
def calculate_kk(x1,x2,theta,type,j):
    if type=='normal':
        difference = x1-x2
        l=(theta[0])
        sigma_f=(theta[1])
        sigma_n=(theta[2])
        intermediate = -(5)**(0.5)*fabs(difference)*(l **-1)
        if (x1==x2).all():
            return (sigma_f ** 2)*(1+(5)**(0.5)*fabs(difference)*(l **-1)+5*(difference)**2/(3*l**2)) * exp(intermediate) + sigma_n ** 2 +j
        else:
            return (sigma_f ** 2)*(1+(5)**(0.5)*fabs(difference)*(l **-1)+5*(difference)**2/(3*l**2)) * exp(intermediate)


#Not squared


def calculate_kk(x1,x2,theta,type,j):
    if type=='normal':
        difference = x1-x2
        #print sigma_n
        l=(theta[0])
        sigma_f=(theta[1])
        sigma_n=(theta[2])
        intermediate = -fabs(difference) * (l **-2)
        if (x1==x2).all():
            return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2 + j
        else:
            return (sigma_f ** 2) * exp(intermediate)

#Simple case for testing derivative
def calculate_kk(x1,x2,theta,type):
    l=exp(theta[0])
    sigma_f=exp(theta[1])
    sigma_n=exp(theta[2])
    difference=x1-x2
    if type=="normal":
        intermediate = -fabs(difference) * (l **-1)
        if (x1==x2).all():
            return (sigma_f)**2 * exp(intermediate) + sigma_n ** 2
        else:
            return (sigma_f)**2 * exp(intermediate)
    elif type=="der_l":
        intermediate = -fabs(difference) * (l **-1)
        return (sigma_f) *fabs(difference)*(l)**-2 * exp(intermediate)
    elif type=="der_f":
        intermediate = -fabs(difference) * (l **-1)
        return exp(intermediate)
    elif type=="der_n":
        if (x1==x2).all():
            return 2*sigma_n
        else:
            return 0


#As in the paper
def calculate_kk(x1,x2,theta,type,j):
    difference = x1-x2
    l1=exp(theta[0])
    sigma_f1=exp(theta[2])
    l2=exp(theta[1])
    sigma_f2=exp(theta[3])
    sigma_n=exp(theta[4])
    if type=='normal':
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        if (x1==x2).all():
            return (sigma_f1 ** 2) * exp(intermediate1)+ (sigma_f2 ** 2) * exp(intermediate2) + sigma_n ** 2 + j
        else:
            return (sigma_f1 ** 2) * exp(intermediate1) + (sigma_f2 ** 2) * exp(intermediate2)
    if type=='der_l1':
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        return (sigma_f1 ** 2)*fabs(difference)*(l1)**-2 * exp(intermediate1)
    if type=='der_l2':
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        return (sigma_f2 ** 2)*fabs(difference)*(l2)**-2 * exp(intermediate2)
    if type=='der_f1':
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        return 2*sigma_f1*exp(intermediate1)
    if type=='der_f2':
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        return 2*sigma_f2*exp(intermediate2)
    if type=='der_n':
        if (x1==x2).all():
            return 2*sigma_n
        else:
            return 0

#With periodic term
def calculate_kk(x1,x2,theta,type,jitter):
    if type=='normal':
        difference = x1-x2
        """l1=exp(theta[0])
        l2=exp(theta[1])
        sigma_f1=exp(theta[2])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])"""
        l1=(theta[0])
        l2=(theta[1])
        sigma_f1=(theta[2])
        sigma_f2=(theta[3])
        sigma_n=(theta[4])
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = - 2* (sin((difference)*0.5))**2 * (l2 **-2)

        if (x1==x2).all():
            return (sigma_f1 ** 2) * exp(intermediate1)+ (sigma_f2 ** 2) * exp(intermediate2) + sigma_n ** 2 + jitter
        else:
            return (sigma_f1 ** 2) * exp(intermediate1) + (sigma_f2 ** 2) * exp(intermediate2)

#periodic with squared
def calculate_kk(x1,x2,theta,type,jitter):
    if type=='normal':
        difference = x1-x2
        """l1=exp(theta[0])
        l2=exp(theta[1])
        sigma_f1=exp(theta[2])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])"""
        l1=(theta[0])
        l2=(theta[1])
        sigma_f1=(theta[2])
        sigma_f2=(theta[3])
        sigma_n=(theta[4])
        intermediate1 = -np.dot(difference.transpose(),difference) * (2 * l1 ** 2) ** -1
        intermediate2 = - 2* (sin((difference)*0.5))**2 * (l2 **-2)

        if (x1==x2).all():
            return (sigma_f1 ** 2) * exp(intermediate1)+ (sigma_f2 ** 2) * exp(intermediate2) + sigma_n ** 2 + jitter
        else:
            return (sigma_f1 ** 2) * exp(intermediate1) + (sigma_f2 ** 2) * exp(intermediate2)
#periodic with 3/2
def calculate_kk(x1,x2,theta,type,jitter):
    if type=='normal':
        difference = x1-x2
        l1=(theta[0])
        l2=(theta[1])
        sigma_f1=(theta[2])
        sigma_f2=(theta[3])
        sigma_n=(theta[4])
        intermediate1 = -(3)**(0.5)*fabs(difference)*(l1 **-2)
        intermediate2 = - 2* (sin((difference)*0.5))**2 * (l2 **-2)

        if (x1==x2).all():
            return (sigma_f1 ** 2)*(1+(3)**(0.5)*fabs(difference)*(l1 **-2)) * exp(intermediate1)+ (sigma_f2 ** 2) * exp(intermediate2) + sigma_n ** 2 + jitter
        else:
            return (sigma_f1 ** 2)*(1+(3)**(0.5)*fabs(difference)*(l1 **-2)) * exp(intermediate1) + (sigma_f2 ** 2) * exp(intermediate2)
#Only periodic:
#Periodic
def calculate_kk(x1,x2,h,type):
    if type=='normal':
        difference = x1-x2
        l=exp(h[0])
        sigma_f=exp(h[1])
        sigma_n=exp(h[2])
        intermediate = - 2*(sin(difference*0.5))**2 * (l **-1)
        if (x1==x2).all():
            return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2
        else:
            return (sigma_f ** 2) * exp(intermediate)

#Test for daily with prior
def calculate_kk(x1,x2,theta,type,jitter):
    if type=='normal':
        difference = x1-x2
        l1=22
        l2=(theta[0])
        sigma_f1=(theta[1])
        sigma_f2=(theta[2])
        sigma_n=(theta[3])
        intermediate1 = -(3)**(0.5)*fabs(difference)*(l1 **-2)
        intermediate2 = - fabs(difference)*(l2**-2)

        if (x1==x2).all():
            return (sigma_f1 ** 2)*(1+(3)**(0.5)*fabs(difference)*(l1 **-2)) * exp(intermediate1)+ (sigma_f2 ** 2) * exp(intermediate2) + sigma_n ** 2 + jitter
        else:
            return (sigma_f1 ** 2)*(1+(3)**(0.5)*fabs(difference)*(l1 **-2)) * exp(intermediate1) + (sigma_f2 ** 2) * exp(intermediate2)
#-----------------------------------------------------------------------------------------------------
#Evaluates matrix K - covariance matrix
#Vector format
def find_K(vector_X, theta,type,jitter):
    outcome = np.zeros((len(vector_X),len(vector_X)))
    for i in range(0, len(vector_X)):
        for j in range(i,len(vector_X)):
            temp = (calculate_k(vector_X[i],vector_X[j],theta,type,jitter))
            outcome[i][j] = temp
            outcome[j][i] = temp

    return outcome


#Evaluates matrix K*
def find_K_star(vector,x_star,theta,type,jitter):
    outcome = np.zeros((1, len(vector)))
    for i in range(0,len(vector)):
        outcome[0][i] = calculate_k(x_star,vector[i],theta,type,jitter)
    return outcome

#Evaluates matrix K**
def find_K_2stars(x,theta,type,jitter):
    return calculate_k(x,x,theta,type,jitter)

#SPREAD
def function_to_minimize_spread(input_data, (end)):
    jitter=0
    vector_x= get_time_vector(get_volatility(end))
    monthly_volatility_observed = np.array(get_monthly_data(end,'spread'))
    average=sum(monthly_volatility_observed)/len(monthly_volatility_observed)
    for index in range(0,len(monthly_volatility_observed)):
        monthly_volatility_observed[index]-=average
    sample_y_trans_opt = monthly_volatility_observed.transpose()
    #print np.linalg.eigvals(find_K(vector_x,input_data,'normal'))
    #chol=sp.linalg.cholesky(find_K(vector_x,input_data,'normal'))

    while True:
        try:
            chol=sp.linalg.cholesky(find_K(vector_x,input_data,'normal',jitter))
            break
        except:
            print 'error'
            jitter+=0.001
            print find_K(vector_x,input_data,'normal',jitter)
            print input_data
            print jitter

    #chol=np.linalg.cholesky(find_K(vector_x,input_data,'normal'))

    transient = 2*cholesky_det(chol)
    chol_trans = chol.transpose()
    K_inverse = np.dot(np.linalg.inv(chol_trans),np.linalg.inv(chol))
    return (0.5 * np.dot(sample_y_trans_opt, np.dot(K_inverse, monthly_volatility_observed)) +0.5 * transient + log10(6.28))


#Function to optimize
def function_to_minimize_volatility(input_data, (end)):
    jitter=0
    vector_x= get_time_vector(get_volatility(end))
    sample_y_opt = np.array(get_volatility(end))
    sample_y_trans_opt = sample_y_opt.transpose()
    #print np.linalg.eigvals(find_K(vector_x,input_data,'normal'))
    #chol=sp.linalg.cholesky(find_K(vector_x,input_data,'normal',jitter))
    while True:
        try:
            chol=sp.linalg.cholesky(find_K(vector_x,input_data,'normal',jitter))
            break
        except:
            print 'error'
            jitter+=0.001

    #chol=np.linalg.cholesky(find_K(vector_x,input_data,'normal'))

    transient = 2*cholesky_det(chol)
    chol_trans = chol.transpose()
    K_inverse = np.dot(np.linalg.inv(chol_trans),np.linalg.inv(chol))
    return (0.5 * np.dot(sample_y_trans_opt, np.dot(K_inverse, sample_y_opt)) +0.5 * transient + log10(6.28))
    #return 0.5 * np.dot(sample_y_trans_opt, np.dot(K_inverse, sample_y_opt)) +0.5 * transient + log10(6.28)+ log((3*3.1415+input_data[1])**-1)+(input_data[0]-21.67)**2/(2*input_data[1]**2)

#OLDDDD
def function_to_minimize_volatilityy(input_data,(end)):

    vector_x= get_time_vector(get_volatility(end))
    sample_y_opt = np.array(get_volatility(end))
    sample_y_trans_opt = sample_y_opt.transpose()
    transient = fabs(np.linalg.det(find_K(vector_x,input_data,'normal')))
    #print input_data
    #print find_K(vector_x,input_data,'normal')
    if transient == 0:
        transient = 3.64e-164
    return (0.5 * np.dot(sample_y_trans_opt, np.dot(np.linalg.inv(find_K(vector_x,input_data,'normal')), sample_y_opt)) +0.5 * \
            log10(transient) + log10(6.28))


def cholesky_det(R):
    sum=0
    for number in range(0,len(R)):
        sum=sum+log10(R[number,number])
    return sum


def K_inverse_function(input_data,l):
    vector_x= get_time_vector(get_volatility(l))
    chol=np.linalg.cholesky(find_K(vector_x,input_data,'normal'))
    chol_trans = chol.transpose()
    K_inverse = np.dot(np.linalg.inv(chol_trans),np.linalg.inv(chol))
    return K_inverse

def jacobian_of_likelihood(input_data,end):
    length=end
    vector_x= get_time_vector(get_volatility(length))
    K_inv = K_inverse_function(input_data,length)
    sample_y_opt = np.array(get_volatility(length))
    sample_y_trans_opt = sample_y_opt.transpose()
    ytKinv=np.dot(sample_y_trans_opt,K_inv)
    Kinvy=np.dot(K_inv,sample_y_opt)
    Kf=find_K(vector_x,input_data,'der_f')
    Kl=find_K(vector_x,input_data,'der_l')
    Kf=-0.5*np.dot(ytKinv,np.dot(Kf,Kinvy))+0.5*np.trace(np.dot(K_inv,Kf))
    Kf=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kf,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kf))
    Kl=-0.5*np.dot(ytKinv,np.dot(Kl,Kinvy))+0.5*np.trace(np.dot(K_inv,Kl))
    Kl=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kl,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kl))
    result=np.array([Kl,Kf])
    return result

def jacobian(input_data,end):
    vector_x= get_time_vector(get_volatility(end))
    sample_y_opt = np.array(get_volatility(end))
    sample_y_trans_opt = sample_y_opt.transpose()
    K_inv=K_inverse_function(input_data,end)
    Kf1=find_K(vector_x,input_data,'der_f1')
    Kf2=find_K(vector_x,input_data,'der_f2')
    Kl1=find_K(vector_x,input_data,'der_l1')
    Kl2=find_K(vector_x,input_data,'der_l2')
    Kn=find_K(vector_x,input_data,'der_n')

    Kf1=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kf1,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kf1))
    Kf2=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kf2,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kf2))
    Kl1=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kl1,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kl1))
    Kl2=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kl2,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kl2))
    Kn= -0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kn,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kn))

    return np.array([Kl1,Kl2,Kf1,Kf2,Kn])



#Moving Average data
def get_moving_average(s):
    length=21
    volatility=get_volatility('all')
    moving_average_vector=[]
    for index in range(length):
        moving_average_vector.append(volatility[0])
    moving_average=[]
    for index in range(s+252):
        moving_average.append(sum(moving_average_vector)/21)
        del moving_average_vector[0]
        moving_average_vector.append(volatility[index])

    print len(moving_average)
    return moving_average

def get_monthly_data(sample,mode):
    if mode=="average":
        volatilty=get_volatility(sample*21)
        months=[]
        for index in range(0,sample*21,21):
            sum=0
            for index2 in range(21):
                sum+=volatilty[index+index2]
            months.append(sum/21)
        return months
    if mode=="spread":
        months=[]
        max_pr,min_pr,diff=get_prices()
        for index in range(0,sample*21,21):
            min=1000000
            max=0
            for index2 in range(21):
                if max_pr[index+index2]>max:
                    max=max_pr[index+index2]
                if min_pr[index+index2]<min:
                    min=min_pr[index+index2]
            months.append(log(log(max)-log(min)))
        return months


def get_prices():
    json_data = open('corn2000.json').read()
    data = json.loads(json_data)
    days = len(data["dataset"]["data"])
    max_prices = np.zeros((1,days))
    min_prices = np.zeros((1,days))
    price_diff = np.zeros((1,days))
    for number in range(0,days):
        max = data["dataset"]["data"][days-number-1][2]
        min = data["dataset"]["data"][days-number-1][3]
        opening = data["dataset"]["data"][days-number-1][1]
        closing = data["dataset"]["data"][days-number-1][4]

        if type(max) != float or type(min)!= float or min ==0  or max==0:
            max_prices[0][number]=max_prices[0][number-1]
            min_prices[0][number]=min_prices[0][number-1]
        else:
            max_prices[0][number]=max
            min_prices[0][number]=min

        if type(opening) != float or type(closing)!= float or opening ==0  or closing==0:
            price_diff[0][number]=price_diff[0][number-1]
        else:
            if abs(log(opening)-log(closing))!=0:
                price_diff[0][number]=log(abs(log(opening)-log(closing)))
            else:
                price_diff[0][number]=price_diff[0][number-1]




    return max_prices[0],min_prices[0],price_diff[0]








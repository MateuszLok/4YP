__author__ = 'Mat'
import json
from math import log
import numpy as np
import GP_functions_2 as gp
import scipy.optimize as optimize
from pyDOE import lhs
from math import exp
import matplotlib.pyplot as plt



"""def get_prices():
    json_data = open('corn2005.json').read()
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

max_pr, min_pr, pr_diff = get_prices()"""

def function_to_optimize(input_data, data_period):
    Dt=gp.get_volatility(data_period)
    c1=0.43
    c2=0.29
    #global log_h_bar
    #log_h_bar = sum(Dt)/len(Dt)-c1
    log_h_bar=input_data[2]


    #log_h_0  at t=0
    log_h=[]
    log_h.append(Dt[0]-c1)
    cov=[]
    cov.append(c2**2)

    rho=input_data[0]
    beta=input_data[1]
    cov.append(beta)

    for index in range(1,len(Dt)):
        log_h_estimate=log_h_bar+rho*(log_h[index-1]-log_h_bar)
        #Q=(rho**index*cov[index-1]+beta**2 *1/252*(rho)
        #Q=beta

        Q=rho**2*cov[index-1]+beta**2*1/252
        cov_estimate=cov[index-1]+Q
        log_h.append(log_h_estimate+(cov_estimate/(cov_estimate+c2**2))*(Dt[index]-c1-log_h_estimate))
        cov.append(cov_estimate-cov_estimate**2/(cov_estimate+c2**2))

    Dt =[elem-c1 for elem in Dt]

    error=0
    for index in range(0,len(Dt)):
        error+=((Dt[index]-log_h[index])**2)/cov[index]
    return error

#Start ______________________

period = 1260

#Latin Hypercube Initlialziation
print 'Starting...'
latin_hypercube_values = lhs(3, samples=10)
latin_hypercube_values=latin_hypercube_values*2

#Optimization part
result=np.zeros((len(latin_hypercube_values),3))
for number in range(0,len(latin_hypercube_values)):
    print number
    wynik= optimize.minimize(function_to_optimize, latin_hypercube_values[number],args=(period,), method='BFGS')
    result[number]=wynik['x']

likelihood=np.zeros((len(latin_hypercube_values),1))
for number in range(0,len(latin_hypercube_values)):
    likelihood[number] = function_to_optimize(result[number],period)
min_index = np.argmin(likelihood)
print likelihood
print min_index
print result[min_index]
hyperparameters = result[min_index]

rho=hyperparameters[0]
beta=hyperparameters[1]
log_h_bar=hyperparameters[2]

Dt=gp.get_volatility(period)
c1=0.43
c2=0.29
log_h=[]
log_h.append(Dt[0]-c1)
cov=[]
cov.append(c2**2)

cov.append(beta)

for index in range(1,len(Dt)):
    log_h_estimate=log_h_bar+rho*(log_h[index-1]-log_h_bar)
    #Q=(rho**index*cov[index-1]+beta**2 *1/252*(rho)
    #Q=beta

    Q=rho**2*cov[index-1]+beta**2*1/252
    cov_estimate=cov[index-1]+Q
    log_h.append(log_h_estimate+(cov_estimate/(cov_estimate+c2**2))*(Dt[index]-c1-log_h_estimate+cov[index-1]*np.random.normal(0,1)))
    cov.append(cov_estimate-cov_estimate**2/(cov_estimate+c2**2))

Dt =[elem-c1 for elem in Dt]



forecast=256

for index in range(period,period+forecast):
    log_h.append(log_h_bar+rho*(log_h[index-1]-log_h_bar))

print 'log h' + str(len(log_h))


h_for_plotting=log_h
log_h=[log_h[index]+0.43 for index in range(period,period+forecast)]

Dt=gp.get_volatility(period+forecast)
error=0
for index in range(0,forecast):
    error+=(log_h[index]-Dt[index+forecast])**2

print error/forecast

Dt=[ele-0.43 for ele in Dt]

plt.plot(gp.get_time_vector(Dt), Dt, 'r.', markersize = 5)
plt.plot(gp.get_time_vector(h_for_plotting), h_for_plotting, 'g.', markersize=5)
plt.axis([0,period+forecast,-7,2])



plt.show()





__author__ = 'Mat'
import numpy as np
import json
from math import log
from math import exp
import scipy.optimize as optimize

k=1
theta=0.2
psi = 0.2
delta= 0.2


def get_prices():
    json_data = open('oil_futures.json').read()
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


#Get all the prices
max_prices,min_prices,diff_prices = get_prices()


#Calculate quadratic variation at 'time'
def get_Q(time,min,max):
    return (log(max[time])-log(min[time]))**2

#Calculate sqrt of difference in quadratic variations at t and t-1
def get_h(time,min,max):
    return (abs(get_Q(time,min,max)-get_Q(time-1,min,max)))**0.5

#Get h from observed prices
h_observed = []
for index in range(1,len(max_prices)):
    h_observed.append(get_h(index,min_prices,max_prices))

print h_observed

#Find parameters that maximise the likelihood
def regarch_likelihood(input_data):
    k=input_data[0]
    theta=input_data[1]
    psi=input_data[2]
    delta=input_data[3]
    sum = 0
    for i in range(2,len(max_prices)):
        D=log(log(max_prices[i-1])-log(min_prices[i-1]))
        X_t_minus_1= (D-0.43-log(h[i-1]))/0.29
        sum-=log(h_observed[i-1])+k*(theta-log(h_observed[i-1]))+psi*X_t_minus_1+delta* (diff_prices[i-1]/h_observed[i-1])
    return sum







h=[]
h.append(get_h(1,min_prices,max_prices))
for index in range(1,len(max_prices)):
    D=log(log(max_prices[index-1])-log(min_prices[index-1]))
    X_t_minus_1= (D-0.43-log(h[index-1]))/0.29
    h_temp=log(h[index-1])+k*(theta-log(h[index-1]))+ psi*(X_t_minus_1)
    h.append(exp(h_temp))

print h


wynik= optimize.minimize(regarch_likelihood,[1,1,1,1])
print wynik






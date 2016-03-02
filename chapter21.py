__author__ = 'Mat'
__author__ = 'Mat'
import json
from math import log
import numpy as np
import GP_functions_2 as gp
import scipy.optimize as optimize
from pyDOE import lhs
from math import exp
import matplotlib.pyplot as plt



def get_prices():
    json_data = open('apple2010.json').read()
    data = json.loads(json_data)
    days = len(data["dataset"]["data"])
    max_prices = np.zeros((1,days))
    min_prices = np.zeros((1,days))
    price_diff = np.zeros((1,days))
    u=np.zeros((1,days))
    closing=[]
    closing.append(data["dataset"]["data"][days-1][4])
    for number in range(1,days):
        max = data["dataset"]["data"][days-number-1][2]
        min = data["dataset"]["data"][days-number-1][3]
        opening = data["dataset"]["data"][days-number-1][1]
        closing.append(data["dataset"]["data"][days-number-1][4])

        if type(max) != float or type(min)!= float or min ==0  or max==0:
            max_prices[0][number]=max_prices[0][number-1]
            min_prices[0][number]=min_prices[0][number-1]
        else:
            max_prices[0][number]=max
            min_prices[0][number]=min

        if type(opening) != float or type(closing[number])!= float or opening ==0  or closing[number]==0:
            price_diff[0][number]=price_diff[0][number-1]
        else:
            if abs(log(opening)-log(closing[number]))!=0:
                price_diff[0][number]=log(closing[number]/closing[number-1])
            else:
                price_diff[0][number]=price_diff[0][number-1]





    return max_prices[0],min_prices[0],price_diff[0][2:]

max_pr, min_pr, pr_diff = get_prices()

u_mean=0
for index in range(1260):
    u_mean+=pr_diff[index]
u_mean=u_mean/1260
print u_mean

sum=0
for index in range(1260):
    sum+=(pr_diff[index]-u_mean)**2

sigma=sum/(1259)
print sigma


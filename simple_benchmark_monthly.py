__author__ = 'Mat'
import json
from math import log
import numpy as np
import GP_functions_2 as gp
import scipy.optimize as optimize
from pyDOE import lhs
from math import exp
import matplotlib.pyplot as plt
from math import fabs
from math import log10


def function_to_optimize(input_data, data_period):
    Dt=gp.get_monthly_data(data_period,'average')
    c1=0.43
    c2=0.29/(21**0.5)
    #c2=0.29
    log_h_bar=input_data[2]
    log_h=[]
    log_h.append(Dt[0]-c1)
    cov=[]
    cov.append(c2**2)

    rho=input_data[0]
    beta=input_data[1]
    error=0
    a=log_h_bar
    b=rho
    c=1
    d=1
    h_e=1
    for index in range(1,len(Dt)):
        log_h_estimate=log_h_bar+rho*(log_h[index-1]-log_h_bar)
        Q=rho**2*cov[index-1]+beta**2*1/12
        cov_estimate=cov[index-1]+Q
        log_h.append(log_h_estimate+(cov_estimate/(cov_estimate+c2**2))*(Dt[index]-c1-log_h_estimate))
        cov.append(cov_estimate-cov_estimate**2/(cov_estimate+c2**2))
        error+=(a+b*log_h[index-1]-b*a-Dt[index]+0.43)**2/(2*0.29**2+2*beta**2*1/12) + log((2*0.29**2+2*beta**2*1/12))*3.14
    return error

#Start ______________________

period = 60

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

#hyperparameters=[  1.00000042e+00,   2.63786285e+03,   4.74198545e+03]

rho=hyperparameters[0]
beta=hyperparameters[1]
log_h_bar=hyperparameters[2]

Dt=gp.get_monthly_data(60,'average')
c1=0.43
c2=0.29/(21**0.5)
#c2=0.29
log_h=[]
log_h.append(Dt[0]-c1)
cov=[]
cov.append(c2**2)

cov.append(beta)

"""for index in range(1,len(Dt)):
    log_h_estimate=log_h_bar+rho*(log_h[index-1]-log_h_bar)
    #Q=(rho**index*cov[index-1]+beta**2 *1/252*(rho)
    #Q=beta

    Q=rho**2*cov[index-1]+beta**2*1/252
    cov_estimate=cov[index-1]+Q
    log_h.append(log_h_estimate+(cov_estimate/(cov_estimate+c2**2))*(Dt[index]-c1-log_h_estimate+cov[index-1]*np.random.normal(0,1)))
    cov.append(cov_estimate-cov_estimate**2/(cov_estimate+c2**2))"""

for index in range(1,len(Dt)):
        log_h_estimate=log_h_bar+rho*(log_h[index-1]-log_h_bar)
        #Q=(rho**index*cov[index-1]+beta**2 *1/252*(rho)
        #Q=beta

        Q=rho**2*cov[index-1]+beta**2*1/12
        cov_estimate=cov[index-1]+Q
        log_h.append(log_h_estimate+(cov_estimate/(cov_estimate+c2**2))*(Dt[index]-c1-log_h_estimate))
        cov.append(cov_estimate-cov_estimate**2/(cov_estimate+c2**2))



#Dt =[elem-c1 for elem in Dt]



forecast=12

for index in range(period,period+forecast):
    log_h.append(log_h_bar+rho*(log_h[index-1]-log_h_bar))
    #cov.append(rho**(2*index-period)*cov[index-1]+beta**2*1/252*(1-rho**(2*index-period))/(1-rho**2))
    Q=rho**2*cov[index-1]+beta**2*1/12
    cov_estimate=cov[index-1]+Q
    cov.append(cov_estimate)

print 'log h' + str(len(log_h))

del cov[0]
h_for_plotting=log_h
log_h=[log_h[index]+0.43 for index in range(0,period+forecast)]

Dt=gp.get_monthly_data(period+forecast,'average')
error=0
for index in range(period,period+forecast):
    error+=(log_h[index]-Dt[index])**2

print error/forecast
cov_2=[1.96*(fabs(cov[index]))**0.5 for index in range(len(cov))]

#Dt=[ele-0.43 for ele in Dt]

print cov[period:period+10]


plt.plot(gp.get_time_vector(Dt), Dt, 'r.', markersize = 6)
plt.plot(gp.get_time_vector(log_h), log_h, 'g.', markersize=8)
plt.errorbar(gp.get_time_vector(log_h), log_h,xerr=0, yerr=cov_2, alpha=0.4, capsize=0, linewidth=1.5, fmt='o')
plt.axis([0,period+forecast,-5.5,-2.5])
plt.axvline(x=60,linewidth=2, color='k')
plt.title('Monthly Volatility Forecast Using Stochastic Volatility Model')
plt.xlabel("Trading month")
plt.ylabel("Volatility approximation")

H_periods=[1,3,6,12]
sum_errors=[0,0,0,0]
for index1 in range(len(H_periods)):
    for index in range(60,60+H_periods[index1]):
        print Dt[index]-log_h[index]
        sum_errors[index1]+=fabs(Dt[index]-log_h[index])**2

print sum_errors

error=[0,0,0,0]
for index1 in range(len(H_periods)):
    error[index1] = (sum_errors[index1]/H_periods[index1])**0.5

print error

sum_nll=[0,0,0,0]


for index1 in range(len(H_periods)):
    for index in range(period,period+H_periods[index1]):
        sum_nll[index1]+=0.5*(log(2*3.14)+log(fabs(cov[index]))+\
                         (Dt[index]-log_h[index])**2*\
                         cov[index]**-1)

print sum_nll


#plt.savefig("Soya2005MonthlySV.png", dpi=200)
plt.show()





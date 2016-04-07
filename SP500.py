import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import GP_functions_2 as gp
from pyDOE import lhs
from math import fabs
from math import log

# ----------------------------------------------------------------------------
#Data input
sample_size =1260
jitter=0
volatility_observed = gp.get_volatility(sample_size)
time_vector = gp.get_time_vector(volatility_observed)
print time_vector
print volatility_observed
new_x = np.array([0.2])

average=sum(volatility_observed)/len(volatility_observed)
print average
for index in range(0,len(volatility_observed)):
    volatility_observed[index]-=average

print volatility_observed

f1=open('hyperparameters','w')

"""#Latin Hypercube Initlialziation
print 'Starting...'
latin_hypercube_values = lhs(5, samples=1)
latin_hypercube_values=latin_hypercube_values*10
#Optimization part
result=np.zeros((len(latin_hypercube_values),4))
for number in range(0,len(latin_hypercube_values)):
    print number
    wynik= optimize.minimize(gp.function_to_minimize_volatility, latin_hypercube_values[number], args=(sample_size,), method='BFGS')
    result[number]=wynik['x']
    r=str(result[number])
    f1.write("%s \n" %r)
    print gp.function_to_minimize_volatility(result[number],sample_size)

likelihood=np.zeros((len(latin_hypercube_values),1))
for number in range(0,len(latin_hypercube_values)):
    likelihood[number] = gp.function_to_minimize_volatility(result[number],sample_size)
min_index = np.argmin(likelihood)
print likelihood
print min_index
print result[min_index]
hyperparameters = result[min_index]"""

#hyperparameters=[3.85032343,19.29309899,-0.9074395,1.62907228,-0.68368862]
#hyperparameters=[-1.293944,8.73993713,-0.7438694,1.536927,-0.636638]
#hyperparameters=[ 4.95706937, -2.84268849,  0.73411089, -1.01649983, -1.04456131]
#hyperparameters=[21.02283188,  12.656023,    0.54348225]
#hyperparameters = [  4.45066253,  10.61178282,  -0.66725735,   0.49673853]
#NG05 periodic
hyperparameters=[   6.85143766,  364.53201194,    0.46546251,    4.09265934,    0.50855517]
f1.close()


#Finding values of K matrices for new values of x
K = gp.find_K(time_vector,hyperparameters,'normal',jitter)
K_2stars_estimate = gp.find_K_2stars(new_x,hyperparameters,'normal',jitter)
K_inv = np.linalg.inv(K)

#--------------------------------

#Vector of new x values
new_values = np.arange(0,sample_size+252,0.20001)

#Initialise matrices to store estimated values of y(volatility) and variance
estimated_values_y = []
estimated_variance_y = []

#Initialise variable to store size of the variance and y vector
estimated_variance_size = 0

#Find y and variance for all 'new values'
for number in new_values:
    K_star_estimate = gp.find_K_star(time_vector,number,hyperparameters,'normal',jitter)
    X_estimate = np.dot(K_star_estimate,K_inv)
    estimated_values_y.append((np.dot(X_estimate,volatility_observed).tolist()))
    K_star_trans_estimate = K_star_estimate.transpose()
    temp = (K_2stars_estimate-np.dot(K_star_estimate,np.dot(K_inv,K_star_trans_estimate)))
    #To list to get rid of matrix representation
    estimated_variance_y.append(temp.tolist())
    estimated_variance_y[estimated_variance_size]=1.96*(fabs(temp)**0.5)
    estimated_variance_size+=1


new_estimated_values_y = []
for number in range(0,len(estimated_values_y)):
    new_estimated_values_y.append(estimated_values_y[number][0])

new_estimated_variance_y = estimated_variance_y

new_estimated_variance_y_1 = []
new_estimated_variance_y_2 = []
for number in range(0,len(new_estimated_variance_y)):
    new_estimated_variance_y_1.append(new_estimated_values_y[number]+new_estimated_variance_y[number])
    new_estimated_variance_y_2.append(new_estimated_values_y[number]-new_estimated_variance_y[number])


moving_avg=gp.get_moving_average(sample_size)-average

#Plotting one new value
all_volatility=gp.get_volatility('all')-average
plt.fill_between(new_values, new_estimated_variance_y_2,new_estimated_variance_y_1,alpha=0.4)
plt.plot(new_values,new_estimated_values_y, 'g-', linewidth=3)
plt.plot(time_vector, volatility_observed, 'r.', markersize = 5)
#plt.plot(gp.get_time_vector(monthly),monthly, 'r.', markersize = 5)
#plt.plot(gp.get_time_vector(moving_avg),moving_avg,'b-',)
plt.plot(gp.get_time_vector(gp.get_volatility('all')),all_volatility, 'r.', markersize = 5)
#plt.axis([min(sample_vector)-0.5, max(sample_vector)+0.5, min(sample_y)-0.5, max(sample_y)+0.5])
plt.axis([0,len(time_vector)+252,-2.2,2.2])
plt.axvline(x=sample_size,linewidth=3, color='k')
plt.xlabel("Trading day")
plt.ylabel("Volatility Approximation")


#Error calculation - RMSE
forecast_period = 252
forecast_volatility = gp.get_volatility(sample_size+forecast_period)

forecast_values = np.arange(sample_size,sample_size+forecast_period,1)

#Initialise matrices to store estimated values of y(volatility) and variance
estimated_values_y_forecast = []

#Find y and variance for all 'new values'
for number in forecast_values:
    K_star_estimate = gp.find_K_star(time_vector,number,hyperparameters,'normal',jitter)
    X_estimate = np.dot(K_star_estimate,K_inv)
    estimated_values_y_forecast.append((np.dot(X_estimate,volatility_observed).tolist()))

new_estimated_values_y_forecast = []
for number in range(0,len(estimated_values_y_forecast)):
    new_estimated_values_y_forecast.append(estimated_values_y_forecast[number][0])

print new_estimated_values_y_forecast

for index in range(0,len(forecast_volatility)):
    forecast_volatility[index]-=average


sum_errors = [0,0,0,0]

x=sample_size+forecast_period

H_periods=[1,5,21,252]
for index1 in range(len(H_periods)):
    for index in range(sample_size,sample_size+H_periods[index1]):
        sum_errors[index1]+=fabs(forecast_volatility[index]-new_estimated_values_y_forecast[index-sample_size])**2

print sum_errors
error=[0,0,0,0]
for index1 in range(len(H_periods)):
    error[index1] = (sum_errors[index1]/H_periods[index1])**0.5

print error


#NLL calculation
#Initialise matrices to store estimated values of y(volatility) and variance
estimated_variance_y_forecast = []

#Find y and variance for all 'new values'
for number in forecast_values:
    K_star_estimate = gp.find_K_star(time_vector,number,hyperparameters,'normal',jitter)
    K_star_trans_estimate = K_star_estimate.transpose()
    temp = (K_2stars_estimate-np.dot(K_star_estimate,np.dot(K_inv,K_star_trans_estimate)))
    #To list to get rid of matrix representation
    estimated_variance_y_forecast.append(temp.tolist())


new_estimated_variance_y_forecast = []
for number in range(0,len(estimated_variance_y_forecast)):
    new_estimated_variance_y_forecast.append(estimated_variance_y_forecast[number][0][0])

print new_estimated_variance_y_forecast

print len(new_estimated_variance_y_forecast)
print len(new_estimated_values_y_forecast)
print len(forecast_volatility)


sum_nll=[0,0,0,0]
for index1 in range(len(H_periods)):
    for index in range(sample_size,sample_size+H_periods[index1]):
        sum_nll[index1]+=0.5*(log(2*3.14)+log(fabs(new_estimated_variance_y_forecast[index-sample_size]))+\
                         (forecast_volatility[index]-new_estimated_values_y_forecast[index-sample_size])**2*\
                         new_estimated_variance_y_forecast[index-sample_size]**-1)
print sum_nll

plt.savefig("naturalgas2005priors.png", dpi=200)
plt.show()

f=open('outputs','w')
for item in hyperparameters:
    item=str(item)
    f.write("%s \n" %item)

for item in sum_errors:
    item=str(item)
    f.write("%s \n" %item)

for item in sum_nll:
    item=str(item)
    f.write("%s \n" %item)

f.close()







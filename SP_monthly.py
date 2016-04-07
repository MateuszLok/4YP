__author__ = 'Mat'
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import GP_functions_2 as gp
from pyDOE import lhs
from math import fabs
from math import log

# ----------------------------------------------------------------------------
#Data input
#N of months
sample_size =60
jitter=0
monthly_volatility_observed = gp.get_monthly_data(sample_size,'average')
time_vector = gp.get_time_vector(monthly_volatility_observed)
print time_vector
print monthly_volatility_observed
new_x = np.array([0.2])

average=sum(monthly_volatility_observed)/len(monthly_volatility_observed)
print average
for index in range(0,len(monthly_volatility_observed)):
    monthly_volatility_observed[index]-=average

print monthly_volatility_observed

f1=open('hyperparameters','w')

#Latin Hypercube Initlialziation
print 'Starting...'
latin_hypercube_values = lhs(3, samples=3)
latin_hypercube_values=latin_hypercube_values*5
print latin_hypercube_values
#Optimization part
result=np.zeros((len(latin_hypercube_values),3))
for number in range(0,len(latin_hypercube_values)):
    print number
    wynik= optimize.minimize(gp.function_to_minimize_volatility, latin_hypercube_values[number], args=(sample_size,), method='BFGS')
    result[number]=wynik['x']
    r=str(result[number])
    f1.write("%s \n" %r)
    print gp.function_to_minimize_spread(result[number],sample_size)

likelihood=np.zeros((len(latin_hypercube_values),1))
for number in range(0,len(latin_hypercube_values)):
    likelihood[number] = gp.function_to_minimize_spread(result[number],sample_size)
min_index = np.argmin(likelihood)
print likelihood
print min_index
print result[min_index]
hyperparameters = result[min_index]

#hyperparameters=[3.85,19.29,-0.907,1.629,-0.68]
#hyperparameters=[4.19517686,7.02150056,0.1]

f1.close()


#Finding values of K matrices for new values of x
K = gp.find_K(time_vector,hyperparameters,'normal',jitter)
K_2stars_estimate = gp.find_K_2stars(new_x,hyperparameters,'normal',jitter)
K_inv = np.linalg.inv(K)

#--------------------------------

#Vector of new x values
new_values = np.arange(0,sample_size+12,0.020001)

#Initialise matrices to store estimated values of y(volatility) and variance
estimated_values_y = []
estimated_variance_y = []

#Initialise variable to store size of the variance and y vector
estimated_variance_size = 0

#Find y and variance for all 'new values'
for number in new_values:
    K_star_estimate = gp.find_K_star(time_vector,number,hyperparameters,'normal',jitter)

    X_estimate = np.dot(K_star_estimate,K_inv)
    estimated_values_y.append((np.dot(X_estimate,monthly_volatility_observed).tolist()))
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




#Plotting one new value
all_monthly_volatility=gp.get_monthly_data(sample_size+12,'average')
print all_monthly_volatility
all_monthly_volatility=[all_monthly_volatility[index]-average for index in range(len(all_monthly_volatility))]

print all_monthly_volatility
plt.fill_between(new_values, new_estimated_variance_y_2,new_estimated_variance_y_1,alpha=0.3)
plt.plot(new_values,new_estimated_values_y, 'g-')
#plt.plot(time_vector, volatility_observed, 'r.', markersize = 5)
#plt.plot(gp.get_time_vector(monthly),monthly, 'r.', markersize = 5)
plt.plot(gp.get_time_vector(all_monthly_volatility),all_monthly_volatility, 'r.', markersize = 6)
#plt.axis([min(sample_vector)-0.5, max(sample_vector)+0.5, min(sample_y)-0.5, max(sample_y)+0.5])
plt.axis([0,len(time_vector)+12,-2.2,2.2])
plt.xlabel("Trading month")
plt.ylabel("Volatility approximation")
#plt.axis([0,80,-2.2,2.2])
plt.axvline(x=sample_size,linewidth=2, color='k')
plt.title('Monthly Volatility forecast using GP with Matern 3/2 covariance function')


#Error calculation - RMSE
forecast_period = 12
forecast_volatility = gp.get_monthly_data(sample_size+forecast_period,'average')

forecast_values = np.arange(sample_size,sample_size+forecast_period,1)

#Initialise matrices to store estimated values of y(volatility) and variance
estimated_values_y_forecast = []

#Find y and variance for all 'new values'
for number in forecast_values:
    K_star_estimate = gp.find_K_star(time_vector,number,hyperparameters,'normal',jitter)
    X_estimate = np.dot(K_star_estimate,K_inv)
    estimated_values_y_forecast.append((np.dot(X_estimate,monthly_volatility_observed).tolist()))

new_estimated_values_y_forecast = []
for number in range(0,len(estimated_values_y_forecast)):
    new_estimated_values_y_forecast.append(estimated_values_y_forecast[number][0])

print new_estimated_values_y_forecast

for index in range(0,len(forecast_volatility)):
    forecast_volatility[index]-=average


sum_errors = [0,0,0,0]

x=sample_size+forecast_period

H_periods=[1,3,6,12]
for index1 in range(len(H_periods)):
    for index in range(sample_size,sample_size+H_periods[index1]):
        print forecast_volatility[index]
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


print len(new_estimated_variance_y_forecast)
print len(new_estimated_values_y_forecast)
print len(forecast_volatility)


sum_nll=[0,0,0,0]
for index1 in range(len(H_periods)):
    for index in range(sample_size,sample_size+H_periods[index1]):
        sum_nll[index1]+=0.5*(log(2*3.14)+log(fabs(new_estimated_variance_y_forecast[index-sample_size]))+\
                         (forecast_volatility[index]-new_estimated_values_y_forecast[index-sample_size])**2)*\
                         new_estimated_variance_y_forecast[index-sample_size]**-1
print sum_nll
#plt.savefig("Soya2005MonthlyGP.png")
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







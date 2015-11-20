__author__ = 'Mat'


from math import exp
from math import log10
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import GP_functions as gp
from pyDOE import lhs
from math import fabs

# ----------------------------------------------------------------------------


#Example of x and y vectors
sample_vector = gp.get_x_values_2d()
sample_y = gp.get_y_values_2d()
new_x = np.array([0.2])
print 'Starting...'
latin_hypercube_values = lhs(3, samples=20)
latin_hypercube_values=latin_hypercube_values

#Optimization part
result=np.zeros((len(latin_hypercube_values),3))
for number in range(0,len(latin_hypercube_values)):
    wynik= optimize.minimize(gp.function_to_minimize, latin_hypercube_values[number],method='BFGS')
    result[number]=wynik['x']

likelihood=np.zeros((len(latin_hypercube_values),1))
for number in range(0,len(latin_hypercube_values)):
    likelihood[number] = gp.function_to_minimize(result[number])
min_index = np.argmin(likelihood)
print result[min_index]
sigma_f = result[min_index][0]
sigma_n = result[min_index][1]
length = result[min_index][2]

"""sigma_f = 1.27
sigma_n = 0.3
length = 1"""



#Finding values of K matrices for new values of x
K = gp.find_K(sample_vector,sigma_f,sigma_n,length)
K_star = gp.find_K_star(sample_vector,new_x,sigma_f,sigma_n,length)
K_2stars = gp.find_K_2stars(new_x,sigma_f,sigma_n,length)

#Find y*
K_inv = np.linalg.inv(K)
X = np.dot(K_star,K_inv)
y_star = np.dot(X, sample_y)

#Find variance of y*
K_star_trans = K_star.transpose()
y_star_var = K_2stars-np.dot(K_star,np.dot(K_inv,K_star_trans))

#--------------------------------

#Vector of new x values
new_values = np.arange(-5,5,0.001)
estimated_values_y = []
estimated_variance_y = []
K_2stars_estimate = gp.find_K_2stars(new_values[1],sigma_f,sigma_n,length)
estimated_variance_size = 0
for number in new_values:
    K_star_estimate = gp.find_K_star(sample_vector,number,sigma_f,sigma_n,length)
    X_estimate = np.dot(K_star_estimate,K_inv)
    # Without conversion to float we have a list of arrays
    estimated_values_y.append((np.dot(X_estimate,sample_y).tolist()))
    K_star_trans_estimate = K_star_estimate.transpose()
    temp = (K_2stars_estimate-np.dot(K_star_estimate,np.dot(K_inv,K_star_trans_estimate)))
    estimated_variance_y.append(temp.tolist())
    estimated_variance_y[estimated_variance_size]=1.96*(fabs(temp)**0.5)
    estimated_variance_size+=1


new_estimated_values_y = []
for number in range(0,len(estimated_values_y)):
    new_estimated_values_y.append(estimated_values_y[number][0][0])

new_estimated_variance_y = estimated_variance_y

print new_estimated_values_y

new_estimated_variance_y_1 = []
new_estimated_variance_y_2 = []
print new_estimated_variance_y
for number in range(0,len(new_estimated_variance_y)):
    new_estimated_variance_y_1.append(new_estimated_values_y[number]+new_estimated_variance_y[number])
    new_estimated_variance_y_2.append(new_estimated_values_y[number]-new_estimated_variance_y[number])
print new_estimated_variance_y_1
print new_estimated_variance_y_2
#Plotting estimated curve
plt.fill_between(new_values, new_estimated_variance_y_2,new_estimated_variance_y_1,alpha=0.5)
plt.plot(new_values,new_estimated_values_y, 'b-')
#pl.fill(np.concatenate([x, x[::-1]]),np.concatenate([y_pred - 1.9600 * sigma,(y_pred + 1.9600 * sigma)[::-1]]),alpha=.5, fc='b', ec='None', label='95% confidence interval')
#Plotting one new value
plt.plot(sample_vector, sample_y, 'r.', markersize = 10)
#plt.axis([min(sample_vector)-0.5, max(sample_vector)+0.5, min(sample_y)-0.5, max(sample_y)+0.5])
plt.axis([-4,4,-4,4])
#plt.plot([0.2],[y_star], 'go')
plt.show()


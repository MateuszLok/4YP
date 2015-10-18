__author__ = 'Mat'


from math import exp
from math import log10
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import GP_functions as gp
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------------------------------------

#Get Input Data
sample_x = gp.get_x_values()
sample_y = gp.get_y_values()

#Define the new point, for which there is no data available
new_x = np.array([0.2,1])

#----Optimization part

#result = optimize.minimize(gp.function_to_minimize, [3,0.5,3])

#wynik = result['x']
#print wynik
"""sigma_f = wynik[0]     1.935
sigma_n = wynik[1]
length = wynik[2]"""

sigma_f = 1.27
sigma_n = 0.3
length = 1
print length


#-------

#Finding values of K, K* and K** matrices
K = gp.find_K(sample_x,sigma_f,sigma_n,length)
K_star = gp.find_K_star(sample_x,new_x,sigma_f,sigma_n,length)
K_2stars = gp.find_K_2stars(new_x,sigma_f,sigma_n,length)

print K

#Find y*
K_inv = np.linalg.inv(K)
X = np.dot(K_star,K_inv)
y_star = np.dot(X, sample_y)

#Find variance of y*
K_star_trans = K_star.transpose()
y_star_var = K_2stars-np.dot(K_star,np.dot(K_inv,K_star_trans))

#--------------------------------

#----Vector of new x values
#Define the x values for which y values are to be found
new_values = np.arange(-2.5,1.5,0.2)
estimated_values_y = []
estimated_variance_y = []
for number in new_values:
    for number2 in new_values:
        K_star_estimate = gp.find_K_star(sample_x,np.array([number,number2]),sigma_f,sigma_n,length)
        K_2stars_estimate = gp.find_K_2stars(np.array([number,number2]),sigma_f,sigma_n,length)
        X_estimate = np.dot(K_star_estimate,K_inv)
        # Without conversion to float we have a list of arrays
        estimated_values_y.append((np.dot(X_estimate,sample_y).tolist()))
        K_star_trans_estimate = K_star_estimate.transpose()
        estimated_variance_y.append(((1.96*(K_2stars_estimate-np.dot(K_star_estimate,np.dot(K_inv,K_star_trans_estimate))**0.5))).tolist())
new_new_values1 = []
for number in range(0,len(new_values)):
    for number2 in range(0,len(new_values)):
        new_new_values1.append(new_values[number])

new_new_values2 = []
for number in range(0,len(new_values)):
    for number2 in range(0,len(new_values)):
        new_new_values2.append(new_values[number2])



new_estimated_values_y = []
for number in range(0,len(estimated_values_y)):
    new_estimated_values_y.append(estimated_values_y[number][0][0])

new_estimated_variance_y = []
for number in range(0,len(estimated_variance_y)):
    new_estimated_variance_y.append(estimated_variance_y[number][0][0])

sample_vector=sample_x.tolist()


#Convert input to 2 lists
sample_vector_x1 = []
sample_vector_x2 = []
for number in range(0,len(sample_vector)):
    sample_vector_x1.append(sample_vector[number][0])

for number in range(0,len(sample_vector)):
    sample_vector_x2.append(sample_vector[number][1])




fig = plt.figure(dpi=100)
ax = fig.add_subplot(111, projection='3d')



#data
fx = new_estimated_values_y
fy = new_new_values1
fz = new_new_values2

#error data
xerror = new_estimated_variance_y
yerror = 0
zerror = 0

#plot points
ax.plot(fx, fy, fz, linestyle="None")
ax.plot(sample_y,sample_vector_x1,sample_vector_x2,'g', linestyle="None", marker="o")

#plot errorbars
for i in np.arange(0, len(fx)):
    ax.plot([fx[i]+xerror[i], fx[i]-xerror[i]], [fy[i], fy[i]], [fz[i], fz[i]], 'r', alpha=0.2, marker="_",)


#configure axes
ax.set_xlim3d(-5, 5)
ax.set_ylim3d(-5, 5)
ax.set_zlim3d(-5, 5)

plt.show()


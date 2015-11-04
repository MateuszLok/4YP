__author__ = 'Mat'
__author__ = 'Mat'


from math import exp
from math import log10
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import GP_functions as gp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pyDOE import lhs

# ----------------------------------------------------------------------------

#Get Input Data
sample_x = gp.get_x_values_3d()
sample_y = gp.get_y_values_3d()

#Define the new point, for which there is no data available
new_x = np.array([0.2,1])

#----Optimization part

#Get hypercube parameters:
print lhs(4,samples=10)
#result stores the result op optimization on function to minimie in 3d, with starting location given
result = optimize.minimize(gp.function_to_minimize_3d, [3.0,1.5,2.0,1])

#We're only interested in the final value of x, whereas results stores also other info
wynik = result['x']

#Assign parameters basing on optimisation
sigma_f = wynik[0]
sigma_n = wynik[1]
length = np.array(([wynik[2],0],[0,wynik[3]]),dtype=float)



#----Plotting optimization parameters



#-------

#Finding values of K, K* and K** matrices (NOTE: just one new value of x)
K = gp.find_K(sample_x,sigma_f,sigma_n,length)
K_star = gp.find_K_star(sample_x,new_x,sigma_f,sigma_n,length)
K_2stars = gp.find_K_2stars(new_x,sigma_f,sigma_n,length)


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
new_values = np.arange(-3,3,0.15)

#Initialise lists for values and variance
estimated_values_y = []
estimated_variance_y = []
for number in range(len(new_values)):
    for number2 in range(len(new_values)):
        #Equations from the the first article (that was sent in September)
        K_star_estimate = gp.find_K_star(sample_x,np.array([new_values[number],new_values[number2]]),sigma_f,sigma_n,length)
        K_2stars_estimate = gp.find_K_2stars(np.array([new_values[number],new_values[number2]]),sigma_f,sigma_n,length)
        X_estimate = np.dot(K_star_estimate,K_inv)
        #Add new value of y to the list
        estimated_values_y.append(((np.dot(X_estimate,sample_y).tolist())))
        K_star_trans_estimate = K_star_estimate.transpose()
        #Add new value of variance to the list
        estimated_variance_y.append(((1.96*(K_2stars_estimate-np.dot(K_star_estimate,np.dot(K_inv,K_star_trans_estimate)\
                                                                     ))**0.5)).tolist())



new_estimated_values_y = np.zeros((len(new_values),len(new_values)))
var = 0
for number in range(0,len(new_values)):
    for number2 in range(0,len(new_values)):
        new_estimated_values_y[number][number2]=(estimated_values_y[var][0][0])
        var+=1


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






fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(new_values,new_values)
surf = ax.plot_surface(Y, X, new_estimated_values_y, rstride=1, cstride=1, cmap='Blues',
        linewidth=0, antialiased=True, shade=0)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
ax.scatter(sample_vector_x1,sample_vector_x2,sample_y)

ax.set_xlim3d(-5, 5)
ax.set_ylim3d(-5, 5)
ax.set_zlim3d(-2, 1)


plt.show()

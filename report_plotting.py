__author__ = 'Mat'
import numpy as np
from math import exp
from math import fabs
import matplotlib.pyplot as plt
import random
from matplotlib import rc

def get_y(type,x):
    if type=="1/2":
        y=[]
        for index in range(len(x)):

            y.append(-0.0005*x[index]**5+0.007*x[index]**4-0.02*x[index]**3+0.2*x[index]**2-0.2*x[index]+1+random.uniform(-1.5,1.5))

        #y=[1.2,1,1.4,1.6,1,1.2,0.8,0,-0.5,-0.2,0.5,0.3,-0.5,-1.1,-1.3,-0.9,-1.4,0.2,0.8,1.0,0,-1.2,-0.9,-0.7,-1.5,-1.9,-2.0,-1.5,-1,-1.2,-2.2,-1,-1.5,-0.6,0]
        y=[2.91091 ,0.0400169 ,2.26258 ,0.661846 ,0.93467 ,0.40812 ,0.933753 ,0.255352 ,1.0552 ,-0.177719 ,1.29351 ,1.38194 ,0.812621 ,0.909573 ,0.481272 ,1.25105 ,-0.656218 ,0.999675 ,1.85086 ,0.930565 ,0.475187 ,-0.225091 ,1.04833 ,0.331311 ,1.01059 ,1.02685 ,1.05603 ,0.349735 ,2.24807 ,0.471922 ,-0.339594 ,0.671002 ,0.918652 ,0.0812971 ,1.20846 ,2.56776 ,1.04089 ,0.906131 ,2.89294 ,1.42461 ,-0.571275 ,1.37615 ,0.905347 ,-0.650028 ,1.00143 ,0.0375789 ,0.442877 ,0.711022 ,0.88284 ,2.98788 ,0.800439 ,1.59484 ,0.583835 ,0.920325 ,-0.587402 ,2.47291 ,0.695034 ,0.115196 ,0.938816 ,2.18102 ,-0.267953 ,2.81121 ,-0.25819 ,0.217325 ,0.338304 ,0.928325 ,0.444369 ,0.436379 ,1.00808 ,0.857756 ,1.33481 ,0.43116 ,-0.770917 ,0.673067 ,1.02255 ,-0.0265277 ,-0.572177 ,1.12311 ,0.0934483 ,-0.394418 ,0.464209 ,0.46802 ,1.03472 ,0.515159 ,3.72241 ,0.433024 ,1.84549 ,0.684395 ,0.614179 ,0.551591 ,0.0113022 ,0.945755 ,0.61723 ,0.908552 ,0.878721 ,0.870151 ,0.943732 ,1.15775 ,0.798082 ,0.884714 ,-0.186931 ,1.0415 ,-0.2883 ,-0.806018 ,1.83195 ,-0.12773 ,1.05457 ,-0.0664085 ,0.752944 ,-0.00527358 ,1.67976 ,0.621616 ,0.868345 ,0.327971 ,-0.198828 ,0.604936 ,0.774784 ,0.685721 ,0.980748 ,0.707524 ,2.47538 ,1.19585 ,-0.379875 ,2.80223 ,2.78807 ,0.7616 ,1.01668 ,1.25478 ,0.825436 ,0.683461 ,1.45078 ,1.12636 ,0.733037 ,1.02298 ,2.16267 ,0.488422 ,0.914342 ,0.844663 ,-0.786815 ,0.845178 ,0.812418 ,-0.424278 ,0.703574 ,1.11798 ,0.354338 ,1.02635 ,1.8308 ,0.434483 ,0.984232 ,-1.1188 ,0.549388 ,0.582517 ,2.41656 ,0.647056 ,0.454099 ,1.7547 ,0.864764 ,0.958793 ,1.69889 ,0.810964 ,0.964545 ,0.734844 ,0.59286 ,1.16696 ,-0.735999 ,1.25715 ,0.395698 ,0.570518 ,-0.0404507 ,1.00447 ,0.556889 ,-0.676827 ,0.832833 ,1.0302 ,0.0666927 ,0.824444 ,1.94374 ,0.267892 ,0.575856 ,1.33219 ,0.913669 ,-0.313984 ,2.08421 ,0.804862 ,-0.451787 ,1.63927 ,0.428931 ,-0.631704 ,0.375956 ,0.765995 ,-0.0820582 ,1.15843 ,0.907063 ,0.28013 ,0.447261 ,-0.76229 ,2.63869 ,3.47326 ,0.452494 ,1.46717]
        return y
    if type=='3/2':

        y=[1.2,1,1.4,1.6,1,1.2,0.8,0,-0.5,-0.2,0.5,0.3,-0.5,-1.1,-1.3,-0.9,-1.4,0.2,0.8,1.0,0,-1.2,-0.9,-0.7,-1.5,-1.9,-2.0,-1.5,-1,-1.2,-2.2,-1,-1.5,-0.6,0]
        y=[elem-2 for elem in y]
        return y
    if type=='5/2':
        y=[1.2,1,1.4,1.6,1,1.2,0.8,0,-0.5,-0.2,0.5,0.3,-0.5,-1.1,-1.3,-0.9,-1.4,0.2,0.8,1.0,0,-1.2,-0.9,-0.7,-1.5,-1.9,-2.0,-1.5,-1,-1.2,-2.2,-1,-1.5,-0.6,0]
        y=[elem+1 for elem in y]
        return y
    if type=='inf':
        y=[]
        for index in range(len(x)):
            y.append(0.0005*x[index]**5-0.007*x[index]**4+0.02*x[index]**3-0.2*x[index]**2+0.2*x[index]+2+random.uniform(-1.5,1.5))
        #y=[1.2,1,1.4,1.6,1,1.2,0.8,0,-0.5,-0.2,0.5,0.3,-0.5,-1.1,-1.3,-0.9,-1.4,0.2,0.8,1.0,0,-1.2,-0.9,-0.7,-1.5,-1.9,-2.0,-1.5,-1,-1.2,-2.2,-1,-1.5,-0.6,0]
        #y=[elem-1 for elem in y]
        return y




def calculate_k(x1,x2,theta,type):
    difference = x1-x2
    l=(theta[0])
    sigma_f=(theta[1])
    sigma_n=(theta[2])
    if type=="1/2":
        intermediate = -fabs(difference) * (l **-1)
        if x1==x2:
            return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2
        else:
            return (sigma_f ** 2) * exp(intermediate)
    elif type=="3/2":
        intermediate = -(3)**(0.5)*fabs(difference)*(l **-1)
        if x1==x2:
            return (sigma_f ** 2)*(1+(3)**(0.5)*fabs(difference)*(l **-1)) * exp(intermediate) + sigma_n ** 2
        else:
            return (sigma_f ** 2)*(1+(3)**(0.5)*fabs(difference)*(l **-1)) * exp(intermediate)
    elif type=="5/2":
        intermediate = -(5)**(0.5)*fabs(difference)*(l **-1)
        if x1==x2:
            return (sigma_f ** 2)*(1+(5)**(0.5)*fabs(difference)*(l **-1)+5*(difference)**2/(3*l**2)) * exp(intermediate) + sigma_n ** 2
        else:
            return (sigma_f ** 2)*(1+(5)**(0.5)*fabs(difference)*(l **-1)+5*(difference)**2/(3*l**2)) * exp(intermediate)

    elif type=="inf":
        intermediate = -(difference)**2 * (2 * l ** 2) ** -1
        if x1==x2:
            return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2
        else:
            return (sigma_f ** 2) * exp(intermediate)



#Evaluates matrix K - covariance matrix
#Vector format
def find_K(vector_X, theta,type):
    outcome = np.zeros((len(vector_X),len(vector_X)))
    for i in range(0, len(vector_X)):
        for j in range(i,len(vector_X)):
            temp = (calculate_k(vector_X[i],vector_X[j],theta,type))
            outcome[i][j] = temp
            outcome[j][i] = temp

    return outcome


#Evaluates matrix K*
def find_K_star(vector,x_star,theta,type):
    outcome = np.zeros((1, len(vector)))
    for i in range(0,len(vector)):
        outcome[0][i] = calculate_k(x_star,vector[i],theta,type)
    return outcome

#Evaluates matrix K**
def find_K_2stars(x,theta,type):
    return calculate_k(x,x,theta,type)


"""y=[]
for i in range(50):
    y.append(random.uniform(-1,1))

print y"""

x=[]
add=0
extra=0.05
for i in range(200):
    x.append(add)
    add+=extra

x=[0.0502943 ,0.818781 ,-0.0828292 ,-1.06369 ,0.285215 ,-0.576611 ,1.79343 ,1.21907 ,0.366946 ,0.587012 ,-0.221551 ,0.196883 ,-1.07978 ,-0.861388 ,-2.00621 ,-0.23149 ,-2.25752 ,-0.366288 ,-0.132441 ,-0.873036 ,0.552013 ,0.861191 ,0.293074 ,-1.98738 ,-0.420445 ,-0.274209 ,0.297499 ,1.46295 ,0.148114 ,-1.18178 ,-3.0628 ,-0.737529 ,-0.907413 ,-2.12519 ,0.362287 ,-0.0376308 ,-0.903702 ,-0.834667 ,0.0552622 ,0.35186 ,2.1198 ,0.331491 ,-0.34367 ,0.690317 ,-0.385905 ,1.15076 ,1.46938 ,-1.0933 ,-1.85058 ,0.0173075 ,-1.63422 ,-0.187847 ,-1.39388 ,-0.797962 ,0.699852 ,0.0953302 ,-0.745593 ,0.758941 ,-0.403746 ,-0.0896275 ,0.822244 ,0.0112439 ,0.598184 ,1.87312 ,-1.42948 ,0.438708 ,2.20404 ,0.538007 ,-0.427475 ,0.257994 ,0.332817 ,1.25239 ,0.94761 ,-0.485612 ,0.445611 ,1.88504 ,0.648665 ,-0.263193 ,0.570396 ,0.916452 ,-1.29047 ,1.72256 ,-0.841186 ,-1.34021 ,2.87238 ,-1.27489 ,-0.147025 ,-1.70766 ,1.26843 ,1.30994 ,0.800823 ,-0.30941 ,-1.15066 ,-0.954266 ,-0.985951 ,0.465218 ,-0.742657 ,0.239846 ,-0.781516 ,-1.90787 ,2.14538 ,0.284408 ,0.615827 ,0.971961 ,0.167777 ,0.873798 ,0.223809 ,0.847595 ,0.482729 ,0.809578 ,-0.17066 ,1.25176 ,0.392262 ,0.539647 ,0.830776 ,0.527684 ,-1.31646 ,-0.762485 ,0.246275 ,-0.53091 ,-0.0315935 ,0.422169 ,0.896237 ,2.44674 ,0.00710068 ,0.506156 ,-0.807028 ,-0.222211 ,-0.728801 ,-0.540276 ,-0.217843 ,0.320291 ,-0.536385 ,-0.431275 ,0.147188 ,0.53255 ,0.293468 ,-0.495369 ,1.00778 ,1.59495 ,-0.824085 ,1.12669 ,0.50992 ,0.409335 ,-1.2315 ,-0.30223 ,-0.131281 ,-1.38256 ,0.295639 ,-2.39828 ,-0.572134 ,0.533403 ,0.11061 ,-1.69643 ,1.23734 ,-0.156505 ,-1.79456 ,-0.329133 ,0.166238 ,-0.464965 ,-0.820017 ,-0.759239 ,-0.713275 ,-0.240518 ,1.07803 ,0.362155 ,-0.605187 ,1.26056 ,-1.4801 ,0.40077 ,-1.41042 ,1.05523 ,-0.776114 ,0.434785 ,1.18815 ,-0.440473 ,-0.104016 ,-0.624161 ,-1.17958 ,0.326495 ,0.28805 ,0.61735 ,-0.101467 ,-1.06451 ,0.638935 ,0.173481 ,-1.2974 ,1.09719 ,-1.26863 ,-0.786809 ,1.14516 ,0.429236 ,0.455908 ,0.538408 ,0.506936 ,0.673345 ,-0.0227002 ,2.86911 ,-1.16671 ,-0.210998]


theta=[1,1,0.01]
#x_1=[0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
y_1=get_y("1/2",x)
print len(y_1)

print len(x)
K=find_K(x,theta,"1/2")
K_inv=np.linalg.inv(K)
estimated_y_1=[]
new_numbers=np.arange(0,10,0.0002001)
for number in new_numbers:
    K_star_estimate = find_K_star(x,number,theta,"1/2")
    X_estimate = np.dot(K_star_estimate,K_inv)
    estimated_y_1.append((np.dot(X_estimate,y_1).tolist()))

plt.plot(new_numbers,estimated_y_1,'r')

y_2=get_y("3/2",x)
x_2=[0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
K=find_K(x_2,theta,"3/2")
K_inv=np.linalg.inv(K)
estimated_y_2=[]
new_numbers=np.arange(0,10,0.002001)
for number in new_numbers:
    K_star_estimate = find_K_star(x_2,number,theta,"3/2")
    X_estimate = np.dot(K_star_estimate,K_inv)
    estimated_y_2.append((np.dot(X_estimate,y_2).tolist()))

plt.plot(new_numbers,estimated_y_2,'y')


y_3=get_y("5/2",x)
x_3=[0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
K=find_K(x_3,theta,"5/2")
K_inv=np.linalg.inv(K)
estimated_y_3=[]
new_numbers=np.arange(0,10,0.002001)
for number in new_numbers:
    K_star_estimate = find_K_star(x_3,number,theta,"5/2")
    X_estimate = np.dot(K_star_estimate,K_inv)
    estimated_y_3.append((np.dot(X_estimate,y_3).tolist()))

plt.plot(new_numbers,estimated_y_3,'b')

x=[]
add=0
extra=0.05
for i in range(200):
    x.append(add)
    add+=extra

y_4=get_y("inf",x)
#x_4=[0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
K=find_K(x,theta,"inf")
K_inv=np.linalg.inv(K)
estimated_y_4=[]
new_numbers=np.arange(0,10,0.0020001)
for number in new_numbers:
    K_star_estimate = find_K_star(x,number,theta,"inf")
    X_estimate = np.dot(K_star_estimate,K_inv)
    estimated_y_4.append((np.dot(X_estimate,y_4).tolist()))

plt.plot(new_numbers,estimated_y_4,'g')
plt.axis([-3,3,-3,3])

plt.show()

r=[]
add=0
extra=0.01
for i in range(300):
    r.append(add)
    add+=extra
a_1=[]
a_2=[]
a_3=[]
a_4=[]
rc('text', usetex=True)
for index in range(len(r)):
    a_1.append(calculate_k(0,r[index],theta,"1/2"))
    a_2.append(calculate_k(0,r[index],theta,"3/2"))
    a_3.append(calculate_k(0,r[index],theta,"5/2"))
    a_4.append(calculate_k(0,r[index],theta,"inf"))
plt.figure(figsize=(4,4),dpi=250)
example,=plt.plot(r,a_1,'g',label=r'$\nu=\frac{1}{2}$')
plt.plot(r,a_2,'r',label=r'$\nu=\frac{3}{2}$')
plt.plot(r,a_3,'b',label=r'$\nu=\frac{5}{2}$')
plt.plot(r,a_4,'y',label=r'$\nu=\infty$')
plt.axis([0,3,0,1])
plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel(r'Distance $|x_1-x_2|$')
plt.ylabel("Covariance")
plt.legend()
plt.savefig("Rolloff.png",dpi=200)


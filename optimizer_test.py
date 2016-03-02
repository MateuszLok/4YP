__author__ = 'Mat'


import scipy.optimize as optimize
from pyDOE import lhs
import numpy as np
import scipy as sp

"""def f(input):
    return input[0]* 2 **3 - input[1]* 2*3 **2 - input[2]*2 +5


print 'Starting...'
latin_hypercube_values = lhs(3, samples=10)
latin_hypercube_values=latin_hypercube_values*5

result=np.zeros((len(latin_hypercube_values),3))

for number in range(0,len(latin_hypercube_values)):
    print number
    wynik= optimize.minimize(f, latin_hypercube_values[number],method='BFGS')
    result[number]=wynik['x']

print result

a = np.array([[1,2],[2,1]])
y=np.array([[1],[2]])
y_trans=y.transpose()


b=np.array([a,a],[a,a])

a=np.linalg.inv(a)



x= (np.dot(a,np.dot(b,a)))

z= np.dot(x,y)
print np.dot(y_trans,z)

list=[0,1,2,3,4,5,6]
M=list[:3]
rest=list[3:]
print M
print rest
M=[2,3]
M=np.vstack(M)
print M"""

a = np.array([[1,1],[1,1]])
while True:
    try:
        print sp.linalg.cholesky(a)
        break
    except:
        print "lol"
        a_sup=np.array([[0.01,0],[0,0.01]])
        a=np.add(a,a_sup)
        print a


print "exit"



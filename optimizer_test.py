__author__ = 'Mat'


import scipy.optimize as optimize
from pyDOE import lhs
import numpy as np

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
print np.dot(y_trans,z)"""

intermediate2 = -(5-2) * (2 **-1)
print intermediate2



__author__ = 'Mat'

import numpy as np

A = np.array([[1.7,1.42],[1.42,1.7]])
R = np.linalg.cholesky(A)
R_trans = R.transpose()
R_inv=np.linalg.inv(R)
R_trans_inv = np.linalg.inv(R_trans)
print R[1,1]
print np.dot(R_trans_inv,R_inv)
print np.linalg.inv(A)

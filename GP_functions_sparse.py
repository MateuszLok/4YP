__author__ = 'Mat'


from math import exp
import numpy as np
from math import log10
from math import log
from math import fabs
import json
from math import sin
import GP_functions_2 as gp
import GP_functions as gp1

#Simple case for testing derivative
def calculate_k(x1,x2,theta,type):
    if type=="normal":
        difference = x1-x2
        #print sigma_n
        l=exp(5)
        sigma_f=exp(theta[0])
        sigma_n=0.001
        intermediate = -fabs(difference) * (l **-1)
        if (x1==x2).all():
            return (sigma_f **2) * exp(intermediate) + sigma_n ** 2
        else:
            return (sigma_f **2) * exp(intermediate)
    elif type=="der_l":
        difference = x1-x2
        #print sigma_n
        l=exp(theta[0])
        sigma_f=exp(theta[1])
        sigma_n=0.001
        intermediate = -fabs(difference) * (l **-1)
        return (sigma_f **2) *fabs(difference)*(l)**-2 * exp(intermediate)
    elif type=="der_f":
        difference = x1-x2
        l=exp(5)
        sigma_f=exp(theta[0])
        sigma_n=0.001
        intermediate = -fabs(difference) * (l **-1)
        return 2*sigma_f* exp(intermediate)

#As in the paper
def calculate_kk(x1,x2,theta,type):
    if type=='normal':
        difference = x1-x2
        l1=exp(theta[0])
        l2=exp(theta[1])
        sigma_f1=exp(theta[2])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        if (x1==x2).all():
            return (sigma_f1 ** 2) * exp(intermediate1)+ (sigma_f2 ** 2) * exp(intermediate2) + sigma_n ** 2
        else:
            return (sigma_f1 ** 2) * exp(intermediate1) + (sigma_f2 ** 2) * exp(intermediate2)

    if type=='der_l1':
        difference = x1-x2
        l1=exp(theta[0])
        l2=exp(theta[1])
        sigma_f1=exp(theta[2])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        return (sigma_f1 ** 2)*fabs(difference)*(l1)**-2 * exp(intermediate1)

    if type=='der_l2':
        difference = x1-x2
        l1=exp(theta[0])
        l2=exp(theta[1])
        sigma_f1=exp(theta[2])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        return (sigma_f2 ** 2)*fabs(difference)*(l2)**-2 * exp(intermediate2)

    if type=='der_f1':
        difference = x1-x2
        l1=exp(theta[0])
        l2=exp(theta[1])
        sigma_f1=exp(theta[2])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        return 2*sigma_f1*exp(intermediate1)


#Evaluates matrix K - covariance matrix
#For MM and NN
def find_K(vector_X, theta,type):
    outcome = np.zeros((len(vector_X),len(vector_X)))
    for i in range(0, len(vector_X)):
        for j in range(i,len(vector_X)):
            temp = (calculate_k(vector_X[i],vector_X[j],theta,type))
            outcome[i][j] = temp
            outcome[j][i] = temp
    return outcome

#For MN
def find_K_2(vector_N, vector_M,theta,type):
    outcome = np.zeros((len(vector_N),len(vector_M)))
    for i in range(0, len(vector_N)):
        for j in range(0,len(vector_M)):
            temp = (calculate_k(vector_N[i],vector_M[j],theta,type))
            outcome[i][j] = temp
    return outcome



#Optimization
def function_to_minimize(input_data,M,end):

    #Alternative version
    vector_M=input_data[:M]
    vector_N=gp.get_time_vector(gp.get_volatility(end))
    theta=input_data[M:]
    y=np.array(gp.get_volatility(end))
    yt=y.transpose()
    K_MM=find_K(vector_M,theta,'normal')
    K_MM_inv=np.linalg.inv(K_MM)
    K_NM=find_K_2(vector_N,vector_M,theta,'normal')
    K_MN=K_NM.transpose()

    #Create Lambda diagonal matrix
    lambda_final=[]
    for index in range(0,len(vector_N)):
        Knn=calculate_k(vector_N[index],vector_N[index],theta,'normal')
        temp2=vector_N[index]
        kn=find_K_2(vector_M,[temp2],theta,'normal')
        temp=Knn-np.dot(kn.transpose(),np.dot(K_MM_inv,kn))
        lambda_final.append(temp)
    lambda_final=np.hstack(lambda_final)
    lambda_final_diag=np.diag(lambda_final[0])
    lambda_final_diag_inv=np.linalg.inv(lambda_final_diag)

    intermediate=K_MM+np.dot(K_MN,np.dot(lambda_final_diag_inv,K_NM))
    inverse_variance=lambda_final_diag_inv-np.dot(lambda_final_diag_inv,np.dot(K_NM,np.dot(np.linalg.inv(intermediate),np.dot(K_MN,lambda_final_diag_inv))))

    #Final likelihood calculations
    variance=np.dot(K_NM,np.dot(K_MM_inv,K_MN))+lambda_final_diag
    chol=np.linalg.cholesky(variance)
    transient=2*cholesky_det(chol)
    return 0.5*transient+0.5*np.dot(yt,np.dot(inverse_variance,y))


    """#Initialize variables
    vector_M=input_data[:M]
    vector_N=gp.get_time_vector(gp.get_volatility(end))
    theta=input_data[M:]
    y=np.array(gp.get_volatility(end))
    yt=y.transpose()
    K_MM=find_K(vector_M,theta,'normal')
    K_MM_inv=np.linalg.inv(K_MM)
    K_NM=find_K_2(vector_N,vector_M,theta,'normal')
    K_MN=K_NM.transpose()

    #Create Lambda diagonal matrix
    lambda_final=[]
    for index in range(0,len(vector_N)):
        Knn=calculate_k(vector_N[index],vector_N[index],theta,'normal')
        temp2=vector_N[index]
        kn=find_K_2(vector_M,[temp2],theta,'normal')
        temp=Knn-np.dot(kn.transpose(),np.dot(K_MM_inv,kn))
        lambda_final.append(temp)
    lambda_final=np.hstack(lambda_final)
    lambda_final_diag=np.diag(lambda_final[0])

    #Final likelihood calculations
    variance=np.dot(K_NM,np.dot(K_MM_inv,K_MN))+lambda_final_diag
    chol=np.linalg.cholesky(variance)
    transient=2*cholesky_det(chol)
    return 0.5*transient+0.5*np.dot(yt,np.dot(np.linalg.inv(variance),y))"""

# is Knm a transpose o Kmn

def cholesky_det(R):
    sum=0
    for number in range(0,len(R)):
        sum=sum+log10(R[number,number])
    return sum



    
































"""

#Fetch actual market data from a JSON file
def get_volatility(x):
    json_data = open('oil_futures.json').read()
    data = json.loads(json_data)
    days = len(data["dataset"]["data"])
    volatility = np.zeros((1,days))
    for number in range(0,days):
        max = data["dataset"]["data"][days-number-1][2]
        min = data["dataset"]["data"][days-number-1][3]
        if type(max) != float or type(min)!= float or min ==0  or max==0:
            volatility[0][number]=volatility[0][number-1]
        elif (log10(max)-log10(min)) == 0:
            volatility[0][number]=volatility[0][number-1]
        else:
            volatility[0][number]=log10(log10(max)-log10(min))
    if x=='all':
        return volatility[0]
    else:
        return volatility[0][0:x]

def get_time_vector(x):
    return np.array((range(0,len(x),1)))

#------------------------------------------------------------------------------

#Calculate covariance for two values
def calculate_kk(x1,x2,theta):
    difference = x1-x2
    l=exp(theta[1])
    sigma_f=exp(theta[0])
    sigma_n=exp(theta[2])
    intermediate = -np.dot(difference.transpose(),difference) * (2 * l ** 2) ** -1
    if (x1==x2).all():
        return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2
    else:
        return (sigma_f ** 2) * exp(intermediate)

#Not squared


def calculate_kk(x1,x2,theta):
    difference = x1-x2
    #print sigma_n
    l=exp(theta[0])
    sigma_f=exp(theta[1])
    sigma_n=exp(theta[2])
    intermediate = -fabs(difference) * (l **-1)
    if (x1==x2).all():
        return (sigma_f ** 2) * exp(intermediate) + sigma_n ** 2
    else:
        return (sigma_f ** 2) * exp(intermediate)

#Simple case for testing derivative
def calculate_kk(x1,x2,theta,type):
    if type=="normal":
        difference = x1-x2
        #print sigma_n
        l=exp(5)
        sigma_f=exp(theta[0])
        sigma_n=0.001
        intermediate = -fabs(difference) * (l **-1)
        if (x1==x2).all():
            return (sigma_f **2) * exp(intermediate) + sigma_n ** 2
        else:
            return (sigma_f **2) * exp(intermediate)
    elif type=="der_l":
        difference = x1-x2
        #print sigma_n
        l=exp(theta[0])
        sigma_f=exp(theta[1])
        sigma_n=0.001
        intermediate = -fabs(difference) * (l **-1)
        return (sigma_f **2) *fabs(difference)*(l)**-2 * exp(intermediate)
    elif type=="der_f":
        difference = x1-x2
        l=exp(5)
        sigma_f=exp(theta[0])
        sigma_n=0.001
        intermediate = -fabs(difference) * (l **-1)
        return 2*sigma_f* exp(intermediate)


#As in the paper
def calculate_kk(x1,x2,theta,type):
    if type=='normal':
        difference = x1-x2
        l1=exp(theta[0])
        sigma_f1=exp(theta[2])
        l2=exp(theta[1])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)

        if (x1==x2).all():
            return (sigma_f1 ** 2) * exp(intermediate1)+ (sigma_f2 ** 2) * exp(intermediate2) + sigma_n ** 2
        else:
            return (sigma_f1 ** 2) * exp(intermediate1) + (sigma_f2 ** 2) * exp(intermediate2)

    if type=='der_l1':
        difference = x1-x2
        l1=exp(theta[0])
        sigma_f1=exp(theta[2])
        l2=exp(theta[1])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        return (sigma_f1 ** 2)*fabs(difference)*(l1)**-2 * exp(intermediate1)

    if type=='der_l2':
        difference = x1-x2
        l1=exp(theta[0])
        sigma_f1=exp(theta[2])
        l2=exp(theta[1])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        return (sigma_f2 ** 2)*fabs(difference)*(l2)**-2 * exp(intermediate2)

    if type=='der_f1':
        difference = x1-x2
        l1=exp(theta[0])
        sigma_f1=exp(theta[2])
        l2=exp(theta[1])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        return 2*sigma_f1*exp(intermediate1)

    if type=='der_f2':
        difference = x1-x2
        l1=exp(theta[0])
        sigma_f1=exp(theta[2])
        l2=exp(theta[1])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = -fabs(difference) * (l2 **-1)
        return 2*sigma_f2*exp(intermediate2)

    if type=='der_n':
        difference = x1-x2
        l1=exp(theta[0])
        sigma_f1=exp(theta[2])
        l2=exp(theta[1])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])
        if (x1==x2).all():
            return 2*sigma_n
        else:
            return 0
#With periodic term
def calculate_k(x1,x2,theta,type):
    if type=='normal':
        difference = x1-x2
        l1=exp(theta[0])
        l2=exp(theta[1])
        sigma_f1=exp(theta[2])
        sigma_f2=exp(theta[3])
        sigma_n=exp(theta[4])
        intermediate1 = -fabs(difference) * (l1 **-1)
        intermediate2 = - 2* (sin((difference)*0.5))**2 * (l2 **-1)

        if (x1==x2).all():
            return (sigma_f1 ** 2) * exp(intermediate1)+ (sigma_f2 ** 2) * exp(intermediate2) + sigma_n ** 2
        else:
            return (sigma_f1 ** 2) * exp(intermediate1) + (sigma_f2 ** 2) * exp(intermediate2)
#-----------------------------------------------------------------------------------------------------
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
def find_K_star(vector,x,theta,type):
    outcome = np.zeros((1, len(vector)))
    for i in range(0,len(vector)):
        outcome[0][i] = calculate_k(x,vector[i],theta,type)
    return outcome

#Evaluates matrix K**
def find_K_2stars(x,theta,type):
    return calculate_k(x,x,theta,type)

#________________________________________________________________________________________________________________

#Function to optimize
def function_to_minimize_volatility(input_data, (end)):
    vector_x= get_time_vector(get_volatility(end))
    sample_y_opt = np.array(get_volatility(end))
    sample_y_trans_opt = sample_y_opt.transpose()
    chol=np.linalg.cholesky(find_K(vector_x,input_data,'normal'))
    transient = 2*cholesky_det(chol)
    chol_trans = chol.transpose()
    K_inverse = np.dot(np.linalg.inv(chol_trans),np.linalg.inv(chol))
    return (0.5 * np.dot(sample_y_trans_opt, np.dot(K_inverse, sample_y_opt)) +0.5 * transient + log10(6.28))


def cholesky_det(R):
    sum=0
    for number in range(0,len(R)):
        sum=sum+log10(R[number,number])
    return sum


def K_inverse_function(input_data,l):
    vector_x= get_time_vector(get_volatility(l))
    chol=np.linalg.cholesky(find_K(vector_x,input_data,'normal'))
    chol_trans = chol.transpose()
    K_inverse = np.dot(np.linalg.inv(chol_trans),np.linalg.inv(chol))
    return K_inverse

def jacobian_of_likelihood(input_data,end):
    length=end
    vector_x= get_time_vector(get_volatility(length))
    K_inv = K_inverse_function(input_data,length)
    sample_y_opt = np.array(get_volatility(length))
    sample_y_trans_opt = sample_y_opt.transpose()
    ytKinv=np.dot(sample_y_trans_opt,K_inv)
    Kinvy=np.dot(K_inv,sample_y_opt)
    Kf=find_K(vector_x,input_data,'der_f')
    #Kl=find_K(vector_x,input_data,'der_l')
    #Kf=-0.5*np.dot(ytKinv,np.dot(Kf,Kinvy))+0.5*np.trace(np.dot(K_inv,Kf))
    Kf=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kf,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kf))
    #Kl=-0.5*np.dot(ytKinv,np.dot(Kl,Kinvy))+0.5*np.trace(np.dot(K_inv,Kl))
    #Kl=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kl,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kl))
    result=np.array([Kf])
    return result.transpose()

def jacobian(input_data,end):
    vector_x= get_time_vector(get_volatility(end))
    sample_y_opt = np.array(get_volatility(end))
    sample_y_trans_opt = sample_y_opt.transpose()
    K_inv=K_inverse_function(input_data,end)
    Kf1=find_K(vector_x,input_data,'der_f1')
    Kf2=find_K(vector_x,input_data,'der_f2')
    Kl1=find_K(vector_x,input_data,'der_l1')
    Kl2=find_K(vector_x,input_data,'der_l2')
    Kn=find_K(vector_x,input_data,'der_n')

    Kf1=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kf1,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kf1))
    Kf2=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kf2,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kf2))
    Kl1=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kl1,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kl1))
    Kl2=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kl2,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kl2))
    Kn=-0.5*np.dot(sample_y_trans_opt,np.dot(K_inv,np.dot(Kn,np.dot(K_inv,sample_y_opt))))+0.5*np.trace(np.dot(K_inv,Kn))

    return np.array([Kl1,Kl2,Kf1,Kf2,Kn])"""






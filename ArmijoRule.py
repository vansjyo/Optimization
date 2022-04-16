# Armijo's rule for updating learning rate in descent

# imports
import numpy as np
import scipy.optimize
import time

# load data
Q = np.loadtxt('Q.txt')
b = np.reshape(np.loadtxt('b.txt'), [Q.shape[0],1])
print(b)
c = np.loadtxt('c.txt')

# define parameters
sigma = 10**-1
epsilon = 10**-5
beta = 0.5
alpha_k = 1
maxiter = 1000

# define f(x)
def fun(x):
    return np.transpose(x) @ Q @ x + np.transpose(b) @ x + c

# define derivative df(x)
def dfun(x):
    return (2*Q) @ x + b

# define armijo step
def armijoStep(x_k, alpha_k, sigma, beta):
    alpha = alpha_k
    count = 1
    while  fun(x_k - (alpha*dfun(x_k))) > fun(x_k) - (sigma*alpha*np.transpose(x_k)) @ x_k :
        alpha = alpha * beta
        # print("Performing Armijo's step: ", count, "at alpha = ", alpha )
        count = count + 1
        # time.sleep(1)
    return alpha

# define steepest descent
def descentStep(x_k, alpha_k, sigma, beta, epsilon, maxiter):
    k = 1
    while  (np.linalg.norm(dfun(x_k)) > epsilon and k <= maxiter) :
        print("-----Iteration", k, "-----")
        print("x: ", np.transpose(x_k))
        alpha = armijoStep(x_k, alpha_k, sigma, beta)
        x_k1 = x_k - (alpha*dfun(x_k))
        x_k = x_k1
        k = k + 1
    print("Number of Iterations: ", k)
    print("Final x: ", x_k)
    print("f(x): ", fun(x_k), " d( f(x) ): ", dfun(x_k) )

# run armijos iterative procedure
x_k = np.ones([b.shape[0], 1])
print(x_k + dfun(x_k))
print("---------- Performing Steepest Gradient Descent: ----------- \n")
descentStep(x_k, alpha_k, sigma, beta, epsilon, maxiter)

# run Matrix inverse method
print("---------- Performing Matrix Inverse: ------------- \n")
print("x: ", 0.5*np.matmul(np.linalg.inv(Q), b))

# run Scipy optimize method
print("---------- Using Scipy Optimize: ------------ \n")
print("x: ", scipy.optimize.minimize(fun,np.ones([b.shape[0], 1]),method='Nelder-Mead',tol=epsilon) )

# Lagrangian Multipliers

# Setting up environment
import numpy as np
import scipy.optimize as sco
# np.seterr(divide="ignore", invalid='ignore')

# Defining Functions
def fun(x):
    return x @ Q @ x
    # return (1/(1-sum(x))**2) - (w @ np.log(x))

def dfun(x):
    return (2 * Q) @ x
    # return (2/(1-sum(x))**3) - np.divide(w, x)

def eqConst(x):
    print(A @ x - b)
    return (A @ x - b)

def Lagrang(x, lamda, C_k):
    return fun(x) + lamda @ (eqConst(x)) + ((C_k/2) * (eqConst(x)@eqConst(x)))

def dLagrang(x, lamda, C_k):
    return dfun(x) + A @ lamda + (C_k*A) @ eqConst(x)

def stopK(x):
    return np.linalg.norm(eqConst(x)) < epsilon

def armijos(x, fun, dfun, C_k):

    x1, k = x, 0
    while ( (dfun(x1, lamda, C_k) @ dfun(x1, lamda, C_k))**0.5 > epsilon ) and k<2000:
        x0, k = x1, k+1
        print("Iteration:", k, " x0 = ", x0, " d(f(x))/dx: ", dfun(x0, lamda, C_k))
        for m in range(0, 1000):
            x1 = x0 - ((beta**m) * alpha * dfun(x0, lamda, C_k))
            if fun(x1, lamda, C_k) <= ( fun(x0, lamda, C_k) - (sigma * (beta**m) * alpha * (dfun(x0, lamda, C_k) @ dfun(x0, lamda, C_k))) ):
                break
    print("\n Using Armijo's Rule: \n Final X: ", x1, "\n Final d(f(x))/dx: ", dfun(x1, lamda, C_k))
    return x1

def argminLagrangian(x, lamda, C_k):
    x = armijos(x, Lagrang, dLagrang, C_k)
    lamda = lamda + C_k*(eqConst(x))
    # print(x, lamda)
    return [x, lamda]

if __name__ == "__main__":
    # Defining inputs
    Q = np.array([[1,2,3],[4,3,2],[4,5,7]])
    A = np.array([[1,3,4],[2,5,1],[1,7,9]])
    b = np.array([1,2,3])
    x = np.array([1,1,1]) #np.array([0.25, 0.25, 0.25]) 
    lamda = np.array([1,0,1])
    alpha, beta, sigma, epsilon, C_k = 0.01, 0.1, 10**-5, 10**-2, 1
    k = 0

    while (not stopK(x)):
        print("\nIteration: ", k, "Stop h(x) = ", eqConst(x))
        [x, lamda] = argminLagrangian(x, lamda, C_k)
        C_k = C_k*1.01
        k = k + 1

    print("\nValue of x is:", x) 
    print("\nValue of lamda is:", lamda) 
    print("\nValue of h(x) is:", eqConst(x))
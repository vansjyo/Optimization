# Armijo's rule udpate for constrained optimization

# Setting up environment
import numpy as np
import scipy.optimize as sco
np.seterr(divide="ignore", invalid='ignore')


# Defining inputs
x = np.array([0.25, 0.25, 0.25])
w = np.array([1, 2, 3])
alpha, beta, sigma, epsilon = 0.0005, 0.5, 0.1, 10**-3

# Defining Constraints
constraints = [
    {'type': 'ineq', 'fun': lambda x: x},
    {'type': 'ineq', 'fun': lambda x: sum(x)}
]

# Defining Functions
def fun(x, w):
    return (1/(1-sum(x))**2) - (w @ np.log(x))

def dfun(x, w):
    return (2/(1-sum(x))**3) - np.divide(w, x)
    

# Armijo's Rule Solution
x1, k = x, 0
while ( (dfun(x1, w) @ dfun(x1, w))**0.5 > epsilon ):
    x0, k = x1, k+1
    print("\n Iteration:", k, " x0 = ", x0, " d(f(x))/dx: ", dfun(x0, w))
    for m in range(0, 1000):
        x1 = x0 - ((beta**m) * alpha * dfun(x0, w))
        if fun(x1, w) <= ( fun(x0, w) - (sigma * (beta**m) * alpha * (dfun(x0, w) @ dfun(x0, w))) ):
            break
print("\n Using Armijo's Rule: \n Final X: ", x1, "\n Final d(f(x))/dx: ", dfun(x1, w))

# Scipy Optimize
xf = sco.minimize(fun, x, args=(w), method='SLSQP', constraints=constraints)
print("\n Using Scipy Optimize:\n Final X: ", xf.x, "\n Final d(f(x))/dx: ", dfun(xf.x, w))
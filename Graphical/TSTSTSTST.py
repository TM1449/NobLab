import numpy as np
from scipy import optimize
import scipy.linalg

a = 0.5
b = 0.4
c = 0.89
k_0 = -0.44
k_1 = 0.1
k_2 = 0.2
alpha = 0.1
beta = 0.1
k = 4.1703

x = 0.831

y = round((-b * x + c) / (1-a),4)
phi = round(k_1 * x / (1+k_2),4)

print(x,y,phi)

A = np.array([[np.exp(y-x) * (2*x - x**2) + k * (alpha + 3 * beta*phi**2), x**2*np.exp(y-x), 6*k*x*beta*phi], [-1*b, a, 0],[k_1,0,-1*k_2]])
val = scipy.linalg.eigvals(A)

print(val)
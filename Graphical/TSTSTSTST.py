import numpy as np
from scipy import optimize
import scipy.linalg

a = 0.89
b = 0.6
c = 0.28
k_0 = 0.04
k_1 = 0.1
k_2 = 0.2
alpha = 0.1
beta = 0.2
k = -3.2

InputSingal = 8

dt = 0.00001
ps = -10
pe = 10

dx = np.arange(ps,pe,dt)
dx = np.round(dx, 1)

"""------------------------"""
"""
for i in range(ps*100000,pe*100000,1):
    i = i * dt
    fx = pow(i,2) * np.exp(((b - a + 1) * i - c) / (a - 1)) + k_0 \
    + ((3*k*beta*pow(k_1,2)) / pow((1 + k_2), 2)) * pow(i,3) \
        + i * k * alpha + InputSingal
    fg = i

    fx_r = round(fx, 4)
    fg_r = round(fg, 4)

    if pow((fx_r - fg_r),2) == 0:
        print(fx_r)
"""

"""------------------------"""
#-0.2118
#0.4605 or 6
#1.755 or 51
#4.5592 or 3 or 4

dz = 4.9015
fx = pow(dz,2) * np.exp(((b - a + 1) * dz - c) / (a - 1)) + k_0 \
    + ((3*k*beta*pow(k_1,2)) / pow((1 + k_2), 2)) * pow(dz,3) \
        + dz * k * alpha + InputSingal
print("x=")
print(dz)
print("fx=")
print(round(fx,4))
print("\n")

"""------------------------"""

x = 4.9015

y = round((-b * x + c) / (1-a),4)
phi = round(k_1 * x / (1+k_2),4)
print("Fixed Point :: x,y,phi = ")
print(x,y,phi)
print()

A = np.array\
    ([[np.exp(y - x) * (2 * x - pow(x, 2)) + k * (alpha + 3 * beta * pow(phi,2)), pow(x, 2) * np.exp(y - x), 6 * k * x * beta * phi],\
        [-b, a, 0],\
            [k_1, 0, -k_2]])
val = scipy.linalg.eigvals(A)
print("Eigenvalue = ")
print(val)

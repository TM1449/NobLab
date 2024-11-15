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
k = 7.6

InputSingal = 0

dt = 1e-06
print(dt)
dt_R = int(np.reciprocal(dt))
print(dt_R)
ps = -10
pe = 10

dx = np.arange(ps,pe,dt)
dx = np.round(dx, 1)

"""------------------------"""
sss = 0
#"""
for i in range(ps*dt_R, pe*dt_R,1):
    i = i * dt
    fx = pow(i,2) * np.exp(((b - a + 1) * i - c) / (a - 1)) + k_0 \
    + ((3*k*beta*pow(k_1,2)) / pow((1 + k_2), 2)) * pow(i,3) \
        + i * k * alpha + InputSingal
    fg = i

    fx_r = round(fx, 6)
    fg_r = round(fg, 6)
    print("\r%d / %d , %fper "%(sss, (pe)*2*dt_R, (sss/((pe)*2*dt_R)*100)), end = "")
    sss += 1

    if pow((fx_r - fg_r),2) == 0:
        print(fx_r)
#"""

"""------------------------"""
#-0.2118
#0.4605 or 6
#1.755 or 51
#4.5592 or 3 or 4

"""
dz = 0.044843
fx = pow(dz,2) * np.exp(((b - a + 1) * dz - c) / (a - 1)) + k_0 \
    + ((3*k*beta*pow(k_1,2)) / pow((1 + k_2), 2)) * pow(dz,3) \
        + dz * k * alpha + InputSingal
print("x=")
print(dz)
print("fx=")
print(round(fx,6))
print("\n")
"""
"""------------------------"""
"""
x = 0.044843

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
"""
import numpy as np
from scipy import optimize

a = 0.89
b = 0.6
c = 0.28
k_0 = 0.04
k_1 = 0.1
k_2 = 0.2
alpha = 0.1
beta = 0.2
k = -3.2

x = 0.0448

y = (-b * x + c) / (1-a)
phi = k_1 * x / (1+k_2)

print(x,y,phi)
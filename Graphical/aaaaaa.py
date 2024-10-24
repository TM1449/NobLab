import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

RunTime = 1000

x = np.zeros(RunTime)
for i in range(RunTime - 1):
    x[i+1] =  0.8 * np.sin(5 * i * np.pi / 180) * 0.3 * np.cos(7 * i * np.pi / 180) + (random.random() * 2 - 1) * 0.2

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

# 曲線をプロット
ax.plot(x[:])
plt.show()
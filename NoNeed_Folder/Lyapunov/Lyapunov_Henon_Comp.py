#====================================================================
import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
import math


#====================================================================
#エノン写像のパラメータ
a = 1.4
b = 0.3 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#空走時間
Burn_in_time = 1000

#実行時間
Runtime = 5000

#全体時間
Alltime = Burn_in_time + Runtime

Lyapunov_expotent = True

#====================================================================
#リアプノフ指数
if Lyapunov_expotent:
    print("\nエノン写像: 最大リアプノフ指数の導出")
    #初期値の指定
    Initial_x = None 
    Initial_y = None
    #--------------------------------------------------------------------
    #基準軌道xと初期値について
    x = np.zeros(Alltime)

    if Initial_x == None:
        x[0] = np.random.random() * 0.01
    else:
        x[0] = Initial_x
    
    y = np.zeros(Alltime)

    if Initial_y == None:
        y[0] = np.random.random() * 0.01
    else:
        y[0] = Initial_y

    #--------------------------------------------------------------------
    print("\n基準軌道の計算")
    #基準軌道の計算部    
    for i in range(Alltime-1):
        print("\r%d / %d"%(i, Alltime), end = "")
        x[i+1] = 1 - a * x[i] ** 2 + b * y[i]
        y[i+1] = x[i]
        
    #--------------------------------------------------------------------
    #リアプノフ指数を取る配列
    x_Lyapunov = np.zeros(Runtime-1)
    y_Lyapunov = np.zeros(Runtime-1)
    
    #初期摂動が与えられた軌道
    x_D = np.zeros(Runtime)
    y_D = np.zeros(Runtime)
    
    #総和を取る配列
    x_sum = 0
    y_sum = 0
    
    #初期摂動を確保する配列
    delta_0_x = np.zeros(Runtime)
    delta_0_y = np.zeros(Runtime)
    
    delta_0_x[0] = 1e-06
    delta_0_y[0] = 1e-06
    
    #初期摂動を1step発展させる配列
    delta_tau_x = np.zeros(Runtime)
    delta_tau_y = np.zeros(Runtime)
    
    delta_tau_x[0] = 1e-06
    delta_tau_y[0] = 1e-06
    
    #--------------------------------------------------------------------
    #規準軌道に初期摂動を与える
    x_D[0] = x[Burn_in_time] + delta_0_x[0]
    y_D[0] = y[Burn_in_time] + delta_0_y[0]
    
    #初期摂動を与えたものを1step時間発展させた
    x_D[1] = 1 - a * x_D[0] ** 2 + b * y_D[0]
    y_D[1] = x_D[0]

    #時間発展させた2つの軌道の差の算出
    delta_tau_x[0] = x_D[1] - x[Burn_in_time+1]
    delta_tau_y[0] = y_D[1] - y[Burn_in_time+1]
    
    #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
    #（負ならば差が縮まる、正なら差が広がる）
    x_Lyapunov[0] = np.log(abs(delta_tau_x[0]) / abs(delta_0_x[0]))
    x_sum += x_Lyapunov[0]
    
    y_Lyapunov[0] = np.log(abs(delta_tau_y[0]) / abs(delta_0_y[0]))
    y_sum += y_Lyapunov[0]

    print("\n\n摂動軌道の計算")
    #摂動軌道の計算部
    for i in range(1,Runtime-1):
        print("\r%d / %d"%(i, Runtime), end = "")
        delta_0_x[i] = (delta_tau_x[i-1] / abs(delta_tau_x[i-1])) * delta_0_x[0]
        delta_0_y[i] = (delta_tau_y[i-1] / abs(delta_tau_y[i-1])) * delta_0_y[0]
    
        #規準軌道に初期摂動を与える
        x_D[i] = x[Burn_in_time+i] + delta_0_x[i]
        y_D[i] = y[Burn_in_time+i] + delta_0_y[i]
        
        #初期摂動を与えたものを1step時間発展させた
        x_D[i+1] = 1 - a * x_D[i] ** 2 + b * y_D[i]
        y_D[i+1] = x_D[i]
        
        #時間発展させた2つの軌道の差の算出
        delta_tau_x[i] = x_D[i+1] - x[Burn_in_time+i+1]
        delta_tau_y[i] = y_D[i+1] - y[Burn_in_time+i+1]
    
        #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
        #（負ならば差が縮まる、正なら差が広がる）
        x_Lyapunov[i] = np.log(abs(delta_tau_x[i]) / abs(delta_0_x[i]))
        x_sum += x_Lyapunov[i]

        y_Lyapunov[i] = np.log(abs(delta_tau_y[i]) / abs(delta_0_y[i]))
        y_sum += y_Lyapunov[i]

    x_ave = x_sum/len(x_Lyapunov)
    y_ave = y_sum/len(y_Lyapunov)
    print("\n\nxのリアプノフ指数")
    print(x_ave)    
    print("\nyのリアプノフ指数")
    print(y_ave)
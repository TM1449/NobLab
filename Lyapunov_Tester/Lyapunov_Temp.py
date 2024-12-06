#====================================================================
import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt


#====================================================================
#電磁束下Chialvoのパラメータ
a = 0.89 #0.89 0.6
b = 0.6 #0.6 0.1
c = 0.28 #0.28 1.4
k0 = 0.04 #0.04 0.1

k1 = 0.1 #0.1 0.1
k2 = 0.2 #0.2 0.2
alpha = 0.1 #0.1 0.1
beta = 0.2 #0.2 0.1

k = 0 #-3.2 -0.5

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#入力信号
InputSingal = 0

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#空走時間
Burn_in_time = 10000

#実行時間
Runtime = 100000

#全体時間
Alltime = Burn_in_time + Runtime

Lyapunov_expotent = True

#====================================================================
#リアプノフ指数
if Lyapunov_expotent:
    print("\n最大リアプノフ指数の導出")
    #初期値の指定
    Initial_x = None #0.0448
    Initial_y = None #2.3011
    Initial_phi = None #0.0037

    #--------------------------------------------------------------------
    #基準軌道xと初期値について
    x = np.zeros(Runtime)

    if Initial_x == None:
        x[0] = np.random.random() * 0.2 - 0.1
    else:
        x[0] = Initial_x

    #--------------------------------------------------------------------
    #基準軌道yと初期値について
    y = np.zeros(Runtime)

    if Initial_y == None:
        y[0] = np.random.random() * 0.2 - 0.1
    else:
        y[0] = Initial_y

    #--------------------------------------------------------------------
    #基準軌道phiと初期値について
    phi = np.zeros(Runtime)

    if Initial_phi == None:
        phi[0] = np.random.random() * 0.2 - 0.1
    else:
        phi[0] = Initial_phi

    #--------------------------------------------------------------------
    #入力信号の作成
    Input_Signal_In = np.ones(Runtime) * InputSingal

    dt = 0.001
    scale = 1 / 500
    
    sigma = 10
    rho = 28
    beta = 8/3

    rorentu_x = np.ones(Runtime)
    rorentu_x[0] = np.random.random() * 0.0001
    Input_Signal_In[0] = rorentu_x[0]
    
    rorentu_y = np.ones(Runtime)
    rorentu_y[0] = np.random.random() * 0.0001
    
    rorentu_z = np.ones(Runtime)
    rorentu_z[0] = np.random.random() * 0.0001

    for n in range(Runtime-1):
        #Input_Signal_In[n] = 0.1 * InputSingal * np.cos(4 * n * np.pi / 180)
        
        rorentu_x[n+1] = dt * (sigma *(rorentu_y[n] - rorentu_x[n])) + rorentu_x[n]
        rorentu_y[n+1] = dt * (rorentu_x[n] * (rho - rorentu_z[n]) - rorentu_y[n]) + rorentu_y[n]
        rorentu_z[n+1] = dt * (rorentu_x[n] * rorentu_y[n] - beta * rorentu_z[n]) + rorentu_z[n]

        Input_Signal_In[n+1] = scale * rorentu_x[n+1]
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(Input_Signal_In)

    plt.show()

    #--------------------------------------------------------------------
    #規準軌道の計算部
    
    #空走時間の計算部
    for i in range(Runtime-1):
        print("\r%d / %d"%(i, Burn_in_time), end = "")
        x[i+1] = pow(x[i], 2) * np.exp(y[i] - x[i]) + k0\
              + k * x[i] * (alpha + 3 * beta * pow(phi[i], 2))\
                + Input_Signal_In[i]
        y[i+1] = a * y[i] - b * x[i] + c
        phi[i+1] = k1 * x[i] - k2 * phi[i]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(x)

    plt.show()
    #--------------------------------------------------------------------
    #リアプノフ指数を取る配列
    x_Lyapunov = np.zeros(Runtime)
    y_Lyapunov = np.zeros(Runtime)
    phi_Lyapunov = np.zeros(Runtime)

    #初期摂動が与えられた軌道
    x_D = np.zeros(Runtime)
    y_D = np.zeros(Runtime)
    phi_D = np.zeros(Runtime)

    #総和を取る配列
    x_sum = 0
    y_sum = 0
    phi_sum = 0

    #初期摂動を確保する配列
    delta_0_x = np.zeros(Runtime)
    delta_0_y = np.zeros(Runtime)
    delta_0_phi = np.zeros(Runtime)
    
    delta_0_x[0] = 1e-06
    delta_0_y[0] = 1e-06
    delta_0_phi[0] = 1e-06

    #初期摂動を1step発展させる配列
    delta_tau_x = np.zeros(Runtime)
    delta_tau_y = np.zeros(Runtime)
    delta_tau_phi = np.zeros(Runtime)
    
    delta_tau_x[0] = 1e-06
    delta_tau_y[0] = 1e-06
    delta_tau_phi[0] = 1e-06

    #--------------------------------------------------------------------
    #規準軌道に初期摂動を与える
    x_D[0] = x[0] + delta_0_x[0]
    y_D[0] = y[0] + delta_0_y[0]
    phi_D[0] = phi[0] + delta_0_phi[0]
    
    #初期摂動を与えたものを1step時間発展させた
    x_D[1] = pow(x_D[0], 2) * np.exp(y_D[0] - x_D[0]) + k0 \
              + k * x_D[0] * (alpha + 3 * beta * pow(phi_D[0], 2)) \
                + Input_Signal_In[0]
    y_D[1] = a * y_D[0] - b * x_D[0] + c
    phi_D[1] = k1 * x_D[0] - k2 * phi_D[0]

    #時間発展させた2つの軌道の差の算出
    delta_tau_x[0] = x_D[1] - x[1]
    delta_tau_y[0] = y_D[1] - y[1]
    delta_tau_phi[0] = phi_D[1] - phi[1]

    #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
    #（負ならば差が縮まる、正なら差が広がる）
    x_Lyapunov[0] = np.log(abs(delta_tau_x[0]) / abs(delta_0_x[0]))
    x_sum += x_Lyapunov[0]

    y_Lyapunov[0] = np.log(abs(delta_tau_y[0]) / abs(delta_0_y[0]))
    y_sum += y_Lyapunov[0]

    phi_Lyapunov[0] = np.log(abs(delta_tau_phi[0]) / abs(delta_0_phi[0]))
    phi_sum += phi_Lyapunov[0]
#    print("\n")
#   print(y_Lyapunov[0])

    #摂動軌道の計算部
    for i in range(1,Runtime-1):
        delta_0_x[i] = (delta_tau_x[i-1] / abs(delta_tau_x[i-1])) * delta_0_x[0]
        delta_0_y[i] = (delta_tau_y[i-1] / abs(delta_tau_y[i-1])) * delta_0_y[0]
        delta_0_phi[i] = (delta_tau_phi[i-1] / abs(delta_tau_phi[i-1])) * delta_0_phi[0]

        #規準軌道に初期摂動を与える
        x_D[i] = x[i] + delta_0_x[i]
        y_D[i] = y[i] + delta_0_y[i]
        phi_D[i] = phi[i] + delta_0_phi[i]
        
        #初期摂動を与えたものを1step時間発展させた
        x_D[i+1] = pow(x_D[i], 2) * np.exp(y_D[i] - x_D[i]) + k0 \
                + k * x_D[i] * (alpha + 3 * beta * pow(phi_D[i], 2)) \
                    + Input_Signal_In[i]
        y_D[i+1] = a * y_D[i] - b * x_D[i] + c
        phi_D[i+1] = k1 * x_D[i] - k2 * phi_D[i]

        #時間発展させた2つの軌道の差の算出
        delta_tau_x[i] = x_D[i+1] - x[i+1]
        delta_tau_y[i] = y_D[i+1] - y[i+1]
        delta_tau_phi[i] = phi_D[i+1] - phi[i+1]

        #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
        #（負ならば差が縮まる、正なら差が広がる）
        x_Lyapunov[i] = np.log(abs(delta_tau_x[i]) / abs(delta_0_x[i]))
        x_sum += x_Lyapunov[i]

        y_Lyapunov[i] = np.log(abs(delta_tau_y[i]) / abs(delta_0_y[i]))
        y_sum += y_Lyapunov[i]

        phi_Lyapunov[i] = np.log(abs(delta_tau_phi[i]) / abs(delta_0_phi[i]))
        phi_sum += phi_Lyapunov[i]

    print("\nxのリアプノフ指数")
    print(x_sum/len(x_Lyapunov))

    print("\nyのリアプノフ指数")
    print(y_sum/len(y_Lyapunov))

    print("\nphiのリアプノフ指数")
    print(phi_sum/len(phi_Lyapunov))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    #ax.plot(x_Lyapunov[0:500])
    ax.plot(y_Lyapunov[0:500])
    ax.plot(phi_Lyapunov[0:500])

    plt.show()

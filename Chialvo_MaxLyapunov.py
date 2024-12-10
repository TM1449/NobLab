#====================================================================
import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
import math
import datetime
import os


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

k = 7.1 #-3.2 -0.5

k_list = np.arange(-7, 7.01, 0.1)
k_list_result = np.zeros(len(k_list))
k_list_result_M = np.zeros((len(k_list), 10))

#print(k_list_result)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#入力信号
InputSingal = 0

#None, sin, cos（ローレンツ方程式）
InputSingal_def = None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#空走時間
Burn_in_time = 500

#実行時間
Runtime = 5000

#全体時間
Alltime = Burn_in_time + Runtime

#リアプノフ指数を1回求める
Lyapunov_expotent = True

#====================================================================
#リアプノフ指数を1回求める
#1step後の2つの軌道の差のプロット
if Lyapunov_expotent:

    print("\n====================================================================")
    print("====================================================================")

    print(f"\n空走時間: {Burn_in_time}")
    print(f"計算時間: {Runtime}")

    print("\n====================================================================")
    print("====================================================================")

    print("\n電磁束下Chialvoニューロン: 最大リアプノフ指数の導出")
    #初期値の指定
    Initial_x = None #0.0448
    Initial_y = None #2.3011
    Initial_phi = None #0.0037

    #--------------------------------------------------------------------
    #基準軌道xと初期値について
    x = np.zeros(Alltime)

    if Initial_x == None:
        x[0] = np.random.random() * 2 - 1
    else:
        x[0] = Initial_x

    #--------------------------------------------------------------------
    #基準軌道yと初期値について
    y = np.zeros(Alltime)

    if Initial_y == None:
        y[0] = np.random.random() * 2 - 1
    else:
        y[0] = Initial_y

    #--------------------------------------------------------------------
    #基準軌道phiと初期値について
    phi = np.zeros(Alltime)

    if Initial_phi == None:
        phi[0] = np.random.random() * 2 - 1
    else:
        phi[0] = Initial_phi

    #--------------------------------------------------------------------
    #入力信号の作成
    Input_Signal_In = np.ones(Alltime - 1) * InputSingal

    if InputSingal_def == None:
        pass

    elif InputSingal_def == np.sin:
        for n in range(Alltime - 1):
            Input_Signal_In[n] = 0.1 * InputSingal * np.cos(4 * n * np.pi / 180)

    elif InputSingal_def == np.cos:
        #微小時間
        dt = 0.01
        #信号の大きさ
        scale = 1 / 10000 * InputSingal

        #パラメータ
        sigma = 10
        rho = 28
        beta = 8/3

        #入力信号の長さ
        rorentu_x = np.ones(Alltime - 1)
        rorentu_x[0] = np.random.random() * 0.0001
        
        rorentu_y = np.ones(Alltime - 1)
        rorentu_y[0] = np.random.random() * 0.0001
        
        rorentu_z = np.ones(Alltime - 1)
        rorentu_z[0] = np.random.random() * 0.0001

        Input_Signal_In[0] = rorentu_x[0]

        for n in range(Alltime - 1):

            rorentu_x[n+1] = dt * (sigma *(rorentu_y[n] - rorentu_x[n])) + rorentu_x[n]
            rorentu_y[n+1] = dt * (rorentu_x[n] * (rho - rorentu_z[n]) - rorentu_y[n]) + rorentu_y[n]
            rorentu_z[n+1] = dt * (rorentu_x[n] * rorentu_y[n] - beta * rorentu_z[n]) + rorentu_z[n]

            Input_Signal_In[n+1] = scale * rorentu_x[n+1]

    """#入力信号のプロット
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Input_Signal_In)

    plt.show()"""
    #--------------------------------------------------------------------
    #規準軌道の計算部
    print("\n--------------------------------------------------------------------")

    print("\n基準軌道の計算")

    for i in range(Alltime - 1):
        print("\r%d / %d"%(i, Alltime), end = "")
        x[i+1] = pow(x[i], 2) * np.exp(y[i] - x[i]) + k0\
              + k * x[i] * (alpha + 3 * beta * pow(phi[i], 2))\
                + Input_Signal_In[i]
        y[i+1] = a * y[i] - b * x[i] + c
        phi[i+1] = k1 * x[i] - k2 * phi[i]

    #--------------------------------------------------------------------
    #最大リアプノフ指数を取る配列
    Lyapunov = np.zeros(Runtime)
    Lyapunov_Sum = 0

    #初期摂動が与えられた軌道
    x_D = np.zeros(Runtime)
    y_D = np.zeros(Runtime)
    phi_D = np.zeros(Runtime)

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
    #基準軌道に初期摂動を与える
    x_D[0] = x[Burn_in_time] + delta_0_x[0]
    y_D[0] = y[Burn_in_time] + delta_0_y[0]
    phi_D[0] = phi[Burn_in_time] + delta_0_phi[0]

    #摂動軌道の算出
    x_D[1] = pow(x_D[0], 2) * np.exp(y_D[0] - x_D[0]) + k0 \
            + k * x_D[0] * (alpha + 3 * beta * pow(phi_D[0], 2)) \
                + Input_Signal_In[Burn_in_time]
    y_D[1] = a * y_D[0] - b * x_D[0] + c
    phi_D[1] = k1 * x_D[0] - k2 * phi_D[0]

    #基準軌道の初期値
    Standard_0 = np.array([x[Burn_in_time], y[Burn_in_time], phi[Burn_in_time]])
    #基準軌道の1Step先の値
    Standard_1 = np.array([x[Burn_in_time+1], y[Burn_in_time+1], phi[Burn_in_time+1]])
    
    #摂動軌道の初期値
    Pert_0 = np.array([x_D[0], y_D[0], phi_D[0]])
    #摂動軌道の1Step先の値
    Pert_1 = np.array([x_D[1], y_D[1], phi_D[1]])

    #初期値の差ベクトル
    Vector_0 = (Pert_0 - Standard_0)
    #1Step先の差ベクトル
    Vector_1 = (Pert_1 - Standard_1)

    #初期値の差ベクトルの大きさ
    Norm_0 = np.linalg.norm(Vector_0)
    #1Step先の差ベクトルの大きさ
    Norm_1 = np.linalg.norm(Vector_1)

    print("\n--------------------------------------------------------------------")
    print(f"\n基準軌道の初期値の点: {Standard_0}\n摂動軌道の初期値の点: {Pert_0}")
    print(f"\n基準軌道の発展先の点: {Standard_1}\n摂動軌道の発展先の点: {Pert_1}")

    print(f"\n初期値の差ベクトル: {Vector_0}\n1Step先の差ベクトル: {Vector_1}")
    print(f"\n初期値の差ベクトルの大きさ: {Norm_0}\n1Step先の差ベクトルの大きさ: {Norm_1}")
    
    Lyapunov[0] = np.log(Norm_1 / Norm_0)
    Lyapunov_Sum = Lyapunov_Sum + Lyapunov[0]
    print(f"\n1回目の最大リアプノフ指数: {Lyapunov[0]}")

    #====================================================================
    for i in range(1,Runtime-1):
        delta_0_x[i] = (Vector_1[0] / Norm_1) * delta_0_x[0]
        delta_0_y[i] = (Vector_1[1] / Norm_1) * delta_0_y[0]
        delta_0_phi[i] = (Vector_1[2] / Norm_1) * delta_0_phi[0]

        #規準軌道に初期摂動を与える
        x_D[i] = x[Burn_in_time+i] + delta_0_x[i]
        y_D[i] = y[Burn_in_time+i] + delta_0_y[i]
        phi_D[i] = phi[Burn_in_time+i] + delta_0_phi[i]

        #摂動軌道の算出
        x_D[i+1] = pow(x_D[i], 2) * np.exp(y_D[i] - x_D[i]) + k0 \
                + k * x_D[i] * (alpha + 3 * beta * pow(phi_D[i], 2)) \
                    + Input_Signal_In[Burn_in_time + i]
        y_D[i+1] = a * y_D[i] - b * x_D[i] + c
        phi_D[i+1] = k1 * x_D[i] - k2 * phi_D[i]

        #基準軌道の初期値
        Standard_0 = np.array([x[Burn_in_time+i], y[Burn_in_time+i], phi[Burn_in_time+i]])
        #基準軌道の1Step先の値
        Standard_1 = np.array([x[Burn_in_time+i+1], y[Burn_in_time+i+1], phi[Burn_in_time+i+1]])
        
        #摂動軌道の初期値
        Pert_0 = np.array([x_D[i], y_D[i], phi_D[i]])
        #摂動軌道の1Step先の値
        Pert_1 = np.array([x_D[i+1], y_D[i+1], phi_D[i+1]])

        #初期値の差ベクトル
        Vector_0 = (Pert_0 - Standard_0)
        #1Step先の差ベクトル
        Vector_1 = (Pert_1 - Standard_1)

        #初期値の差ベクトルの大きさ
        Norm_0 = np.linalg.norm(Vector_0)
        #1Step先の差ベクトルの大きさ
        Norm_1 = np.linalg.norm(Vector_1)

        print("\n--------------------------------------------------------------------")
        print(f"\n基準軌道の初期値の点: {Standard_0}\n摂動軌道の初期値の点: {Pert_0}")
        print(f"\n基準軌道の発展先の点: {Standard_1}\n摂動軌道の初期値の点: {Pert_1}")

        print(f"\n初期値の差ベクトル: {Vector_0}\n1Step先の差ベクトル: {Vector_1}")
        print(f"\n初期値の差ベクトルの大きさ: {Norm_0}\n1Step先の差ベクトルの大きさ: {Norm_1}")
        
        Lyapunov[i] = np.log(Norm_1 / Norm_0)
        Lyapunov_Sum = Lyapunov_Sum + Lyapunov[i]
        print(f"\n{i+1}回目の最大リアプノフ指数: {Lyapunov[i]}")

    Lyapunov_Ave = Lyapunov_Sum / (Runtime - 1)
    print(Lyapunov_Ave)
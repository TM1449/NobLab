#====================================================================

import numpy as np
import scipy.linalg
import time

#====================================================================
#電磁束下Chialvoのパラメータ
a = 0.89 #0.89
b = 0.6 #0.6
c = 0.28 #0.28
k0 = 0.04 #0.04

k1 = 0.1 #0.1
k2 = 0.2 #0.2
alpha = 0.1 #0.1
beta = 0.2 #0.2

k = -3.2

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#入力信号
InputSingal = 0

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#実行時間
Runtime = 10000

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#計測間隔
setting = {
    "dx"    : 1e-07,
    "dx_R"  : 1e+07,
    "round" : 7
}

#計測間隔
dx = setting["dx"]
print(f"計測間隔：{dx}")
dx_R = int(setting["dx_R"])
print(f"計測間隔の逆数：{dx_R}")
round_Pre = setting["round"]

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#グラフの交点の探索開始, 終了地点
Plot_Start = -1
Plot_End = 1

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#交点を求める関数
Derive_of_Intersections = False
#固定点の導出
Derive_of_FixedPoint = False
#リアプノフ指数
Lyapunov_expotent = True

#固定点を加えるためのリスト
FixedPoint_List = list() #list() #[]

#====================================================================
#交点を求める関数
if Derive_of_Intersections:
    print("\nNullclineの交点導出")
    Counter = 0

    Time_Start = time.time()
    
    for i in range(Plot_Start * dx_R, Plot_End * dx_R, 1):
        i = i * dx
        Fx = pow(i,2) * np.exp(((b - a + 1) * i - c) / (a - 1)) + k0 \
            + ((3*k*beta*pow(k1,2)) / pow((1 + k2), 2)) * pow(i,3) \
                + i * k * alpha + InputSingal
        Gx = i

        Fx_r = round(Fx, round_Pre)
        Gx_r = round(Gx, round_Pre)
        print("\r%d / %d , %fPercent "%(Counter, (Plot_End - Plot_Start) * dx_R, (Counter / ((Plot_End - Plot_Start) * dx_R)) * 100), end = "")
        Counter += 1

        if pow((Fx_r - Gx_r), 2) == 0:
            print(f"xの値: {i}, Chialvoの値: {Fx_r}")
            FixedPoint_List += [i]
        
    
    Time_End = time.time()
    
    Time_Diff = Time_End - Time_Start
    print("\n\n不動点の値: ")
    print(f"{FixedPoint_List}")
    print(f"実行時間: {Time_Diff}")

#====================================================================
#固定点の導出
if Derive_of_FixedPoint:
    print("\n不動点の導出")

    for i in FixedPoint_List:
        x = i
        y = round((-b * x + c) / (1 - a), round_Pre)
        phi = round((k1 * x) / (1 + k2), round_Pre)

        print(f"不動点: x = {x}, y = {y}, phi = {phi}")
        Jac_Matrix = np.array([[np.exp(y - x) * (2 * x - pow(x, 2)) + k * (alpha + 3 * beta * pow(phi, 2)), pow(x, 2) * np.exp(y - x), 6 * k * x * beta * phi],\
                                [-b, a, 0], \
                                    [k1, 0, -k2]])
        
        Eig_Val= np.linalg.eigvals(Jac_Matrix)
        print(f"固有値 = {Eig_Val}")

#====================================================================
#リアプノフ指数
if Lyapunov_expotent:
    print("\n最大リアプノフ指数の導出")

    Initial_x = None
    Initial_y = None
    Initial_phi = None

    tau = 1

    Input_Signal_In = np.ones(Runtime) * InputSingal

    delta_0 = np.zeros(Runtime)
    delta_tau = np.zeros(Runtime)
    #--------------------------------------------------------------------
    #初期値xについて
    x = np.zeros(Runtime)
    x_L = np.zeros(Runtime)

    if Initial_x == None:
        x[0] = np.random.random() * 2 - 1
    else:
        x[0] = Initial_x

    #--------------------------------------------------------------------
    #初期値yについて
    y = np.zeros(Runtime)
    y_L = np.zeros(Runtime)

    if Initial_y == None:
        y[0] = np.random.random() * 2 - 1
    else:
        y[0] = Initial_y

    #--------------------------------------------------------------------
    #初期値phiについて
    phi = np.zeros(Runtime)
    phi_L = np.zeros(Runtime)

    if Initial_phi == None:
        phi[0] = np.random.random() * 2 - 1
    else:
        phi[0] = Initial_phi

    #--------------------------------------------------------------------
    for k in range(Runtime-1):
        x[k+1] = pow(x[k], 2) * np.exp(y[k] - x[k]) + k0 \
              + k * x[k] * (alpha + 3 * beta * pow(phi[k], 2)) \
                + Input_Signal_In[k]
        y[k+1] = a * y[k] - b * x[k] + c
        phi[k+1] = k1 * x[k] - k2 * phi[k]

    x_L[0] = x[0] + 1e-06
    y_L[0] = y[0] + 1e-06
    phi_L[0] = phi[0] + 1e-06

    x_L[1] = x_L[0] **2

    for i in range(Runtime-1):
        x_L[i] = x[i] + 1e-06
        y_L[i] = y[i] + 1e-06
        phi_L[i] = phi[i] + 1e-06

        x_L[i+1] = pow(x_L[i], 2) * np.exp(y_L[i] - x_L[i]) + k0 \
            + k * x_L[i] * (alpha + 3 * beta * pow(phi_L[i], 2)) \
                + Input_Signal_In[i]
        y_L[i+1] = a * y_L[i] - b * x_L[i] + c
        phi_L[i+1] = k1 * x_L[i] - k2 * phi_L[i]

    
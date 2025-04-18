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

k = -6 #-3.2 -0.5

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#入力信号
InputSingal = 0

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#空走時間
Burn_in_time = 1000

#実行時間
Runtime = 10000

#全体時間
Alltime = Burn_in_time + Runtime

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
        y = round((b * x - c) / (a - 1), round_Pre)
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
    #初期値の指定
    Initial_x = None
    Initial_y = None
    Initial_phi = None

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
    Input_Signal_In = np.ones(Alltime) * InputSingal

    for n in range(Alltime):
        Input_Signal_In[n] = 0.1 * InputSingal * np.cos(4 * n * np.pi / 180)

    #--------------------------------------------------------------------
    #規準軌道の計算部
    
    #空走時間の計算部
    for i in range(Burn_in_time-1):
        print("\r%d / %d"%(i, Burn_in_time), end = "")
        x[i+1] = pow(x[i], 2) * np.exp(y[i] - x[i]) + k0\
              + k * x[i] * (alpha + 3 * beta * pow(phi[i], 2))\
                + Input_Signal_In[i]
        y[i+1] = a * y[i] - b * x[i] + c
        phi[i+1] = k1 * x[i] - k2 * phi[i]

    #計算時間の計算部
    for i in range(Burn_in_time, Alltime -1):
        print("\r%d / %d"%(i, Alltime), end = "")
        x[i+1] = pow(x[i], 2) * np.exp(y[i] - x[i]) + k0\
              + k * x[i] * (alpha + 3 * beta * pow(phi[i], 2))\
                + Input_Signal_In[i]
        y[i+1] = a * y[i] - b * x[i] + c
        phi[i+1] = k1 * x[i] - k2 * phi[i]

    #--------------------------------------------------------------------
    #リアプノフ指数を取る配列
    x_Lyapunov = np.zeros(Runtime+1)
    y_Lyapunov = np.zeros(Runtime+1)
    phi_Lyapunov = np.zeros(Runtime+1)

    #初期摂動が与えられた軌道
    x_D = np.zeros(Runtime+1)
    y_D = np.zeros(Runtime+1)
    phi_D = np.zeros(Runtime+1)

    #総和を取る配列
    x_sum = 0
    y_sum = 0
    phi_sum = 0

    #初期摂動を確保する配列
    delta_0_x = np.zeros(Runtime+1)
    delta_0_y = np.zeros(Runtime+1)
    delta_0_phi = np.zeros(Runtime+1)
    
    delta_0_x[0] = 1e-06
    delta_0_y[0] = 1e-06
    delta_0_phi[0] = 1e-06

    #初期摂動を1step発展させる配列
    delta_1_x = np.zeros(Runtime+1)
    delta_1_y = np.zeros(Runtime+1)
    delta_1_phi = np.zeros(Runtime+1)
    
    delta_1_x[0] = 1e-06
    delta_1_y[0] = 1e-06
    delta_1_phi[0] = 1e-06

    #--------------------------------------------------------------------
    #規準軌道に初期摂動を与える
    x_D[0] = x[Burn_in_time] + delta_0_x[0]
    y_D[0] = y[Burn_in_time] + delta_0_y[0]
    phi_D[0] = phi[Burn_in_time] + delta_0_phi[0]
    
    #初期摂動を与えたものを1step時間発展させた
    x_D[1] = pow(x_D[Burn_in_time], 2) * np.exp(y_D[Burn_in_time] - x_D[Burn_in_time]) + k0 \
              + k * x_D[Burn_in_time] * (alpha + 3 * beta * pow(phi_D[Burn_in_time], 2)) \
                + Input_Signal_In[Burn_in_time]
    y_D[1] = a * y_D[Burn_in_time] - b * x_D[Burn_in_time] + c
    phi_D[1] = k1 * x_D[Burn_in_time] - k2 * phi_D[Burn_in_time]

    #時間発展させた2つの軌道の差の算出
    delta_1_x[0] = x_D[1] - x[Burn_in_time]
    delta_1_y[0] = y_D[1] - y[Burn_in_time]
    delta_1_phi[0] = phi_D[1] - phi[Burn_in_time]

    #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
    #（負ならば差が縮まる、正なら差が広がる）
    x_Lyapunov[0] = np.log(abs(delta_1_x[0]) / abs(delta_0_x[0]))
    x_sum += x_Lyapunov[0]

    y_Lyapunov[0] = np.log(abs(delta_1_y[0]) / abs(delta_0_y[0]))
    y_sum += y_Lyapunov[0]

    phi_Lyapunov[0] = np.log(abs(delta_1_phi[0]) / abs(delta_0_phi[0]))
    phi_sum += phi_Lyapunov[0]
    print("\n")

    z = 1
    #摂動軌道の計算部
    for i in range(Burn_in_time,Alltime-1):
        #初期摂動の値を統一する（向きはそのまま）
        delta_0_x[z] = (delta_1_x[z-1] / abs(delta_1_x[z-1])) * delta_0_x[0]
        delta_0_y[z] = (delta_1_y[z-1] / abs(delta_1_y[z-1])) * delta_0_y[0]
        delta_0_phi[z] = (delta_1_phi[z-1] / abs(delta_1_phi[z-1])) * delta_0_phi[0]

        #規準軌道に初期摂動を与える
        x_D[z] = x[i] + delta_0_x[z]
        y_D[z] = y[i] + delta_0_y[z]
        phi_D[z] = phi[i] + delta_0_phi[z]

        #摂動を与えた軌道を1step分計算する
        x_D[z+1] = pow(x_D[z], 2) * np.exp(y_D[z] - x_D[z]) + k0\
              + k * x_D[z] * (alpha + 3 * beta * pow(phi_D[z], 2))\
                + Input_Signal_In[z]
        y_D[z+1] = a * y_D[z] - b * x_D[z] + c
        phi_D[z+1] = k1 * x_D[z] - k2 * phi_D[z]

        #2つの軌道の差
        delta_1_x[z] = x_D[z+1] - x[i+1]
        delta_1_y[z] = y_D[z+1] - y[i+1]
        delta_1_phi[z] = phi_D[z+1] - phi[i+1]

        x_Lyapunov[z] = np.log(abs(delta_1_x[z]) / abs(delta_0_x[z]))
        y_Lyapunov[z] = np.log(abs(delta_1_y[z]) / abs(delta_0_y[z]))
        phi_Lyapunov[z] = np.log(abs(delta_1_phi[z]) / abs(delta_0_phi[z]))

        x_sum += x_Lyapunov[z]
        y_sum += y_Lyapunov[z]
        phi_sum += phi_Lyapunov[z]

        z += 1

    x_ave = x_sum / len(x_Lyapunov)
    print("xのリアプノフ指数")
    print(x_ave)
    
    y_ave = y_sum / len(y_Lyapunov)
    print("\nyのリアプノフ指数")
    print(y_ave)
    
    phi_ave = phi_sum / len(phi_Lyapunov)
    print("\nphiのリアプノフ指数")
    print(phi_ave)

    l = []
    l.append(x_ave)
    l.append(y_ave)
    l.append(phi_ave)

    print("\n最大リアプノフ指数")
    print(max(l))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(x_Lyapunov[0:500])
    ax.plot(y_Lyapunov[0:500])
    ax.plot(phi_Lyapunov[0:500])
    plt.show()

    #ax.plot(x_D[0:500])
    #plt.show()
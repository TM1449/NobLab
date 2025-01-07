#====================================================================
import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
import math


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

k_list = np.arange(-7, 7.1, 0.1)
k_list_result = np.zeros(len(k_list))
k_list_result_M = np.zeros((len(k_list), 10))

#print(k_list_result)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#入力信号
InputSingal = 3

#None, sin, cos（ローレンツ方程式）
InputSingal_def = None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#空走時間
Burn_in_time = 1000

#実行時間
Runtime = 10000

#全体時間
Alltime = Burn_in_time + Runtime

#リアプノフ指数を1回求める
Lyapunov_expotent = True

#リアプノフ指数を10回求め、平均を取る
Lyapunov_expotent_ten = False

#リストの分＋リアプノフ指数10回平均＋グラフに図示
Lyapunov_expotent_ten_List = False

#====================================================================
#リアプノフ指数を1回求める
#1step後の2つの軌道の差のプロット
if Lyapunov_expotent:

    print("\n電磁束下Chialvoニューロン: 最大リアプノフ指数の導出")
    #初期値の指定
    Initial_x = None #0.0448
    Initial_y = None #2.3011
    Initial_phi = None #0.0037

    #--------------------------------------------------------------------
    #基準軌道xと初期値について
    x = np.zeros(Alltime)

    if Initial_x == None:
        x[0] = np.random.random() * 0.2 - 0.1
    else:
        x[0] = Initial_x

    #--------------------------------------------------------------------
    #基準軌道yと初期値について
    y = np.zeros(Alltime)

    if Initial_y == None:
        y[0] = np.random.random() * 0.2 - 0.1
    else:
        y[0] = Initial_y

    #--------------------------------------------------------------------
    #基準軌道phiと初期値について
    phi = np.zeros(Alltime)

    if Initial_phi == None:
        phi[0] = np.random.random() * 0.2 - 0.1
    else:
        phi[0] = Initial_phi

    #--------------------------------------------------------------------
    #入力信号の作成
    Input_Signal_In = np.ones(Alltime) * InputSingal

    if InputSingal_def == None:
        pass

    elif InputSingal_def == np.sin:
        for n in range(Alltime):
            Input_Signal_In[n] = 0.1 * InputSingal * np.cos(4 * n * np.pi / 180)

    elif InputSingal_def == np.cos:
        #微小時間
        dt = 0.01
        #信号の大きさ
        scale = 1 / 5000 * InputSingal

        #パラメータ
        sigma = 10
        rho = 28
        beta = 8/3

        #入力信号の長さ
        rorentu_x = np.ones(Alltime)
        rorentu_x[0] = np.random.random() * 0.0001
        
        rorentu_y = np.ones(Alltime)
        rorentu_y[0] = np.random.random() * 0.0001
        
        rorentu_z = np.ones(Alltime)
        rorentu_z[0] = np.random.random() * 0.0001

        Input_Signal_In[0] = rorentu_x[0]

        for n in range(Alltime-1):

            rorentu_x[n+1] = dt * (sigma *(rorentu_y[n] - rorentu_x[n])) + rorentu_x[n]
            rorentu_y[n+1] = dt * (rorentu_x[n] * (rho - rorentu_z[n]) - rorentu_y[n]) + rorentu_y[n]
            rorentu_z[n+1] = dt * (rorentu_x[n] * rorentu_y[n] - beta * rorentu_z[n]) + rorentu_z[n]

            Input_Signal_In[n+1] = scale * rorentu_x[n+1]

    #入力信号のプロット
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Input_Signal_In)

    plt.show()
    #--------------------------------------------------------------------
    #規準軌道の計算部
    print("\n基準軌道の計算")
    for i in range(Alltime-1):
        print("\r%d / %d"%(i, Alltime), end = "")
        x[i+1] = pow(x[i], 2) * np.exp(y[i] - x[i]) + k0\
              + k * x[i] * (alpha + 3 * beta * pow(phi[i], 2))\
                + Input_Signal_In[i]
        y[i+1] = a * y[i] - b * x[i] + c
        phi[i+1] = k1 * x[i] - k2 * phi[i]

    #--------------------------------------------------------------------
    #リアプノフ指数を取る配列
    x_Lyapunov = np.zeros(Runtime-1)
    y_Lyapunov = np.zeros(Runtime-1)
    phi_Lyapunov = np.zeros(Runtime-1)

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
    x_D[0] = x[Burn_in_time] + delta_0_x[0]
    y_D[0] = y[Burn_in_time] + delta_0_y[0]
    phi_D[0] = phi[Burn_in_time] + delta_0_phi[0]
    
    #初期摂動を与えたものを1step時間発展させた
    x_D[1] = pow(x_D[0], 2) * np.exp(y_D[0] - x_D[0]) + k0 \
              + k * x_D[0] * (alpha + 3 * beta * pow(phi_D[0], 2)) \
                + Input_Signal_In[Burn_in_time]
    y_D[1] = a * y_D[0] - b * x_D[0] + c
    phi_D[1] = k1 * x_D[0] - k2 * phi_D[0]

    #時間発展させた2つの軌道の差の算出
    delta_tau_x[0] = x_D[1] - x[Burn_in_time+1]
    delta_tau_y[0] = y_D[1] - y[Burn_in_time+1]
    delta_tau_phi[0] = phi_D[1] - phi[Burn_in_time+1]

    #軌道の差のプロット
    """fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x[Burn_in_time:Burn_in_time+2])
    ax.plot(x_D[0:2])
    plt.show()"""

    #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
    #（負ならば差が縮まる、正なら差が広がる）
    x_Lyapunov[0] = np.log(abs(delta_tau_x[0]) / abs(delta_0_x[0]))
    x_sum += x_Lyapunov[0]

    y_Lyapunov[0] = np.log(abs(delta_tau_y[0]) / abs(delta_0_y[0]))
    y_sum += y_Lyapunov[0]

    phi_Lyapunov[0] = np.log(abs(delta_tau_phi[0]) / abs(delta_0_phi[0]))
    phi_sum += phi_Lyapunov[0]

    #摂動軌道の計算部
    print("\n\n摂動軌道の計算")
    for i in range(1,Runtime-1):
        print("\r%d / %d"%(i, Runtime), end = "")

        delta_0_x[i] = (delta_tau_x[i-1] / abs(delta_tau_x[i-1])) * delta_0_x[0]
        delta_0_y[i] = (delta_tau_y[i-1] / abs(delta_tau_y[i-1])) * delta_0_y[0]
        delta_0_phi[i] = (delta_tau_phi[i-1] / abs(delta_tau_phi[i-1])) * delta_0_phi[0]

        #規準軌道に初期摂動を与える
        x_D[i] = x[Burn_in_time+i] + delta_0_x[i]
        y_D[i] = y[Burn_in_time+i] + delta_0_y[i]
        phi_D[i] = phi[Burn_in_time+i] + delta_0_phi[i]
        
        #初期摂動を与えたものを1step時間発展させた
        x_D[i+1] = pow(x_D[i], 2) * np.exp(y_D[i] - x_D[i]) + k0 \
                + k * x_D[i] * (alpha + 3 * beta * pow(phi_D[i], 2)) \
                    + Input_Signal_In[Burn_in_time+i]
        y_D[i+1] = a * y_D[i] - b * x_D[i] + c
        phi_D[i+1] = k1 * x_D[i] - k2 * phi_D[i]

        #時間発展させた2つの軌道の差の算出
        delta_tau_x[i] = x_D[i+1] - x[Burn_in_time+i+1]
        delta_tau_y[i] = y_D[i+1] - y[Burn_in_time+i+1]
        delta_tau_phi[i] = phi_D[i+1] - phi[Burn_in_time+i+1]

        #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
        #（負ならば差が縮まる、正なら差が広がる）
        x_Lyapunov[i] = np.log(abs(delta_tau_x[i]) / abs(delta_0_x[i]))
        x_sum += x_Lyapunov[i]

        y_Lyapunov[i] = np.log(abs(delta_tau_y[i]) / abs(delta_0_y[i]))
        y_sum += y_Lyapunov[i]

        phi_Lyapunov[i] = np.log(abs(delta_tau_phi[i]) / abs(delta_0_phi[i]))
        phi_sum += phi_Lyapunov[i]

    #時間平均でのリアプノフ指数
    x_ave = x_sum / len(x_Lyapunov)
    y_ave = y_sum / len(y_Lyapunov)
    phi_ave = phi_sum / len(phi_Lyapunov)

    Lyapunov_List = []
    Lyapunov_List.append(x_ave)
    Lyapunov_List.append(y_ave)
    Lyapunov_List.append(phi_ave)
    
    print("\n")
    print({f"xのリアプノフ指数: {x_ave}"})
    print({f"yのリアプノフ指数: {y_ave}"})
    print({f"phiのリアプノフ指数: {phi_ave}"})
    print("\n最大リアプノフ指数")
    print(max(Lyapunov_List))

    #リアプノフ指数の推移
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_Lyapunov[1000:1500])
    plt.show()

    #xのプロット
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x[Burn_in_time:])
    plt.show()

#====================================================================
#リアプノフ指数を10回求め、平均を取る
if Lyapunov_expotent_ten:
    l = 0
    break_count = 0

    Lyapunov_Max = 0
    Lyapunov_Max_List = []
    
    print("\n電磁束下Chialvoニューロン: 10回の平均最大リアプノフ指数の導出")
    while l < 10:

        print(f"\n試行回数: {l}")
        #初期値の指定
        Initial_x = None #0.0448
        Initial_y = None #2.3011
        Initial_phi = None #0.0037

        #--------------------------------------------------------------------
        #基準軌道xと初期値について
        x = np.zeros(Alltime)

        if Initial_x == None:
            x[0] = np.random.random() * 0.2 - 0.1
        else:
            x[0] = Initial_x

        #--------------------------------------------------------------------
        #基準軌道yと初期値について
        y = np.zeros(Alltime)

        if Initial_y == None:
            y[0] = np.random.random() * 0.2 - 0.1
        else:
            y[0] = Initial_y

        #--------------------------------------------------------------------
        #基準軌道phiと初期値について
        phi = np.zeros(Alltime)

        if Initial_phi == None:
            phi[0] = np.random.random() * 0.2 - 0.1
        else:
            phi[0] = Initial_phi

        #--------------------------------------------------------------------
        #入力信号の作成
        Input_Signal_In = np.ones(Alltime) * InputSingal

        if InputSingal_def == None:
            pass

        elif InputSingal_def == np.sin:
            for n in range(Alltime):
                Input_Signal_In[n] = 0.1 * InputSingal * np.cos(2 * n * np.pi / 180)

        elif InputSingal_def == np.cos:
            #微小時間
            dt = 0.01
            #信号の大きさ
            scale = 1 / 5000 * InputSingal

            #パラメータ
            sigma = 10
            rho = 28
            beta = 8/3

            #入力信号の長さ
            rorentu_x = np.ones(Alltime)
            rorentu_x[0] = np.random.random() * 0.0001
            
            rorentu_y = np.ones(Alltime)
            rorentu_y[0] = np.random.random() * 0.0001
            
            rorentu_z = np.ones(Alltime)
            rorentu_z[0] = np.random.random() * 0.0001

            Input_Signal_In[0] = rorentu_x[0]

            for n in range(Alltime-1):

                rorentu_x[n+1] = dt * (sigma *(rorentu_y[n] - rorentu_x[n])) + rorentu_x[n]
                rorentu_y[n+1] = dt * (rorentu_x[n] * (rho - rorentu_z[n]) - rorentu_y[n]) + rorentu_y[n]
                rorentu_z[n+1] = dt * (rorentu_x[n] * rorentu_y[n] - beta * rorentu_z[n]) + rorentu_z[n]

                Input_Signal_In[n+1] = scale * rorentu_x[n+1]
        #--------------------------------------------------------------------
        #規準軌道の計算部
        print("\n基準軌道の計算")
        for i in range(Alltime-1):
            print("\r%d / %d"%(i, Alltime), end = "")
            x[i+1] = pow(x[i], 2) * np.exp(y[i] - x[i]) + k0\
                + k * x[i] * (alpha + 3 * beta * pow(phi[i], 2))\
                    + Input_Signal_In[i]
            y[i+1] = a * y[i] - b * x[i] + c
            phi[i+1] = k1 * x[i] - k2 * phi[i]

        #--------------------------------------------------------------------
        #リアプノフ指数を取る配列
        x_Lyapunov = np.zeros(Runtime-1)
        y_Lyapunov = np.zeros(Runtime-1)
        phi_Lyapunov = np.zeros(Runtime-1)

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
        x_D[0] = x[Burn_in_time] + delta_0_x[0]
        y_D[0] = y[Burn_in_time] + delta_0_y[0]
        phi_D[0] = phi[Burn_in_time] + delta_0_phi[0]
        
        #初期摂動を与えたものを1step時間発展させた
        x_D[1] = pow(x_D[0], 2) * np.exp(y_D[0] - x_D[0]) + k0 \
                + k * x_D[0] * (alpha + 3 * beta * pow(phi_D[0], 2)) \
                    + Input_Signal_In[Burn_in_time]
        y_D[1] = a * y_D[0] - b * x_D[0] + c
        phi_D[1] = k1 * x_D[0] - k2 * phi_D[0]

        #時間発展させた2つの軌道の差の算出
        delta_tau_x[0] = x_D[1] - x[Burn_in_time+1]
        delta_tau_y[0] = y_D[1] - y[Burn_in_time+1]
        delta_tau_phi[0] = phi_D[1] - phi[Burn_in_time+1]

        #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
        #（負ならば差が縮まる、正なら差が広がる）
        x_Lyapunov[0] = np.log(abs(delta_tau_x[0]) / abs(delta_0_x[0]))
        x_sum += x_Lyapunov[0]

        y_Lyapunov[0] = np.log(abs(delta_tau_y[0]) / abs(delta_0_y[0]))
        y_sum += y_Lyapunov[0]

        phi_Lyapunov[0] = np.log(abs(delta_tau_phi[0]) / abs(delta_0_phi[0]))
        phi_sum += phi_Lyapunov[0]

        #摂動軌道の計算部
        print("\n\n摂動軌道の計算")
        for i in range(1,Runtime-1):
            print("\r%d / %d"%(i, Runtime), end = "")

            delta_0_x[i] = (delta_tau_x[i-1] / abs(delta_tau_x[i-1])) * delta_0_x[0]
            delta_0_y[i] = (delta_tau_y[i-1] / abs(delta_tau_y[i-1])) * delta_0_y[0]
            delta_0_phi[i] = (delta_tau_phi[i-1] / abs(delta_tau_phi[i-1])) * delta_0_phi[0]

            #規準軌道に初期摂動を与える
            x_D[i] = x[Burn_in_time+i] + delta_0_x[i]
            y_D[i] = y[Burn_in_time+i] + delta_0_y[i]
            phi_D[i] = phi[Burn_in_time+i] + delta_0_phi[i]
            
            #初期摂動を与えたものを1step時間発展させた
            x_D[i+1] = pow(x_D[i], 2) * np.exp(y_D[i] - x_D[i]) + k0 \
                    + k * x_D[i] * (alpha + 3 * beta * pow(phi_D[i], 2)) \
                        + Input_Signal_In[Burn_in_time+i]
            y_D[i+1] = a * y_D[i] - b * x_D[i] + c
            phi_D[i+1] = k1 * x_D[i] - k2 * phi_D[i]

            #時間発展させた2つの軌道の差の算出
            delta_tau_x[i] = x_D[i+1] - x[Burn_in_time+i+1]
            delta_tau_y[i] = y_D[i+1] - y[Burn_in_time+i+1]
            delta_tau_phi[i] = phi_D[i+1] - phi[Burn_in_time+i+1]

            #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
            #（負ならば差が縮まる、正なら差が広がる）
            x_Lyapunov[i] = np.log(abs(delta_tau_x[i]) / abs(delta_0_x[i]))
            x_sum += x_Lyapunov[i]

            y_Lyapunov[i] = np.log(abs(delta_tau_y[i]) / abs(delta_0_y[i]))
            y_sum += y_Lyapunov[i]

            phi_Lyapunov[i] = np.log(abs(delta_tau_phi[i]) / abs(delta_0_phi[i]))
            phi_sum += phi_Lyapunov[i]


        if math.isnan(x_sum):
            print(f"リアプノフ指数が{x_sum}となったため、除外して再度実行")
            break_count = break_count + 1
            print(f"Break_count: {break_count}")

            if break_count < 10:
                pass
            else:
                break

        else:
            #各実行時のリアプノフ指数の算出
            x_ave = x_sum / len(x_Lyapunov)
            y_ave = y_sum / len(y_Lyapunov)
            phi_ave = phi_sum / len(phi_Lyapunov)

            #各実行時の最大リアプノフ指数の算出
            Lyapunov_List = []
            Lyapunov_List.append(x_ave)
            Lyapunov_List.append(y_ave)
            Lyapunov_List.append(phi_ave)
            print(Lyapunov_List)

            Lyapunov_Max_List.append(max(Lyapunov_List))

            Lyapunov_Max = Lyapunov_Max + max(Lyapunov_List)
            
            l = l + 1

    #10回実行したときの最大リアプノフ指数の平均
    Lyapunov_Max_ave = Lyapunov_Max / 10
    print("\n")
    print(f"最大リアプノフ指数: {Lyapunov_Max_ave}")
    print(f"{Lyapunov_Max_List}")

    #入力信号のプロット
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Input_Signal_In)

    plt.show()

#====================================================================
#リストの分＋リアプノフ指数10回平均＋グラフに図示
if Lyapunov_expotent_ten_List:
    count = 0
    print("\n電磁束下Chialvoニューロン: 10回の平均最大リアプノフ指数の導出")

    for k in k_list:
        print({f"{count+1} / {len(k_list_result)} 回"})
        print(f"パラメータkの値: {k}")

        l = 0
        break_count = 0
        Lyapunov_Max = 0

        while l < 10:

            print(f"\n試行回数: {l}")
            #初期値の指定
            Initial_x = None #0.0448
            Initial_y = None #2.3011
            Initial_phi = None #0.0037

            #--------------------------------------------------------------------
            #基準軌道xと初期値について
            x = np.zeros(Alltime)

            if Initial_x == None:
                x[0] = np.random.random() * 0.2 - 0.1
            else:
                x[0] = Initial_x

            #--------------------------------------------------------------------
            #基準軌道yと初期値について
            y = np.zeros(Alltime)

            if Initial_y == None:
                y[0] = np.random.random() * 0.2 - 0.1
            else:
                y[0] = Initial_y

            #--------------------------------------------------------------------
            #基準軌道phiと初期値について
            phi = np.zeros(Alltime)

            if Initial_phi == None:
                phi[0] = np.random.random() * 0.2 - 0.1
            else:
                phi[0] = Initial_phi

            #--------------------------------------------------------------------
            #入力信号の作成
            Input_Signal_In = np.ones(Alltime) * InputSingal

            if InputSingal_def == None:
                pass

            elif InputSingal_def == np.sin:
                for n in range(Alltime):
                    Input_Signal_In[n] = 0.1 * InputSingal * np.cos(4 * n * np.pi / 180)

            elif InputSingal_def == np.cos:
                #微小時間
                dt = 0.01
                #信号の大きさ
                scale = 1 / 5000 * InputSingal

                #パラメータ
                sigma = 10
                rho = 28
                beta = 8/3

                #入力信号の長さ
                rorentu_x = np.ones(Alltime)
                rorentu_x[0] = np.random.random() * 0.0001
                
                rorentu_y = np.ones(Alltime)
                rorentu_y[0] = np.random.random() * 0.0001
                
                rorentu_z = np.ones(Alltime)
                rorentu_z[0] = np.random.random() * 0.0001

                Input_Signal_In[0] = rorentu_x[0]

                for n in range(Alltime-1):

                    rorentu_x[n+1] = dt * (sigma *(rorentu_y[n] - rorentu_x[n])) + rorentu_x[n]
                    rorentu_y[n+1] = dt * (rorentu_x[n] * (rho - rorentu_z[n]) - rorentu_y[n]) + rorentu_y[n]
                    rorentu_z[n+1] = dt * (rorentu_x[n] * rorentu_y[n] - beta * rorentu_z[n]) + rorentu_z[n]

                    Input_Signal_In[n+1] = scale * rorentu_x[n+1]

            #--------------------------------------------------------------------
            #規準軌道の計算部
            #print("\n基準軌道の計算")
            for i in range(Alltime-1):
                #print("\r%d / %d"%(i, Alltime), end = "")
                x[i+1] = pow(x[i], 2) * np.exp(y[i] - x[i]) + k0\
                    + k * x[i] * (alpha + 3 * beta * pow(phi[i], 2))\
                        + Input_Signal_In[i]
                y[i+1] = a * y[i] - b * x[i] + c
                phi[i+1] = k1 * x[i] - k2 * phi[i]

            #--------------------------------------------------------------------
            #リアプノフ指数を取る配列
            x_Lyapunov = np.zeros(Runtime-1)
            y_Lyapunov = np.zeros(Runtime-1)
            phi_Lyapunov = np.zeros(Runtime-1)

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
            x_D[0] = x[Burn_in_time] + delta_0_x[0]
            y_D[0] = y[Burn_in_time] + delta_0_y[0]
            phi_D[0] = phi[Burn_in_time] + delta_0_phi[0]
            
            #初期摂動を与えたものを1step時間発展させた
            x_D[1] = pow(x_D[0], 2) * np.exp(y_D[0] - x_D[0]) + k0 \
                    + k * x_D[0] * (alpha + 3 * beta * pow(phi_D[0], 2)) \
                        + Input_Signal_In[Burn_in_time]
            y_D[1] = a * y_D[0] - b * x_D[0] + c
            phi_D[1] = k1 * x_D[0] - k2 * phi_D[0]

            #時間発展させた2つの軌道の差の算出
            delta_tau_x[0] = x_D[1] - x[Burn_in_time+1]
            delta_tau_y[0] = y_D[1] - y[Burn_in_time+1]
            delta_tau_phi[0] = phi_D[1] - phi[Burn_in_time+1]

            #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
            #（負ならば差が縮まる、正なら差が広がる）
            x_Lyapunov[0] = np.log(abs(delta_tau_x[0]) / abs(delta_0_x[0]))
            x_sum += x_Lyapunov[0]

            y_Lyapunov[0] = np.log(abs(delta_tau_y[0]) / abs(delta_0_y[0]))
            y_sum += y_Lyapunov[0]

            phi_Lyapunov[0] = np.log(abs(delta_tau_phi[0]) / abs(delta_0_phi[0]))
            phi_sum += phi_Lyapunov[0]

            #摂動軌道の計算部
            #print("\n\n摂動軌道の計算")
            for i in range(1,Runtime-1):
                #print("\r%d / %d"%(i, Runtime), end = "")

                delta_0_x[i] = (delta_tau_x[i-1] / abs(delta_tau_x[i-1])) * delta_0_x[0]
                delta_0_y[i] = (delta_tau_y[i-1] / abs(delta_tau_y[i-1])) * delta_0_y[0]
                delta_0_phi[i] = (delta_tau_phi[i-1] / abs(delta_tau_phi[i-1])) * delta_0_phi[0]

                #規準軌道に初期摂動を与える
                x_D[i] = x[Burn_in_time+i] + delta_0_x[i]
                y_D[i] = y[Burn_in_time+i] + delta_0_y[i]
                phi_D[i] = phi[Burn_in_time+i] + delta_0_phi[i]
                
                #初期摂動を与えたものを1step時間発展させた
                x_D[i+1] = pow(x_D[i], 2) * np.exp(y_D[i] - x_D[i]) + k0 \
                        + k * x_D[i] * (alpha + 3 * beta * pow(phi_D[i], 2)) \
                            + Input_Signal_In[Burn_in_time+i]
                y_D[i+1] = a * y_D[i] - b * x_D[i] + c
                phi_D[i+1] = k1 * x_D[i] - k2 * phi_D[i]

                #時間発展させた2つの軌道の差の算出
                delta_tau_x[i] = x_D[i+1] - x[Burn_in_time+i+1]
                delta_tau_y[i] = y_D[i+1] - y[Burn_in_time+i+1]
                delta_tau_phi[i] = phi_D[i+1] - phi[Burn_in_time+i+1]

                #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
                #（負ならば差が縮まる、正なら差が広がる）
                x_Lyapunov[i] = np.log(abs(delta_tau_x[i]) / abs(delta_0_x[i]))
                x_sum += x_Lyapunov[i]

                y_Lyapunov[i] = np.log(abs(delta_tau_y[i]) / abs(delta_0_y[i]))
                y_sum += y_Lyapunov[i]

                phi_Lyapunov[i] = np.log(abs(delta_tau_phi[i]) / abs(delta_0_phi[i]))
                phi_sum += phi_Lyapunov[i]

            if math.isnan(x_sum):
                #print(f"リアプノフ指数が{x_sum}となったため、除外して再度実行")
                break_count = break_count + 1
                print(f"Break_count: {break_count}")

                if break_count < 20:
                    pass
                else:
                    break

            else:
                #各実行時のリアプノフ指数の算出
                x_ave = x_sum / len(x_Lyapunov)
                y_ave = y_sum / len(y_Lyapunov)
                phi_ave = phi_sum / len(phi_Lyapunov)

                #各実行時の最大リアプノフ指数の算出
                Lyapunov_List = []
                Lyapunov_List.append(x_ave)
                Lyapunov_List.append(y_ave)
                Lyapunov_List.append(phi_ave)
                print(Lyapunov_List)

                #Lyapunov_Max_List.append(max(Lyapunov_List))
                k_list_result_M[count, l] = max(Lyapunov_List)

                Lyapunov_Max = Lyapunov_Max + max(Lyapunov_List)
                
                l = l + 1

        if break_count == 20:
            k_list_result[count] = None

        else:
            #10回実行したときの最大リアプノフ指数の平均
            Lyapunov_Max_ave = Lyapunov_Max / 10
            #print("\n")
            print(f"最大リアプノフ指数: {Lyapunov_Max_ave}")
            k_list_result[count] = Lyapunov_Max_ave

        count = count + 1

    #入力信号のプロット
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Input_Signal_In)

    plt.show()
    
    #各パラメータによるリアプノフ指数の推移
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.grid()
    ax.set_xticks(np.arange(-7, 7.1, 0.5))
    ax.plot(k_list,k_list_result)
    
    plt.show()


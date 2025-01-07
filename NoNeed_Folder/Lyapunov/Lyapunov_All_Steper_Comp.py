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

k = -1.1 #-3.2 -0.5

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

#何ステップ後の差を計算するか
Step = 10

#全体時間
Alltime = Burn_in_time + Runtime

#リアプノフ指数を1回求める
Lyapunov_expotent = True

#リアプノフ指数を10回求め、平均を取る
#ステップ数の実装だめ
Lyapunov_expotent_ten = False

#リストの分＋リアプノフ指数10回平均＋グラフに図示
#ステップ数の実装だめ
Lyapunov_expotent_ten_List = False

"""Now = datetime.datetime.now()
ProjectFile_Path = "Data/SimpleEvaluations"
Figure_Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
Figure_Name = "MaxLyapunovCheck_" + Figure_Date + "_"
Result_Path = os.path.join(ProjectFile_Path, "Results")
UFigure_Name = "MaxLyapunovCheck_Lorenz_" + Figure_Date
UFigure_Path = os.path.join(Result_Path, UFigure_Name)

# ディレクトリの作成
os.makedirs(UFigure_Path, exist_ok=True)"""

# 描画設定
FigSize = (16, 9)
FontSize_Label = 24
FontSize_Title = 24
LineWidth = 3
FileFormat = ".png"

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
    x = np.zeros(Alltime + (Step - 1))

    if Initial_x == None:
        x[0] = np.random.random() * 2 - 1
    else:
        x[0] = Initial_x

    #--------------------------------------------------------------------
    #基準軌道yと初期値について
    y = np.zeros(Alltime + (Step - 1))

    if Initial_y == None:
        y[0] = np.random.random() * 2 - 1
    else:
        y[0] = Initial_y

    #--------------------------------------------------------------------
    #基準軌道phiと初期値について
    phi = np.zeros(Alltime + (Step - 1))

    if Initial_phi == None:
        phi[0] = np.random.random() * 2 - 1
    else:
        phi[0] = Initial_phi

    #--------------------------------------------------------------------
    #入力信号の作成
    Input_Signal_In = np.ones(Alltime + (Step - 1)) * InputSingal

    if InputSingal_def == None:
        pass

    elif InputSingal_def == np.sin:
        for n in range(Alltime + (Step - 1)):
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
        rorentu_x = np.ones(Alltime + (Step - 1))
        rorentu_x[0] = np.random.random() * 0.0001
        
        rorentu_y = np.ones(Alltime + (Step - 1))
        rorentu_y[0] = np.random.random() * 0.0001
        
        rorentu_z = np.ones(Alltime + (Step - 1))
        rorentu_z[0] = np.random.random() * 0.0001

        Input_Signal_In[0] = rorentu_x[0]

        for n in range(Alltime + (Step - 1) - 1):

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
    for i in range(Alltime + (Step - 1) - 1):
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
    x_D = np.zeros(Runtime - 1 + Step)
    y_D = np.zeros(Runtime - 1 + Step)
    phi_D = np.zeros(Runtime -1 + Step)

    #総和を取る配列
    x_sum = 0
    y_sum = 0
    phi_sum = 0

    #初期摂動を確保する配列
    delta_0_x = np.zeros(Runtime)
    delta_0_y = np.zeros(Runtime)
    delta_0_phi = np.zeros(Runtime)
    
    delta_0_x[0] = 1e-08
    delta_0_y[0] = 1e-08
    delta_0_phi[0] = 1e-08

    #初期摂動を1step発展させる配列
    delta_tau_x = np.zeros(Runtime)
    delta_tau_y = np.zeros(Runtime)
    delta_tau_phi = np.zeros(Runtime)
    
    delta_tau_x[0] = 1e-08
    delta_tau_y[0] = 1e-08
    delta_tau_phi[0] = 1e-08

    #--------------------------------------------------------------------
    #規準軌道に初期摂動を与える
    x_D[0] = x[Burn_in_time] + delta_0_x[0]
    y_D[0] = y[Burn_in_time] + delta_0_y[0]
    phi_D[0] = phi[Burn_in_time] + delta_0_phi[0]
    
    #初期摂動を与えたものを1step時間発展させた
    for z in range(1, Step + 1):
        x_D[z] = pow(x_D[z-1], 2) * np.exp(y_D[z-1] - x_D[z-1]) + k0 \
                + k * x_D[z-1] * (alpha + 3 * beta * pow(phi_D[z-1], 2)) \
                    + Input_Signal_In[Burn_in_time + z - 1]
        y_D[z] = a * y_D[z-1] - b * x_D[z-1] + c
        phi_D[z] = k1 * x_D[z-1] - k2 * phi_D[z-1]

    #時間発展させた2つの軌道の差の算出
    delta_tau_x[0] = x_D[Step] - x[Burn_in_time+Step]

    #軌道の差
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x[Burn_in_time:Burn_in_time+Step+1])
    ax.plot(x_D[0:Step+1])
    plt.show()

    #時間発展させた2つの軌道の差の算出
    delta_tau_y[0] = y_D[Step] - y[Burn_in_time+Step]

    #軌道の差
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(y[Burn_in_time:Burn_in_time+Step+1])
    ax.plot(y_D[0:Step+1])
    plt.show()

    #時間発展させた2つの軌道の差の算出
    delta_tau_phi[0] = phi_D[Step] - phi[Burn_in_time+Step]
    
    #軌道の差
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(phi[Burn_in_time:Burn_in_time+Step+1])
    ax.plot(phi_D[0:Step+1])
    plt.show()

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
        
        for zz in range(1, Step + 1):
            #初期摂動を与えたものをstep時間発展させた
            x_D[i+zz] = pow(x_D[i+zz-1], 2) * np.exp(y_D[i+zz-1] - x_D[i+zz-1]) + k0 \
                    + k * x_D[i+zz-1] * (alpha + 3 * beta * pow(phi_D[i+zz-1], 2)) \
                        + Input_Signal_In[Burn_in_time+i+zz-1]
            y_D[i+zz] = a * y_D[i+zz-1] - b * x_D[i+zz-1] + c
            phi_D[i+zz] = k1 * x_D[i+zz-1] - k2 * phi_D[i+zz-1]

        #時間発展させた2つの軌道の差の算出
        delta_tau_x[i] = x_D[i+Step] - x[Burn_in_time+i+Step]
        delta_tau_y[i] = y_D[i+Step] - y[Burn_in_time+i+Step]
        delta_tau_phi[i] = phi_D[i+Step] - phi[Burn_in_time+i+Step]

        #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
        #（負ならば差が縮まる、正なら差が広がる）
        x_Lyapunov[i] = np.log(abs(delta_tau_x[i]) / abs(delta_0_x[i]))
        x_sum += x_Lyapunov[i]

        y_Lyapunov[i] = np.log(abs(delta_tau_y[i]) / abs(delta_0_y[i]))
        y_sum += y_Lyapunov[i]

        phi_Lyapunov[i] = np.log(abs(delta_tau_phi[i]) / abs(delta_0_phi[i]))
        phi_sum += phi_Lyapunov[i]

    #時間平均でのリアプノフ指数
    x_ave = (1 / Step) * (x_sum / len(x_Lyapunov))
    y_ave = (1 / Step) * (y_sum / len(y_Lyapunov))
    phi_ave = (1 / Step) * (phi_sum / len(phi_Lyapunov))
    
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
    """fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_D[Runtime-10:])
    plt.show()"""

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
        x = np.zeros(Alltime + (Step - 1))

        if Initial_x == None:
            x[0] = np.random.random() * 0.2 - 0.1
        else:
            x[0] = Initial_x

        #--------------------------------------------------------------------
        #基準軌道yと初期値について
        y = np.zeros(Alltime + (Step - 1))

        if Initial_y == None:
            y[0] = np.random.random() * 0.2 - 0.1
        else:
            y[0] = Initial_y

        #--------------------------------------------------------------------
        #基準軌道phiと初期値について
        phi = np.zeros(Alltime + (Step - 1))

        if Initial_phi == None:
            phi[0] = np.random.random() * 0.2 - 0.1
        else:
            phi[0] = Initial_phi

        #--------------------------------------------------------------------
        #入力信号の作成
        Input_Signal_In = np.ones(Alltime + (Step - 1)) * InputSingal

        if InputSingal_def == None:
            pass

        elif InputSingal_def == np.sin:
            for n in range(Alltime + (Step - 1)):
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
            rorentu_x = np.ones(Alltime + (Step - 1))
            rorentu_x[0] = np.random.random() * 0.0001
            
            rorentu_y = np.ones(Alltime + (Step - 1))
            rorentu_y[0] = np.random.random() * 0.0001
            
            rorentu_z = np.ones(Alltime + (Step - 1))
            rorentu_z[0] = np.random.random() * 0.0001

            Input_Signal_In[0] = rorentu_x[0]

            for n in range(Alltime + (Step - 1) - 1):

                rorentu_x[n+1] = dt * (sigma *(rorentu_y[n] - rorentu_x[n])) + rorentu_x[n]
                rorentu_y[n+1] = dt * (rorentu_x[n] * (rho - rorentu_z[n]) - rorentu_y[n]) + rorentu_y[n]
                rorentu_z[n+1] = dt * (rorentu_x[n] * rorentu_y[n] - beta * rorentu_z[n]) + rorentu_z[n]

                Input_Signal_In[n+1] = scale * rorentu_x[n+1]
        #--------------------------------------------------------------------
        #規準軌道の計算部
        print("\n基準軌道の計算")
        for i in range(Alltime + (Step - 1) - 1):
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
        x_D = np.zeros(Runtime + (Step - 1))
        y_D = np.zeros(Runtime + (Step - 1))
        phi_D = np.zeros(Runtime + (Step - 1))

        #総和を取る配列
        x_sum = 0
        y_sum = 0
        phi_sum = 0

        #初期摂動を確保する配列
        delta_0_x = np.zeros(Runtime + (Step - 1))
        delta_0_y = np.zeros(Runtime + (Step - 1))
        delta_0_phi = np.zeros(Runtime + (Step - 1))
        
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
        for z in range(1, Step + 1):
            x_D[z] = pow(x_D[z-1], 2) * np.exp(y_D[z-1] - x_D[z-1]) + k0 \
                    + k * x_D[z-1] * (alpha + 3 * beta * pow(phi_D[z-1], 2)) \
                        + Input_Signal_In[Burn_in_time + z - 1]
            y_D[z] = a * y_D[z-1] - b * x_D[z-1] + c
            phi_D[z] = k1 * x_D[z-1] - k2 * phi_D[z-1]

        #時間発展させた2つの軌道の差の算出
        delta_tau_x[0] = x_D[Step] - x[Burn_in_time+Step]
        delta_tau_y[0] = y_D[Step] - y[Burn_in_time+Step]
        delta_tau_phi[0] = phi_D[Step] - phi[Burn_in_time+Step]

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
                
            for zz in range(1, Step + 1):
                #初期摂動を与えたものをstep時間発展させた
                x_D[i+zz] = pow(x_D[i + zz - 1], 2) * np.exp(y_D[i + zz -1] - x_D[i + zz -1]) + k0 \
                        + k * x_D[i + zz-1] * (alpha + 3 * beta * pow(phi_D[i + zz-1], 2)) \
                            + Input_Signal_In[Burn_in_time+i + zz-1]
                y_D[i+zz] = a * y_D[i + zz-1] - b * x_D[i + zz-1] + c
                phi_D[i+zz] = k1 * x_D[i+zz-1] - k2 * phi_D[i+zz-1]
            
            #時間発展させた2つの軌道の差の算出
            delta_tau_x[i] = x_D[i+Step] - x[Burn_in_time+i+Step]
            delta_tau_y[i] = y_D[i+Step] - y[Burn_in_time+i+Step]
            delta_tau_phi[i] = phi_D[i+Step] - phi[Burn_in_time+i+Step]

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
            x_ave = (1 / Step) * (x_sum / len(x_Lyapunov))
            y_ave = (1 / Step) * (y_sum / len(y_Lyapunov))
            phi_ave = (1 / Step) * (phi_sum / len(phi_Lyapunov))

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

    Now = datetime.datetime.now()
    ProjectFile_Path = "Data/SimpleEvaluations"
    Figure_Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
    Figure_Name = "MaxLyapunovCheck_" + Figure_Date + "_"
    Result_Path = os.path.join(ProjectFile_Path, "Results")
    UFigure_Name = "MaxLyapunovCheck_Lorenz_" + Figure_Date
    UFigure_Path = os.path.join(Result_Path, UFigure_Name)

    # ディレクトリの作成
    os.makedirs(UFigure_Path, exist_ok=True)
    count = 0
    print("\n電磁束下Chialvoニューロン: 10回の平均最大リアプノフ指数の導出")

    for k in k_list:
        print({f"{count+1} / {len(k_list_result)} 回"})
        print(f"パラメータkの値: {k}")

        l = 0
        break_count = 0
        Lyapunov_Max = 0

        while l < 5:

            print(f"\n試行回数: {l}")
            #初期値の指定
            Initial_x = None #0.0448
            Initial_y = None #2.3011
            Initial_phi = None #0.0037

            #--------------------------------------------------------------------
            #基準軌道xと初期値について
            x = np.zeros(Alltime + (Step - 1))

            if Initial_x == None:
                x[0] = np.random.random() * 0.2 - 0.1
            else:
                x[0] = Initial_x

            #--------------------------------------------------------------------
            #基準軌道yと初期値について
            y = np.zeros(Alltime + (Step - 1))

            if Initial_y == None:
                y[0] = np.random.random() * 0.2 - 0.1
            else:
                y[0] = Initial_y

            #--------------------------------------------------------------------
            #基準軌道phiと初期値について
            phi = np.zeros(Alltime + (Step - 1))

            if Initial_phi == None:
                phi[0] = np.random.random() * 0.2 - 0.1
            else:
                phi[0] = Initial_phi

            #--------------------------------------------------------------------
            #入力信号の作成
            Input_Signal_In = np.ones(Alltime + (Step - 1)) * InputSingal

            if InputSingal_def == None:
                pass

            elif InputSingal_def == np.sin:
                for n in range(Alltime + (Step - 1)):
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
                rorentu_x = np.ones(Alltime + (Step - 1))
                rorentu_x[0] = np.random.random() * 0.0001
                
                rorentu_y = np.ones(Alltime + (Step - 1))
                rorentu_y[0] = np.random.random() * 0.0001
                
                rorentu_z = np.ones(Alltime + (Step - 1))
                rorentu_z[0] = np.random.random() * 0.0001

                Input_Signal_In[0] = rorentu_x[0]

                for n in range(Alltime + (Step - 1) - 1):

                    rorentu_x[n+1] = dt * (sigma *(rorentu_y[n] - rorentu_x[n])) + rorentu_x[n]
                    rorentu_y[n+1] = dt * (rorentu_x[n] * (rho - rorentu_z[n]) - rorentu_y[n]) + rorentu_y[n]
                    rorentu_z[n+1] = dt * (rorentu_x[n] * rorentu_y[n] - beta * rorentu_z[n]) + rorentu_z[n]

                    Input_Signal_In[n+1] = scale * rorentu_x[n+1]

            #--------------------------------------------------------------------
            #規準軌道の計算部
            #print("\n基準軌道の計算")
            for i in range(Alltime + (Step - 1) - 1):
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
            x_D = np.zeros(Runtime + (Step - 1))
            y_D = np.zeros(Runtime + (Step - 1))
            phi_D = np.zeros(Runtime + (Step - 1))

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
            for z in range(1, Step + 1):
                x_D[z] = pow(x_D[z-1], 2) * np.exp(y_D[z-1] - x_D[z-1]) + k0 \
                        + k * x_D[z-1] * (alpha + 3 * beta * pow(phi_D[z-1], 2)) \
                            + Input_Signal_In[Burn_in_time + z - 1]
                y_D[z] = a * y_D[z-1] - b * x_D[z-1] + c
                phi_D[z] = k1 * x_D[z-1] - k2 * phi_D[z-1]
            
            #時間発展させた2つの軌道の差の算出
            delta_tau_x[0] = x_D[Step] - x[Burn_in_time+Step]
            delta_tau_y[0] = y_D[Step] - y[Burn_in_time+Step]
            delta_tau_phi[0] = phi_D[Step] - phi[Burn_in_time+Step]

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
                
                for zz in range(1, Step + 1):
                    #初期摂動を与えたものをstep時間発展させた
                    x_D[i+zz] = pow(x_D[i + zz - 1], 2) * np.exp(y_D[i + zz -1] - x_D[i + zz -1]) + k0 \
                            + k * x_D[i + zz-1] * (alpha + 3 * beta * pow(phi_D[i + zz-1], 2)) \
                                + Input_Signal_In[Burn_in_time+i + zz-1]
                    y_D[i+zz] = a * y_D[i + zz-1] - b * x_D[i + zz-1] + c
                    phi_D[i+zz] = k1 * x_D[i+zz-1] - k2 * phi_D[i+zz-1]
                    
                #時間発展させた2つの軌道の差の算出
                delta_tau_x[i] = x_D[i+Step] - x[Burn_in_time+i+Step]
                delta_tau_y[i] = y_D[i+Step] - y[Burn_in_time+i+Step]
                delta_tau_phi[i] = phi_D[i+Step] - phi[Burn_in_time+i+Step]

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
                x_ave = (1 / Step) * (x_sum / len(x_Lyapunov))
                y_ave = (1 / Step) * (y_sum / len(y_Lyapunov))
                phi_ave = (1 / Step) * (phi_sum / len(phi_Lyapunov))

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
            Lyapunov_Max_ave = Lyapunov_Max / 5
            #print("\n")
            print(f"最大リアプノフ指数: {Lyapunov_Max_ave}")
            k_list_result[count] = Lyapunov_Max_ave

        count = count + 1

    #入力信号のプロット
    fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1)

    ax.set_title('Input Signal', fontsize=FontSize_Title)
    ax.set_xlabel('Time Step', fontsize=FontSize_Label)
    ax.set_ylabel('Signal', fontsize=FontSize_Label)
    ax.grid(True)
    
    ax.plot(Input_Signal_In)
    ax.legend()
    fig.savefig(os.path.join(UFigure_Path, Figure_Name + "InputSignal" + FileFormat))
    plt.close()
    
    #各パラメータによるリアプノフ指数の推移
    fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1)
    
    
    ax.set_title('Maximum Lyapunov Exponent', fontsize=FontSize_Title)
    ax.set_xlabel('k', fontsize=FontSize_Label)
    ax.set_ylabel('Maximum Lyapunov Exponent', fontsize=FontSize_Label)
    ax.set_xticks(np.arange(-7, 7.1, 0.5))
    ax.grid(True)

    ax.plot(k_list,k_list_result)
    ax.legend()
    fig.savefig(os.path.join(UFigure_Path, Figure_Name + "Lyapunov_List_50Step_7_701_001kannkaku_10Ave" + FileFormat))

    plt.close()


#====================================================================
import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
import math
import datetime
import os
from mpl_toolkits.mplot3d import axes3d, Axes3D

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

k = -2.5 #-3.2 -0.5

#パラメータkのリスト
k_lista = np.arange(-1.0, 7.01, 0.1)
k_list = np.round(k_lista, 4)

#各パラメータの平均の結果を格納する
k_list_result = np.zeros(len(k_list))
#各パラメータのリアプノフ指数
k_list_result_M = np.zeros((len(k_list), 10))

#print(k_list_result)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#入力信号
InputSingal = 0
#バイアス
#振幅の0.1倍
Bias = 0
#周期
Period = 1000

#None, sin, cos（ローレンツ方程式）
InputSingal_def = None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#空走時間
Burn_in_time = 1000

#実行時間
Runtime = 5000

#全体時間
Alltime = Burn_in_time + Runtime
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#入力信号を出力するか？
Plot_InputSignal = True

#プロット図全体のフォントサイズ
FigSize = (16,9)
FontSize_Axis = 32                #「各軸ラベル（Time Stepなど）」のフォントサイズ
FontSize_Title = 28               #「タイトル（一番上）」のフォントサイズ
FontSize_TickLabel = 14           #「各ラベルの単位（-2.0,-1.9など）」のフォントサイズ
FontSize_legend = 10              #「各凡例」のフォントサイズ
LineWidth = 2                     #線の太さ
FileFormat = ".png"          #ファイルフォーマット

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#リアプノフ指数を1回求める
Lyapunov_expotent = True

#リアプノフ指数のリスト
Lyapunov_List_10 = False

#print("--------------------------------------------------------------------")
def TimeDate():
    Time_Delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(Time_Delta ,'JST')
    Now = datetime.datetime.now(JST)
    Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
    return Date


#====================================================================
#リアプノフ指数を1回求める
#1step後の2つの軌道の差のプロット
if Lyapunov_expotent:

    print("\n====================================================================")
    print("====================================================================")

    print(f"\n空走時間（空走ステップ数）: {Burn_in_time}")
    print(f"計算時間（計算ステップ数）: {Runtime}")
    print(f"総合時間（総合ステップ数）: {Alltime}")

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
    Input_Signal_In = np.zeros(Alltime) * InputSingal

    if InputSingal_def == None:
        #入力信号：定常信号
        print("\n====================================================================")
        print("入力信号: 定常信号")
        pass

    elif InputSingal_def == np.sin:
        print("\n====================================================================")
        print("入力信号: sin波")
        for n in range(Alltime):
            Input_Signal_In[n] = 0.1 * InputSingal * np.sin(2 * n * np.pi / Period) + Bias

    elif InputSingal_def == np.cos:
        print("\n====================================================================")
        print("入力信号: ローレンツ方程式")

        #微小時間
        dt = 0.01
        #信号の大きさ
        scale = 1 / 1000 * InputSingal

        #パラメータ
        sigma = 10
        rho = 28
        beta = 8/3

        np.random.seed(728)
        #入力信号の長さ
        rorentu_x = np.ones(Alltime)
        rorentu_x[0] = np.random.random() * 0.02 - 0.01
        
        rorentu_y = np.ones(Alltime)
        rorentu_y[0] = np.random.random() * 0.02 - 0.01
        
        rorentu_z = np.ones(Alltime)
        rorentu_z[0] = np.random.random() * 0.02 - 0.01
        np.random.seed(None)

        Input_Signal_In[0] = rorentu_x[0]

        for n in range(0, Alltime - 1):
            rorentu_x[n+1] = dt * (sigma *(rorentu_y[n] - rorentu_x[n])) + rorentu_x[n]
            rorentu_y[n+1] = dt * (rorentu_x[n] * (rho - rorentu_z[n]) - rorentu_y[n]) + rorentu_y[n]
            rorentu_z[n+1] = dt * (rorentu_x[n] * rorentu_y[n] - beta * rorentu_z[n]) + rorentu_z[n]

            Input_Signal_In[n+1] = scale * rorentu_x[n+1]

    #入力信号のプロット
    if Plot_InputSignal:
        print("\n--------------------------------------------------------------------")
        print("入力信号の描写")

        fig = plt.figure(figsize = FigSize)
        ax = fig.add_subplot(1,1,1)
        ax.plot(Input_Signal_In[Burn_in_time:],'-', lw = LineWidth)
        ax.set_xlabel("Time Step", fontsize = FontSize_Axis)
        ax.set_ylabel("x", fontsize = FontSize_Axis)
        ax.grid()
        fig.tight_layout()

        plt.savefig(f"./{TimeDate()}_Input_Signal.png")
        plt.show()
    #--------------------------------------------------------------------
    #規準軌道の計算部
    print("\n--------------------------------------------------------------------")
    print("\n基準軌道の計算")

    for i in range(0, Alltime - 1):
        print("\r%d / %d"%(i, Alltime), end = "")
        x[i+1] = pow(x[i], 2) * np.exp(y[i] - x[i]) + k0\
              + k * x[i] * (alpha + 3 * beta * pow(phi[i], 2))\
                + Input_Signal_In[i]
        y[i+1] = a * y[i] - b * x[i] + c
        phi[i+1] = k1 * x[i] - k2 * phi[i]

    print("\n\n--------------------------------------------------------------------")
    print("\n電磁束下Chialvoの基準軌道")

    fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1, projection = '3d')
    ax.plot(x[Burn_in_time:], y[Burn_in_time:], phi[Burn_in_time:], '-', lw = LineWidth)
    ax.set_xlabel("x", fontsize = (FontSize_Axis / 10) * 7.5)
    ax.set_ylabel("y", fontsize = (FontSize_Axis / 10) * 7.5)
    ax.set_zlabel("phi", fontsize = (FontSize_Axis / 10) * 7.5)
    ax.grid()
    fig.tight_layout()

    plt.savefig(f"./{TimeDate()}_EMChialvo_Standard.png")
    plt.show()

    #--------------------------------------------------------------------
    #最大リアプノフ指数を取る配列
    Lyapunov = np.zeros(Runtime-1)
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

    print("\n====================================================================")
    print("====================================================================")
    print(f"\n基準軌道の初期値の点: {Standard_0}\n摂動軌道の初期値の点: {Pert_0}")
    print(f"\n基準軌道の発展先の点: {Standard_1}\n摂動軌道の発展先の点: {Pert_1}")

    print(f"\n初期値の差ベクトル: {Vector_0}\n1Step先の差ベクトル: {Vector_1}")
    print(f"\n初期値の差ベクトルの大きさ: {Norm_0}\n1Step先の差ベクトルの大きさ: {Norm_1}")
    
    Lyapunov[0] = np.log(Norm_1 / Norm_0)
    Lyapunov_Sum = Lyapunov_Sum + Lyapunov[0]
    print(f"\n{1}回目の最大リアプノフ指数: {Lyapunov[0]}")

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
        print(f"\n基準軌道の発展先の点: {Standard_1}\n摂動軌道の発展先の点: {Pert_1}")

        print(f"\n初期値の差ベクトル: {Vector_0}\n1Step先の差ベクトル: {Vector_1}")
        print(f"\n初期値の差ベクトルの大きさ: {Norm_0}\n1Step先の差ベクトルの大きさ: {Norm_1}")
        
        Lyapunov[i] = np.log(Norm_1 / Norm_0)
        Lyapunov_Sum = Lyapunov_Sum + Lyapunov[i]
        print(f"\n{i+1}回目の最大リアプノフ指数: {Lyapunov[i]}")

    print("\n====================================================================")
    print("====================================================================")
    print("リアプノフ指数")

    fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(Lyapunov[:],'-', lw = LineWidth)
    ax.set_xlabel("Time Step", fontsize = FontSize_Axis)
    ax.set_ylabel("Lyapunov", fontsize = FontSize_Axis)
    ax.grid()
    fig.tight_layout()

    plt.savefig(f"./{TimeDate()}_Lyapunov.png")
    plt.show()

    print("入力信号の描写")

    fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(x[Burn_in_time:],'-', lw = LineWidth)
    ax.set_xlabel("Time Step", fontsize = FontSize_Axis)
    ax.set_ylabel("x", fontsize = FontSize_Axis)
    ax.grid()
    fig.tight_layout()

    plt.savefig(f"./{TimeDate()}_Chialvo_x.png")
    plt.show()

    Lyapunov_Ave = Lyapunov_Sum / (len(Lyapunov))
    print(Lyapunov_Ave)

if Lyapunov_List_10:
    print("\n====================================================================")
    print("====================================================================")

    print(f"\n空走時間（空走ステップ数）: {Burn_in_time}")
    print(f"計算時間（計算ステップ数）: {Runtime}")
    print(f"総合時間（総合ステップ数）: {Alltime}")

    print("\n====================================================================")
    print("====================================================================")

    print("\n電磁束下Chialvoニューロン: パラメータ変化時の最大リアプノフ指数の導出")
    
    #kのカウンター
    k_count = 0
    
    for k in k_list:
        print(f"\n{k_count + 1} / {len(k_list_result)} 回")
        print(f"パラメータのkの値: {k}")

        #10回カウンターのl
        l = 0
        #測定不可のカウンター
        break_count = 0
        Lyapunov_Max = 0
        Lyapunov_Ave = 0

        while l < 10:
            #print(f"試行回数: {l}")
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
                #入力信号：定常信号
                pass

            elif InputSingal_def == np.sin:
                for n in range(Alltime):
                    Input_Signal_In[n] = 0.1 * InputSingal * np.sin(2 * n * np.pi / Period) + Bias

            elif InputSingal_def == np.cos:
                #微小時間
                dt = 0.01
                #信号の大きさ
                scale = 1 / 500 * InputSingal

                #パラメータ
                sigma = 10
                rho = 28
                beta = 8/3

                np.random.seed(728)
                #入力信号の長さ
                rorentu_x = np.ones(Alltime)
                rorentu_x[0] = np.random.random() * 0.02 - 0.01
                
                rorentu_y = np.ones(Alltime)
                rorentu_y[0] = np.random.random() * 0.02 - 0.01
                
                rorentu_z = np.ones(Alltime)
                rorentu_z[0] = np.random.random() * 0.02 - 0.01

                np.random.seed(None)
                Input_Signal_In[0] = rorentu_x[0]

                for n in range(0, Alltime - 1):
                    rorentu_x[n+1] = dt * (sigma *(rorentu_y[n] - rorentu_x[n])) + rorentu_x[n]
                    rorentu_y[n+1] = dt * (rorentu_x[n] * (rho - rorentu_z[n]) - rorentu_y[n]) + rorentu_y[n]
                    rorentu_z[n+1] = dt * (rorentu_x[n] * rorentu_y[n] - beta * rorentu_z[n]) + rorentu_z[n]

                    Input_Signal_In[n+1] = scale * rorentu_x[n+1]


            #--------------------------------------------------------------------
            #規準軌道の計算部
            for i in range(0, Alltime - 1):
                x[i+1] = pow(x[i], 2) * np.exp(y[i] - x[i]) + k0\
                    + k * x[i] * (alpha + 3 * beta * pow(phi[i], 2))\
                        + Input_Signal_In[i]
                y[i+1] = a * y[i] - b * x[i] + c
                phi[i+1] = k1 * x[i] - k2 * phi[i]
            
            #--------------------------------------------------------------------
            #最大リアプノフ指数を取る配列
            Lyapunov = np.zeros(Runtime-1)
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

            Lyapunov[0] = np.log(Norm_1 / Norm_0)
            Lyapunov_Sum = Lyapunov_Sum + Lyapunov[0]

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
                
                Lyapunov[i] = np.log(Norm_1 / Norm_0)
                Lyapunov_Sum = Lyapunov_Sum + Lyapunov[i]
            

            #Lyapunov_SumがNanだったとき
            if math.isnan(Lyapunov_Sum):
                break_count = break_count + 1
                print(f"Over Flow: {break_count}")

                #かつ計算不可が10回未満
                if break_count < 10:
                    pass

                #計算不可が10回以上
                else:
                    print(f"計測失敗")
                    k_list_result[k_count] = None
                    break
            
            #通常通り算出できた場合
            else:
                Lyapunov_Ave = Lyapunov_Sum / (len(Lyapunov))
                #print(f"リアプノフ指数: {Lyapunov_Ave}")
                l = l + 1

        if break_count == 10:
            k_list_result[k_count] = None

        else:
            k_list_result[k_count] = Lyapunov_Ave / 10
            print(f"リアプノフ指数: {k_list_result[k_count]}")
        
        k_count = k_count + 1
    

    print("入力信号の描写")

    fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(Input_Signal_In[Burn_in_time:],'-', lw = LineWidth)
    ax.set_xlabel("Time Step", fontsize = FontSize_Axis)
    ax.set_ylabel("x", fontsize = FontSize_Axis)
    ax.grid()
    fig.tight_layout()

    plt.savefig(f"./{TimeDate()}_Input_Signal.png")
    plt.show()


    #各パラメータによるリアプノフ指数の推移
    fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1)

    ax.plot(k_list, k_list_result, '-', lw = LineWidth)
    ax.set_xticks(np.arange(-1.0, 7.01, 0.5))
    ax.set_xlabel("k", fontsize = FontSize_Axis)
    ax.set_ylabel("Maximum Lyapunov", fontsize = FontSize_Axis)
    
    ax.grid()
    fig.tight_layout()

    plt.savefig(f"./{TimeDate()}_Maximum_Lyapunov_k.png")
    plt.show()



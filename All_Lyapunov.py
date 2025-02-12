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

alpha_Chialvo = 0.1 #0.1 0.1
beta_Chialvo = 0.2 #0.2 0.1

k = -3.2 #-3.2 -0.5

#--------------------------------------------------------------------
#ローレンツ方程式のパラメータ
sigma = 10
rho = 28
beta_Lorentz = 8/3

#時間刻み幅
dt = 1/1000
#何ステップ先まで実行するか
T = 1

#====================================================================
#データ保存用パラメータ
FigSize = (16,9)
FontSize_Axis = 32                #「各軸ラベル（Time Stepなど）」のフォントサイズ
FontSize_Title = 28               #「タイトル（一番上）」のフォントサイズ
FontSize_TickLabel = 14           #「各ラベルの単位（-2.0,-1.9など）」のフォントサイズ
FontSize_legend = 10              #「各凡例」のフォントサイズ
LineWidth = 2                     #線の太さ
FileFormat = ".png"          #ファイルフォーマット

#====================================================================
#空走時間
Burnin_Time = 200
#時間ステップ数
Run_Time = 500

#空走時間
Burnin_Time_Lorenz = 25000
#時間ステップ数
Run_Time_Lorenz = 100000

#パラメータkのリスト
k_list_E = np.arange(-5.0, 5.01, 0.01)
k_list = np.round(k_list_E, 4)
#パラメータbのリスト
b_list_E = np.arange(0, 1.001, 0.0005)
b_list = np.round(b_list_E, 4)


#====================================================================
#Chialvoにおける1回のリアプノフスペクトル
Lyapunov_Chialvo = False
Lyapunov_Chialvo_K_Loop = True

#Lorenzにおけるリアプノフスペクトラム
Lyapunov_Lorenz = False
#====================================================================
"""共通関数"""

#時間取得
def TimeDate():
    Time_Delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(Time_Delta ,'JST')
    Now = datetime.datetime.now(JST)
    Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
    return Date

#最初の関数
def Print_Info(Equation, Burnin, TimeStep):
    print("\n====================================================================")
    print("====================================================================")

    print(f"\n空走時間（空走ステップ数）: {Burnin}")
    print(f"評価時間（評価ステップ数）: {TimeStep}")
    print(f"\n方程式 : {Equation}")

    print("\n====================================================================")
    print("====================================================================")

#--------------------------------------------------------------------
"""計算部関数"""

#Chialvoの更新式
def ChialvoSystem(old_c, a, b, c, k, k0, k1, k2, alpha, beta):
    x = old_c[0]
    y = old_c[1]
    phi = old_c[2]

    dx = pow(x, 2) * np.exp(y - x) + k0 + k * x * (alpha + 3 * beta *pow(phi, 2))
    dy = a * y - b * x + c
    dphi = k1 * x - k2 * phi

    new_s = np.stack([dx, dy, dphi])

    return new_s

#Chialvoのヤコビ行列
def ChialvoSystem_J(old_c, a, b, c, k, k0, k1, k2, alpha, beta):
    x = old_c[0]
    y = old_c[1]
    phi = old_c[2]

    J_cont = np.zeros((3, 3))
    
    J_cont[0, 0] = np.exp(y - x) * (2 * x - pow(x, 2)) + k * (alpha + 3 * beta *pow(phi, 2))
    J_cont[0, 1] = pow(x, 2) * np.exp(y - x)
    J_cont[0, 2] = 6 * k * x * beta * phi

    J_cont[1, 0] = -b
    J_cont[1, 1] = a
    J_cont[1, 2] = 0

    J_cont[2, 0] = k1
    J_cont[2, 1] = 0
    J_cont[2, 2] = -k2

    return J_cont

#--------------------------------------------------------------------
#ローレンツ方程式の更新式
def LorenzSystem(old_l, sigma, rho, beta, dt, T):
    x = old_l[0]
    y = old_l[1]
    z = old_l[2]

    for i in range(T):
        x = x + ((sigma * (y - x)) * dt)
        y = y + ((x * (rho - z) - y) * dt)
        z = z + ((x * y - beta * z) * dt)

    new_s = np.stack([x, y, z])
    
    return new_s

#ローレンツ方程式のヤコビ行列
def LorenzSystem_J(old_l, sigma, rho, beta, dt):
    x = old_l[0]
    y = old_l[1]
    z = old_l[2]
    
    J_cont = np.zeros((3, 3))
    
    J_cont[0, 0] = -sigma
    J_cont[0, 1] = sigma
    J_cont[0, 2] = 0

    J_cont[1, 0] = rho - z
    J_cont[1, 1] = -1
    J_cont[1, 2] = -x

    J_cont[2, 0] = y
    J_cont[2, 1] = x
    J_cont[2, 2] = -beta

    # 離散時間系のヤコビアン
    J_map = np.eye(3) + dt * J_cont

    return J_map


#====================================================================

if Lyapunov_Chialvo:
    Print_Info("Chialvio System", Burnin_Time, Run_Time)

    #初期値生成
    chialvo = ((np.random.rand(3) * 2 - 1) * 0.1)

    #Chialvoの更新式の配列
    chialvo_Result_List = np.zeros((Run_Time, 3))

    #空走時間
    print(f"\n空走時間")
    for n in range(Burnin_Time):
        print("\r%d / %d"%(n+1, Burnin_Time), end = "")

        #Chialvoの更新式
        chialvo = ChialvoSystem(chialvo, a, b, c, k, k0, k1, k2, alpha_Chialvo, beta_Chialvo)

    #時刻0における単位行列
    Q = np.eye(3)

    #リアプノフスペクトルの総和
    Lyapunov_Result = np.zeros(3)
    #各時刻のリアプノフスペクトル
    Lyapunov_Result_Time = np.zeros((Run_Time, 3))

    #実行時間
    print(f"\n実行時間")
    for i in range(Run_Time):
        print("\r%d / %d"%(i+1, Run_Time), end = "")

        #Chialvoの更新式
        chialvo = ChialvoSystem(chialvo, a, b, c, k, k0, k1, k2, alpha_Chialvo, beta_Chialvo)
        #各時刻のChialvoの更新式
        chialvo_Result_List[i, :] = chialvo

        #各時刻のヤコビ行列をQR分解
        Q, R = np.linalg.qr(ChialvoSystem_J(chialvo_Result_List[i,:], a, b, c, k, k0, k1, k2, alpha_Chialvo, beta_Chialvo) @ Q)
        #上三角行列の対角成分取得
        R_diag = np.diag(R)
        #それぞれの対角要素の絶対値のlogを総和
        Lyapunov_Result = Lyapunov_Result + np.log(np.abs(R_diag))
        #各時刻のリアプノフスペクトルを取得
        Lyapunov_Result_Time[i, :] = np.log(np.abs(R_diag)) / (i + 1)

    #総時間のリアプノフスペクトル
    Lyapunov = Lyapunov_Result / Run_Time 

    #各成分におけるリアプノフスペクトル
    Lyapunov_X = round(Lyapunov[0], 4)
    Lyapunov_Y = round(Lyapunov[1], 4)
    Lyapunov_Phi = round(Lyapunov[2], 4)

    Lyapunov_XYPhi = np.stack([Lyapunov_X, Lyapunov_Y, Lyapunov_Phi])
    
    print("\n\n====================================================================")
    print("====================================================================")
    print(f"\nChialvoのリアプノフスペクトル: {Lyapunov_XYPhi}")

    fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(Lyapunov_Result_Time[:,0],'-', label = "x")
    ax.plot(Lyapunov_Result_Time[:,1],'-', label = "y")
    ax.plot(Lyapunov_Result_Time[:,2],'-', label = "phi")
    ax.set_xlabel("Time", fontsize = FontSize_Axis)
    ax.set_ylabel("Lyapunov", fontsize = FontSize_Axis)
    ax.grid()
    ax.legend()
    fig.tight_layout()

    plt.show()

if Lyapunov_Chialvo_K_Loop:
    Print_Info("Chialvio System: List", Burnin_Time, Run_Time)

    #変数カウンター
    Count = 0
    Result_Lyapunov = np.zeros(((len(k_list)), 3))

    for z in range(0, len(k_list)):

        k = k_list[z]
        print(f"\n{Count + 1} / {len(k_list)} 回")
        print(f"kの値: {k}")

        #初期値生成
        chialvo = ((np.random.rand(3) * 2 - 1) * 0.1)

        #Chialvoの更新式の配列
        chialvo_Result_List = np.zeros((Run_Time, 3))

        #空走時間
        print(f"\n空走時間")
        for n in range(Burnin_Time):
            print("\r%d / %d"%(n+1, Burnin_Time), end = "")
            #Chialvoの更新式
            chialvo = ChialvoSystem(chialvo, a, b, c, k, k0, k1, k2, alpha_Chialvo, beta_Chialvo)

        #時刻0における単位行列
        Q = np.eye(3)

        #リアプノフスペクトルの総和
        Lyapunov_Result = np.zeros(3)
        #各時刻のリアプノフスペクトル
        Lyapunov_Result_Time = np.zeros((Run_Time, 3))

        #実行時間
        print(f"\n実行時間")
        for i in range(Run_Time):
            print("\r%d / %d"%(i+1, Run_Time), end = "")

            #Chialvoの更新式
            chialvo = ChialvoSystem(chialvo, a, b, c, k, k0, k1, k2, alpha_Chialvo, beta_Chialvo)
            #各時刻のChialvoの更新式
            chialvo_Result_List[i, :] = chialvo

            #各時刻のヤコビ行列をQR分解
            Q, R = np.linalg.qr(ChialvoSystem_J(chialvo_Result_List[i,:], a, b, c, k, k0, k1, k2, alpha_Chialvo, beta_Chialvo) @ Q)
            #上三角行列の対角成分取得
            R_diag = np.diag(R)
            #それぞれの対角要素の絶対値のlogを総和
            Lyapunov_Result = Lyapunov_Result + np.log(np.abs(R_diag))

        #総時間のリアプノフスペクトル
        Lyapunov = Lyapunov_Result / Run_Time
        print(f"\nリアプノフスペクトラム: {Lyapunov}")
        print(f" \n====================================================================")

        Result_Lyapunov[Count, 0] = round(Lyapunov[0], 5)
        Result_Lyapunov[Count, 1] = round(Lyapunov[1], 5)
        Result_Lyapunov[Count, 2] = round(Lyapunov[2], 5)

        Count = Count + 1

    fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(k_list,Result_Lyapunov[:,0],'.', label = "x")
    ax.plot(k_list,Result_Lyapunov[:,1],'.', label = "y")
    ax.plot(k_list,Result_Lyapunov[:,2],'.', label = "phi")
    ax.set_xlabel("k", fontsize = FontSize_Axis)
    ax.set_ylabel("Lyapunov", fontsize = FontSize_Axis)
    ax.set_xticks(np.arange(-5, 5.01, 0.5))
    ax.grid()
    ax.legend()
    fig.tight_layout()

    plt.savefig(f"./{TimeDate()}_Lyapunov_Spectrum.png")
    plt.show()



#--------------------------------------------------------------------

if Lyapunov_Lorenz:
    Print_Info("Lorenz System", Burnin_Time_Lorenz, Run_Time_Lorenz)

    #初期値生成
    Lorenz = ((np.random.rand(3) * 2 - 1) * 0.1)

    #Chialvoの更新式の配列
    Lorenz_Result_List = np.zeros((Run_Time_Lorenz, 3))

    #空走時間
    print(f"\n空走時間")
    for n in range(Burnin_Time_Lorenz):
        print("\r%d / %d"%(n+1, Burnin_Time_Lorenz), end = "")

        #Chialvoの更新式
        Lorenz = LorenzSystem(Lorenz, sigma, rho, beta_Lorentz, dt, T)

    #時刻0における単位行列
    Q = np.eye(3)

    #リアプノフスペクトルの総和
    Lyapunov_Result = np.zeros(3)
    #各時刻のリアプノフスペクトル
    Lyapunov_Result_Time = np.zeros((Run_Time_Lorenz, 3))

    #実行時間
    print(f"\n実行時間")
    for i in range(Run_Time_Lorenz):
        print("\r%d / %d"%(i+1, Run_Time_Lorenz), end = "")

        #Chialvoの更新式
        Lorenz = LorenzSystem(Lorenz, sigma, rho, beta_Lorentz, dt, T)
        #各時刻のChialvoの更新式
        Lorenz_Result_List[i, :] = Lorenz

        #各時刻のヤコビ行列をQR分解
        Q, R = np.linalg.qr(LorenzSystem_J(Lorenz_Result_List[i,:], sigma, rho, beta_Lorentz, dt) @ Q)
        #上三角行列の対角成分取得
        R_diag = np.diag(R)
        #それぞれの対角要素の絶対値のlogを総和
        Lyapunov_Result = Lyapunov_Result + np.log(np.abs(R_diag))
        #各時刻のリアプノフスペクトルを取得
        Lyapunov_Result_Time[i, :] = np.log(np.abs(R_diag)) / (i + 1)

    #総時間のリアプノフスペクトル
    Lyapunov = Lyapunov_Result / (Run_Time_Lorenz * dt)

    #各成分におけるリアプノフスペクトル
    Lyapunov_X = round(Lyapunov[0], 5)
    Lyapunov_Y = round(Lyapunov[1], 5)
    Lyapunov_Phi = round(Lyapunov[2], 5)

    Lyapunov_XYPhi = np.stack([Lyapunov_X, Lyapunov_Y, Lyapunov_Phi])
    
    print("\n\n====================================================================")
    print("====================================================================")
    print(f"\nLorenzのリアプノフスペクトル: {Lyapunov_XYPhi}")

    fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(Lyapunov_Result_Time[:,0],'-', label = "x")
    ax.plot(Lyapunov_Result_Time[:,1],'-', label = "y")
    ax.plot(Lyapunov_Result_Time[:,2],'-', label = "phi")
    ax.set_xlabel("Time", fontsize = FontSize_Axis)
    ax.set_ylabel("Lyapunov", fontsize = FontSize_Axis)
    ax.grid()
    ax.legend()
    fig.tight_layout()

    plt.show()



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

FigSize = (16,9)
FontSize_Axis = 32                #「各軸ラベル（Time Stepなど）」のフォントサイズ
FontSize_Title = 28               #「タイトル（一番上）」のフォントサイズ
FontSize_TickLabel = 14           #「各ラベルの単位（-2.0,-1.9など）」のフォントサイズ
FontSize_legend = 10              #「各凡例」のフォントサイズ
LineWidth = 2                     #線の太さ
FileFormat = ".png"          #ファイルフォーマット


#電磁束下Chialvoのパラメータ
a = 0.6 #0.89 0.6
b = 0.1 #0.6 0.1
c = 1.4 #0.28 1.4
k0 = 0.1 #0.04 0.1

k1 = 0.1 #0.1 0.1
k2 = 0.2 #0.2 0.2
alpha = 0.1 #0.1 0.1
beta = 0.1 #0.2 0.1

k = -0.5 #-3.2 -0.5

#ローレンツ方程式のパラメータ
sigma = 10
rhos = 28
beta_Lo = 8/3
dt = 0.001


#時間ステップの指定
time_steps = 10000

#====================================================================

#Chialvoの更新式
def ChialvoSystem(old_s, a, b, c, k, k0, k1, k2, alpha, beta):
    x = old_s[0]
    y = old_s[1]
    phi = old_s[2]

    dx = pow(x, 2) * np.exp(y - x) + k0 + k * x * (alpha + 3 * beta *pow(phi, 2))
    dy = a * y - b * x + c
    dphi = k1 * x - k2 * phi

    new_s = np.stack([dx, dy, dphi])

    return new_s

#Chialvoのヤコビ行列
def ChialvoSystem_J(old_s, a, b, c, k, k0, k1, k2, alpha, beta):
    x = old_s[0]
    y = old_s[1]
    phi = old_s[2]

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

#====================================================================

#ローレンツ方程式の更新式
def LorenzSystem(s, sigma, rho, beta, dt):
    x = s[0]
    y = s[1]
    z = s[2]

    dx = (sigma * (y - x))
    dy = (x * (rho - z) - y)
    dz = (x * y - beta * z)

    nx = x + (dx * dt)
    ny = y + (dy * dt)
    nz = z + (dz * dt)

    new_s = np.stack([nx, ny, nz])
    
    return new_s

#ローレンツ方程式のヤコビ行列
def LorenzSystem_J(s, sigma, rho, beta, dt):
    x = s[0]
    y = s[1]
    z = s[2]
    
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

#エノン写像の更新式
def HenonSystem(old_s, A, B):
    x = old_s[0]
    y = old_s[1]

    dx = 1 + y - (A * pow(x, 2))
    dy = B * x

    new_s = np.stack([dx, dy])

    return new_s

def HenonSystem_J(old_s, A, B):
    x = old_s[0]
    y = old_s[1]
    
    J_cont = np.zeros((2, 2))
    
    J_cont[0, 0] = -2 * A * x
    J_cont[0, 1] = 1
    
    J_cont[1, 0] = B
    J_cont[1, 1] = 0
    
    return J_cont

#====================================================================

def Main_Lorenz():

    #ローレンツ方程式の初期値
    #s0 = [17.06769169, 11.48976297, 43.64991193]
    #Lorenz = s0

    s0 = np.random.rand(3) * 2 - 1
    Lorenz = s0
    #ローレンツ方程式の結果配列
    Lorenz_List = np.zeros((time_steps, 3))

    #空走時間
    for x in range(20000):
        Lorenz = LorenzSystem(Lorenz, sigma, rhos, beta_Lo, dt)

    for i in range(time_steps):
        #ローレンツ方程式の更新
        Lorenz = LorenzSystem(Lorenz, sigma, rhos, beta_Lo, dt)
        #ローレンツ方程式の結果収納
        Lorenz_List[i, :] = Lorenz

    
    #単位行列
    Q = np.eye(3)
    #リアプノフスペクトルの結果配列
    lyapu_list = np.zeros(3)
    lyapu_list_time = np.zeros((time_steps, 3))

    for j in range(time_steps):
        #各時刻のヤコビ行列をQR分解
        Q, R = np.linalg.qr(LorenzSystem_J(Lorenz_List[j,:], sigma, rhos, beta_Lo, dt) @ Q)
        #上三角行列の対角成分取得
        R_diag = np.diag(R)
        #それぞれの対角要素の絶対値のlogを総和
        lyapu_list = lyapu_list + np.log(np.abs(R_diag))
        #各時刻

    print(lyapu_list)
    lyapunov = lyapu_list / (time_steps * dt)
    print(lyapunov)

    lyapunov_x = round(lyapunov[0], 6)
    lyapunov_y = round(lyapunov[1], 6)
    lyapunov_z = round(lyapunov[2], 6)

    lll = [lyapunov_x, lyapunov_y, lyapunov_z]
    print(lll)


        
    """fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1, projection = '3d')
    ax.plot(Lorenz_List[1000:,0], Lorenz_List[1000:,1],Lorenz_List[1000:,2])

    plt.show()"""

Main_Lorenz()

#====================================================================

def Main_Henon():
    A = 1.4
    B = 0.3
    #ローレンツ方程式の初期値
    #s0 = [17.06769169, 11.48976297, 43.64991193]
    #Lorenz = s0

    s0 = np.random.rand(2) * 2 - 1
    Henon = s0
    #ローレンツ方程式の結果配列
    Henon_List = np.zeros((time_steps, 2))

    #空走時間
    for x in range(20000):
        Henon = HenonSystem(Henon, A, B)

    for i in range(time_steps):
        #ローレンツ方程式の更新
        Henon = HenonSystem(Henon, A, B)
        #ローレンツ方程式の結果収納
        Henon_List[i, :] = Henon

    
    #単位行列
    Q = np.eye(2)
    #リアプノフスペクトルの結果配列
    lyapu_list = np.zeros(2)
    lyapu_list_time = np.zeros((time_steps, 2))

    for j in range(time_steps):
        #各時刻のヤコビ行列をQR分解
        Q, R = np.linalg.qr(HenonSystem_J(Henon_List[j,:], A, B) @ Q)
        #上三角行列の対角成分取得
        R_diag = np.diag(R)
        #それぞれの対角要素の絶対値のlogを総和
        lyapu_list = lyapu_list + np.log(np.abs(R_diag))
        #各時刻

    print(lyapu_list)
    lyapunov = lyapu_list / (time_steps)
    print(lyapunov)

    lyapunov_x = round(lyapunov[0], 6)
    lyapunov_y = round(lyapunov[1], 6)

    lll = [lyapunov_x, lyapunov_y]
    print(lll)


        
    """fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1, projection = '3d')
    ax.plot(Lorenz_List[1000:,0], Lorenz_List[1000:,1],Lorenz_List[1000:,2])

    plt.show()"""

#Main_Henon()

#====================================================================

def Main_Chialvo():
    #ローレンツ方程式の初期値
    #s0 = [17.06769169, 11.48976297, 43.64991193]
    #Lorenz = s0

    s0 = (np.random.rand(3) * 2 - 1) * 0.1
    chialvo = s0
    #ローレンツ方程式の結果配列
    Chialvo_List = np.zeros((time_steps, 3))

    #空走時間
    for x in range(20000):
        chialvo = ChialvoSystem(chialvo, a, b, c, k, k0, k1, k2, alpha, beta)


    
    #単位行列
    Q = np.eye(3)
    #リアプノフスペクトルの結果配列
    lyapu_list = np.zeros(3)
    lyapu_list_time = np.zeros((time_steps, 3))

    for j in range(time_steps):
        
        #Chialvoの更新式
        chialvo = ChialvoSystem(chialvo, a, b, c, k, k0, k1, k2, alpha, beta)
        #ローレンツ方程式の結果収納
        Chialvo_List[j, :] = chialvo

        #各時刻のヤコビ行列をQR分解
        Q, R = np.linalg.qr(ChialvoSystem_J(Chialvo_List[j,:], a, b, c, k, k0, k1, k2, alpha, beta) @ Q)
        #上三角行列の対角成分取得
        R_diag = np.diag(R)
        #それぞれの対角要素の絶対値のlogを総和
        lyapu_list = lyapu_list + np.log(np.abs(R_diag))
        #各時刻

    #print(lyapu_list)
    lyapunov = lyapu_list / (time_steps)
    #print(lyapunov)

    lyapunov_x = round(lyapunov[0], 5)
    lyapunov_y = round(lyapunov[1], 5)
    lyapunov_z = round(lyapunov[2], 5)

    lll = [lyapunov_x, lyapunov_y, lyapunov_z]
    print(lll)

#Main_Chialvo()

def Chialvo_loop():

    #kのリスト
    k_lista = np.arange(-3, 0.01, 0.01)
    k_list = np.round(k_lista, 4)

    Result_Lyapunov = np.zeros(((len(k_list)), 3))
    ex = 0

    for z in range(0,len(k_list)):

        k = k_list[z]
        print(f"kの値: {k}")

        s0 = (np.random.rand(3) * 2 - 1) * 0.1
        chialvo = s0
        #ローレンツ方程式の結果配列
        Chialvo_List = np.zeros((time_steps, 3))

        #空走時間
        for x in range(1000):
            chialvo = ChialvoSystem(chialvo, a, b, c, k, k0, k1, k2, alpha, beta)
        
        #単位行列
        Q = np.eye(3)
        #リアプノフスペクトルの結果配列
        Result_Lyapunov_one = np.zeros(3)

        for j in range(time_steps):
            
            #ローレンツ方程式の更新
            chialvo = ChialvoSystem(chialvo, a, b, c, k, k0, k1, k2, alpha, beta)
            #ローレンツ方程式の結果収納
            Chialvo_List[j, :] = chialvo

            #各時刻のヤコビ行列をQR分解
            Q, R = np.linalg.qr(ChialvoSystem_J(Chialvo_List[j,:], a, b, c, k, k0, k1, k2, alpha, beta) @ Q)
            #上三角行列の対角成分取得
            R_diag = np.diag(R)
            #それぞれの対角要素の絶対値のlogを総和
            Result_Lyapunov_one = Result_Lyapunov_one + np.log(np.abs(R_diag))

        Lyapunov = Result_Lyapunov_one / (time_steps)
        print(Lyapunov)

        Result_Lyapunov[ex, 0] = round(Lyapunov[0], 5)
        Result_Lyapunov[ex, 1] = round(Lyapunov[1], 5)
        Result_Lyapunov[ex, 2] = round(Lyapunov[2], 5)

        ex = ex + 1

    fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(k_list,Result_Lyapunov[:,0],'.', label = "x")
    ax.plot(k_list,Result_Lyapunov[:,1],'.', label = "y")
    ax.plot(k_list,Result_Lyapunov[:,2],'.', label = "phi")
    ax.set_xlabel("k", fontsize = FontSize_Axis)
    ax.set_ylabel("Lyapunov", fontsize = FontSize_Axis)
    ax.grid()
    ax.set_xticks(np.arange(-3, 0.01, 1))
    ax.legend()
    fig.tight_layout()

    plt.show()

#Chialvo_loop()
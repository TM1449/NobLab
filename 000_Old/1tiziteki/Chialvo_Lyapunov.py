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
a = 0.89 #0.89 0.6
b = 0.6 #0.6 0.1
c = 0.28 #0.28 1.4
k0 = 0.04 #0.04 0.1

k1 = 0.1 #0.1 0.1
k2 = 0.2 #0.2 0.2
alpha = 0.1 #0.1 0.1
beta = 0.2 #0.2 0.1

k = -0.8 #-3.2 -0.5

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

#ローレンツ方程式の更新式
def LorenzSystem(s, sigma, rho, beta, dt):
    x = s[0]
    y = s[1]
    z = s[2]

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    ds = np.stack([x, y, z])
    new_s = s + ds * dt
    
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


def HenonSystem(s):
    A = 1.4
    B = 0.3

    x = s[0]
    y = s[1]

    dx = 1 - A * pow(x, 2) + y
    dy = B * x

    news = np.stack([dx, dy])

    return news

def HenonSystem_J(s):
    A = 1.4
    B = 0.3

    x = s[0]
    y = s[1]
    
    J_cont = np.zeros((2, 2))
    
    J_cont[0, 0] = -2 * A * x
    J_cont[0, 1] = 1
    
    J_cont[1, 0] = b
    J_cont[1, 1] = 0
    
    return J_cont

#====================================================================

### メイン関数 ###
def Main():

    #初期状態の設定
    s_old = (np.ones((3)) * (np.random.rand(3) * 2 - 1))

    #リアプノフ指数の格納用配列
    lamdas_t_qr = np.zeros((time_steps, 3))
    lyapunov_exponents_qr = np.zeros(3)

    #空走時間
    for i in range(1000):
        s_new = ChialvoSystem(s_old, a, b, c, k, k0, k1, k2, alpha, beta)
        s_old = s_new

    Q_old = np.eye(3)

    for j in range(time_steps):

        #Chialvoの更新式
        s_new = ChialvoSystem(s_old, a, b, c, k, k0, k1, k2, alpha, beta)
        #ヤコビ行列の更新
        Chialvo_J = ChialvoSystem_J(s_old, a, b, c, k, k0, k1, k2, alpha, beta)  @ Q_old
        
        #QR分解
        Q_old, R_old = np.linalg.qr(Chialvo_J)
        
        #上三角行列の対角成分を取得
        lyapunov_exponents_qr = lyapunov_exponents_qr + np.log(np.abs(np.diag(R_old)))

        s_old = s_new

    lamdas_math_qr = lyapunov_exponents_qr / time_steps
    l_x = round(lamdas_math_qr[0], 8)
    l_y = round(lamdas_math_qr[1], 8)
    l_p = round(lamdas_math_qr[2], 8)
    
    lll = [l_x,l_y,l_p]

    print(lll)

#Main()

def k_loop():

    #パラメータkのリスト
    k_lista = np.arange(-7, 7.1, 0.01)
    k_list = np.round(k_lista, 4)

    #各リアプノフスペクトルの結果収納用配列
    l_x = np.zeros(len(k_list))
    l_y = np.zeros(len(k_list))
    l_p = np.zeros(len(k_list))

    lll = np.zeros(((len(k_list)), 3))

    #配列用個別変数
    ex = 0

    for z in range(0,len(k_list)):
        
        #kの値変更
        k = k_list[z]
        print(f"kの値: {k}")

        #初期状態の設定
        s_old = (np.ones((3)) * (np.random.rand(3) * 2 - 1))

        #リアプノフ指数の格納用配列
        lamdas_t_qr = np.zeros((time_steps, 3))
        lyapunov_exponents_qr = np.zeros(3)

        #空走時間
        for i in range(1000):
            s_new = ChialvoSystem(s_old, a, b, c, k, k0, k1, k2, alpha, beta)
            s_old = s_new

        Q_old = np.eye(3)

        for j in range(time_steps):

            #Chialvoの更新式
            s_new = ChialvoSystem(s_old, a, b, c, k, k0, k1, k2, alpha, beta)
            #ヤコビ行列の更新
            Chialvo_J = ChialvoSystem_J(s_old, a, b, c, k, k0, k1, k2, alpha, beta)  @ Q_old
            
            #QR分解
            Q_old, R_old = np.linalg.qr(Chialvo_J)
            
            #上三角行列の対角成分を取得
            lyapunov_exponents_qr = lyapunov_exponents_qr + np.log(np.abs(np.diag(R_old)))

            s_old = s_new

        lamdas_math_qr = lyapunov_exponents_qr / time_steps
        l_x[ex] = round(lamdas_math_qr[0], 6)
        l_y[ex] = round(lamdas_math_qr[1], 6)
        l_p[ex] = round(lamdas_math_qr[2], 6)

        lll[ex,:] = np.stack([l_x[ex],l_y[ex],l_p[ex]])
        print(lll[ex,:])

        ex = ex + 1
    
    fig = plt.figure(figsize = FigSize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(k_list,l_x,'.', label = "x")
    ax.plot(k_list,l_y,'.', label = "y")
    ax.plot(k_list,l_p,'.', label = "phi")
    ax.set_xlabel("k", fontsize = FontSize_Axis)
    ax.set_ylabel("Lyapunov", fontsize = FontSize_Axis)
    ax.grid()
    ax.set_xticks(np.arange(-7, 7.01, 1))
    ax.legend()
    fig.tight_layout()

    plt.show()

#k_loop()

#====================================================================

### テスト関数 ###

def Test():
    
    #初期状態の設定
    s_old = (np.ones((3)) * (np.random.rand(3) * 2 - 1)) * 0.1

    #リアプノフ指数の格納用配列
    lamdas_t_qr = np.zeros((time_steps, 3))
    lyapunov_exponents_qr = np.zeros(3)

    #空走時間
    for i in range(1000):
        s_new = LorenzSystem(s_old, sigma, rhos, beta_Lo, dt)
        s_old = s_new

    Q_old = np.eye(3)

    for j in range(time_steps):
        #Chialvoの更新式
        s_new = LorenzSystem(s_old, sigma, rhos, beta_Lo, dt)
        #ヤコビ行列の更新
        Lorenz_J = LorenzSystem_J(s_old, sigma, rhos, beta_Lo, dt)  @ Q_old
        
        #QR分解
        Q_old, R_old = np.linalg.qr(Lorenz_J)
        
        #上三角行列の対角成分を取得
        lyapunov_exponents_qr = lyapunov_exponents_qr + np.log(np.abs(np.diag(R_old)))

        s_old = s_new

    lamdas_math_qr = lyapunov_exponents_qr / time_steps
    l_x = round(lamdas_math_qr[0], 8)
    l_y = round(lamdas_math_qr[1], 8)
    l_p = round(lamdas_math_qr[2], 8)
    
    lll = [l_x,l_y,l_p]

    print(lll)

Test()

def Test_H():

    #初期状態の設定
    s_old = (np.ones((2)) * (np.random.rand(2))) * 0.01

    #リアプノフ指数の格納用配列
    lamdas_t_qr = np.zeros((time_steps, 2))
    lyapunov_exponents_qr = np.zeros(2)

    #空走時間
    for i in range(1000):
        s_new = HenonSystem(s_old)
        s_old = s_new

    Q_old = np.eye(2)

    for j in range(time_steps):

        #Chialvoの更新式
        s_new = HenonSystem(s_old)
        #ヤコビ行列の更新
        Henon_J = HenonSystem_J(s_old)  @ Q_old
        
        #QR分解
        Q_old, R_old = np.linalg.qr(Henon_J)
        
        #上三角行列の対角成分を取得
        lyapunov_exponents_qr = lyapunov_exponents_qr + np.log(np.abs(np.diag(R_old)))

        s_old = s_new

    lamdas_math_qr = lyapunov_exponents_qr / time_steps
    l_x = round(lamdas_math_qr[0], 8)
    l_y = round(lamdas_math_qr[1], 8)

    lll = [l_x,l_y]

    print(lll)


#Test_H()
#====================================================================
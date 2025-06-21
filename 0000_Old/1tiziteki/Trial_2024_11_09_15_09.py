### リアプノフスペクトル算出 ###
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

### メイン ###
def Main():
    Now = datetime.datetime.now()
    ProjectFile_Path = "Data/SimpleEvaluations"
    Figure_Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
    Figure_Name = "MaxLyapunovCheck_" + Figure_Date + "_"
    Result_Path = os.path.join(ProjectFile_Path, "Results")
    UFigure_Name = "MaxLyapunovCheck_Lorenz_" + Figure_Date
    UFigure_Path = os.path.join(Result_Path, UFigure_Name)

    # ディレクトリの作成
    os.makedirs(UFigure_Path, exist_ok=True)
    
    # 描画設定
    FigSize = (16, 9)
    FontSize_Label = 24
    FontSize_Title = 24
    LineWidth = 3
    FileFormat = ".png"
    
    fig = plt.figure(figsize=FigSize)
    ax = fig.add_subplot(2, 1, 1)
    
    time_steps = 1000

    # パラメータ設定
    rhos = 28#np.linspace(0.1, 28, num_of_points)
    sigma = 10#10
    beta = 8/3#8./3.
    dt = 1./1.

    a = 0.89
    b = 0.6
    c = 0.28
    k0 = 0.04

    k_chialvo = 3.2

    k1 = 0.1
    k2 = 0.2
    alpha = 0.1
    beta_chialvo = 0.2

    # 初期状態の設定
    ss = 1e-5 * np.ones((3))
    ps = np.eye(3) / np.linalg.norm(np.eye(3), axis=1, keepdims=True) * 1e-7
    deltas_0 = np.linalg.norm(ps, axis=1)
    
    ss_t = np.zeros((time_steps, 3))

    # リアプノフ指数の格納用配列
    lamdas_t_pert_gram = np.zeros((time_steps, 3))
    lamdas_t_pert_qr = np.zeros((time_steps, 3))
    lamdas_t_svd = np.zeros((time_steps, 3))
    lamdas_t_qr = np.zeros((time_steps, 3))
    lyapunov_exponents_pert_gram = np.zeros(3)
    lyapunov_exponents_pert_qr = np.zeros(3)
    lyapunov_exponents_svd = np.zeros(3)
    lyapunov_exponents_qr = np.zeros(3)

    # 累積ヤコビアン行列の初期化
    U = np.eye(3)
    Q = np.eye(3)

    # 初期過渡状態の除去
    for i in range(10000):
        ss = ChialvoSystem(ss, a, b, c, k_chialvo, k0, k1, k2, alpha, beta_chialvo)
        
    ss_p_qr = ss + ps
    ss_p_gram = ss + ps

    # メインループ
    for i in range(time_steps):
        ss_t[i] = ss

        #Js = LorenzSystem_J(ss, sigma, rhos, beta, dt)
        #ss = LorenzSystem(ss, sigma, rhos, beta, dt)
        #ss_p_qr = LorenzSystem(ss_p_qr.T, sigma, rhos, beta, dt).T
        #ss_p_gram = LorenzSystem(ss_p_gram.T, sigma, rhos, beta, dt).T
        ss = ChialvoSystem(ss, a, b, c, k_chialvo, k0, k1, k2, alpha, beta_chialvo)
        ss_p_gram = ChialvoSystem(ss_p_gram.T, a, b, c, k_chialvo, k0, k1, k2, alpha, beta_chialvo).T
        Js = ChialvoSystem_J(ss, a, b, c, k_chialvo, k0, k1, k2, alpha, beta_chialvo)

        # 摂動法によるリアプノフ指数の計算
        d_tau = ss_p_gram - ss
        d_tau_norm = np.linalg.norm(d_tau, axis = 1, ord = 2)
        #d_tau = d_tau[np.argsort(d_tau_norm)[::-1]]
        d_Orthogonal = d_tau.copy()
        norm_d_Orthogonal = np.zeros((3))
        for k in range(3):
            for l in range(k):
                d_Orthogonal[k] -= np.dot(d_tau[k], d_Orthogonal[l]) / (norm_d_Orthogonal[l]**2 + 1e-20) * d_Orthogonal[l]
            norm_d_Orthogonal[k] = np.linalg.norm(d_Orthogonal[k], ord = 2)
            
        ss_p_gram = np.array([ss + deltas_0[k] / norm_d_Orthogonal[k] * d_Orthogonal[k] for k in range(3)])
        lyapunov_exponents_pert_gram += np.log(norm_d_Orthogonal / deltas_0 + 1e-20)
        lamdas_t_pert_gram[i] = lyapunov_exponents_pert_gram / ((i + 1) * dt)
        
        # 摂動法によるリアプノフ指数の計算
        """d_tau = ss_p_qr - ss
        d_tau_norm = np.linalg.norm(d_tau, axis = 1, ord = 2)
        #d_tau = d_tau[np.argsort(d_tau_norm)[::-1]]
        pQ, pR = np.linalg.qr(d_tau.T)
        ss_p_qr = np.array([ss + deltas_0[k] * pQ.T[k] for k in range(3)])
        lyapunov_exponents_pert_qr += np.log(np.abs(np.diag(pR)) / deltas_0 + 1e-20)
        lamdas_t_pert_qr[i] = lyapunov_exponents_pert_qr / ((i + 1) * dt)"""

        # SVD法
        """A = Js @ U
        U, S, Vh = np.linalg.svd(A)
        U = (U @ np.eye(3) @ Vh).T
        lyapunov_exponents_svd += np.log(S + 1e-20)
        lamdas_t_svd[i] = lyapunov_exponents_svd / ((i + 1) * dt)"""

        # QR法
        Q = Js @ Q
        Q, R = np.linalg.qr(Q)
        lyapunov_exponents_qr += np.log(np.abs(np.diag(R)) + 1e-20)
        lamdas_t_qr[i] = lyapunov_exponents_qr / ((i + 1) * dt)
                
        print("\r progress : %d / %d" % (i + 1, time_steps), end="")
    print("")
    
    # リアプノフ指数の計算
    lamdas_pert_gram = lyapunov_exponents_pert_gram / (time_steps * dt)
    #lamdas_pert_qr = lyapunov_exponents_pert_qr / (time_steps * dt)
    #lamdas_math_svd = lyapunov_exponents_svd / (time_steps * dt)
    lamdas_math_qr = lyapunov_exponents_qr / (time_steps * dt)

    # グラフの描画設定
    ax.plot(np.linspace(0, time_steps*dt, time_steps), ss_t[:, 0], label="x")
    ax.plot(np.linspace(0, time_steps*dt, time_steps), ss_t[:, 1], label="y")
    ax.plot(np.linspace(0, time_steps*dt, time_steps), ss_t[:, 2], label="z")

    ax.set_title('Lorenz System Orbits', fontsize=FontSize_Title)
    ax.set_xlabel('rho', fontsize=FontSize_Label)
    ax.set_ylabel('orbit', fontsize=FontSize_Label)
    ax.grid(True)
    
    print("Pert Gram : " + str(lamdas_pert_gram) + ", QR : " + str(lamdas_math_qr))
    
    # リアプノフ指数のプロット
    ax2 = fig.add_subplot(2, 1, 2)
    
    ax2.plot(np.linspace(0, time_steps*dt, time_steps), lamdas_t_pert_gram, label="Lyapunov Perturbation Gram")
    #ax2.plot(np.linspace(0, time_steps*dt, time_steps), lamdas_t_pert_qr, label="Lyapunov Perturbation QR")
    #ax2.plot(np.linspace(0, time_steps*dt, time_steps), lamdas_t_svd, label="Lyapunov SVD")
    ax2.plot(np.linspace(0, time_steps*dt, time_steps), lamdas_t_qr, label="Lyapunov QR")

    #ax2.plot(rhos, lamdas_pert, label="Lyapunov Perturbation")
    #ax2.plot(rhos, lamdas_math_svd, label="Lyapunov SVD")
    #ax2.plot(rhos, lamdas_math_qr, label="Lyapunov QR")

    ax2.set_title('Corresponding MLE', fontsize=FontSize_Title)
    ax2.set_xlabel('rho', fontsize=FontSize_Label)
    ax2.set_ylabel('MLE', fontsize=FontSize_Label)
    ax2.grid(True)
    ax2.legend()

    # グラフの保存と表示
    fig.savefig(os.path.join(UFigure_Path, Figure_Name + "TestResults" + FileFormat))
    plt.show()
    
#def LorenzSystem(s, sigma, rho, beta, dt):
#    y = s[0]
#    z = s[1]
#    x = s[2]

#    dx = sigma * (y - x)
#    dy = x * (rho - z) - y
#    dz = x * y - beta * z

#    ds = np.stack([dy, dz, dx])
#    new_s = s + dt * ds

#    return new_s

#def LorenzSystem_J(s, sigma, rho, beta, dt):
#    y = s[0]
#    z = s[1]
#    x = s[2]
    
#    J_cont = np.zeros((3, 3))
    
#    J_cont[0, 0] = -1
#    J_cont[0, 1] = -x
#    J_cont[0, 2] = (rho - z)

#    J_cont[1, 0] = x
#    J_cont[1, 1] = -beta
#    J_cont[1, 2] = y

#    J_cont[2, 0] = sigma
#    J_cont[2, 1] = 0
#    J_cont[2, 2] = -sigma

#    # 離散時間系のヤコビアン
#    J_map = np.eye(3) + dt * J_cont

#    return J_map

#def LorenzSystem(s, sigma, rho, beta, dt):
#    z = s[0]
#    x = s[1]
#    y = s[2]

#    dx = sigma * (y - x)
#    dy = x * (rho - z) - y
#    dz = x * y - beta * z

#    ds = np.stack([dz, dx, dy])
#    new_s = s + dt * ds

#    return new_s

#def LorenzSystem_J(s, sigma, rho, beta, dt):
#    z = s[0]
#    x = s[1]
#    y = s[2]
    
#    J_cont = np.zeros((3, 3))
    
#    J_cont[0, 0] = -beta
#    J_cont[0, 1] = y
#    J_cont[0, 2] = x

#    J_cont[1, 0] = 0
#    J_cont[1, 1] = -sigma
#    J_cont[1, 2] = sigma

#    J_cont[2, 0] = -x
#    J_cont[2, 1] = (rho - z)
#    J_cont[2, 2] = -1

#    # 離散時間系のヤコビアン
#    J_map = np.eye(3) + dt * J_cont

#    return J_map

def LorenzSystem(s, sigma, rho, beta, dt):
    x = s[0]
    y = s[1]
    z = s[2]

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    ds = np.stack([dx, dy, dz])
    new_s = s + dt * ds

    return new_s

def ChialvoSystem(old_s, a, b, c, k, k0, k1, k2, alpha, beta):
    x = old_s[0]
    y = old_s[1]
    phi = old_s[2]

    dx = pow(x, 2) * np.exp(y - x) + k0 + k * x * (alpha + 3 * beta *pow(phi, 2))
    dy = a * y - b * x + c
    dphi = k1 * x - k2 * phi

    new_s = np.stack([dx, dy, dphi])

    return new_s


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

    # 離散時間系のヤコビアン
    J_map = J_cont

    return J_map

Main()
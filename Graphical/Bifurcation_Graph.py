import numpy as np
import scipy.linalg
import time

#電磁束下Chialvoのパラメータ
a = 0.89
b = 0.6
c = 0.28
k0 = 0.04

k1 = 0.1
k2 = 0.2
alpha = 0.1
beta = 0.2

k = -22

#入力信号
InputSingal = 0

Runtime = 10000

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

#開始, 終了地点
Plot_Start = -1
Plot_End = 1

#交点を求める関数
Derive_of_Intersections = False
#固定点の導出
Derive_of_FixedPoint = True
#リアプノフ指数
Lyapunov_expotent = False

#固定点を加えるためのリスト
FixedPoint_List = [0.013131072] #list()[]

"""------------------------------------------------------------"""
if Derive_of_Intersections:
    print("\nDerive_of_Intersections")
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
    print("\n\nFixed Point")
    print(f"{FixedPoint_List}")
    print(f"実行時間: {Time_Diff}")

"""------------------------------------------------------------"""
if Derive_of_FixedPoint:
    print("\nDerive_of_FixedPoint")

    for i in FixedPoint_List:
        x = i
        y = round((-b * x + c) / (1 - a), round_Pre)
        phi = round((k1 * x) / (1 + k2), round_Pre)

        print(f"Fixed Point: x = {x}, y = {y}, phi = {phi}")
        Jac_Matrix = np.array([[np.exp(y - x) * (2 * x - pow(x, 2)) + k * (alpha + 3 * beta * pow(phi, 2)), pow(x, 2) * np.exp(y - x), 6 * k * x * beta * phi],\
                                [-b, a, 0], \
                                    [k1, 0, -k2]])
        
        Eig_Val= np.linalg.eigvals(Jac_Matrix)
        print(f"Eigenvalue = {Eig_Val}")
"""------------------------------------------------------------"""
if Lyapunov_expotent:
    print("Lyapunov_expotent")

    x = np.random.uniform(-1,1,Runtime)
    x_Lya = np.random.uniform(x[0] - 1e-04, x[0] + 1e-04, Runtime)

    y = np.random.uniform(-1,1,Runtime)
    y_Lya = np.random.uniform(y[0] - 1e-04, y[0] + 1e-04,Runtime)
    
    phi = np.random.uniform(-1,1,Runtime)
    phi_Lya = np.random.uniform(phi[0] - 1e-04, phi[0] + 1e-04,Runtime)
    
    Input_Signal_In = np.zeros(Runtime)
    Input_Signal_In[:] = InputSingal
    
    Lyapunov_Lambda =  np.zeros(Runtime)

    for i in range(Runtime - 1):
        print("\r%d / %d"%(i, Runtime), end = "")
        x[i+1] = pow(x[i], 2) * np.exp(y[i] - x[i]) + k0 \
              + k * x[i] * (alpha + 3 * beta * pow(phi[i], 2)) \
                + Input_Signal_In[i]
        y[i+1] = a * y[i] - b * x[i] + c
        phi[i+1] = k1 * x[i] - k2 * phi[i]

        x_Lya[i+1] = pow(x_Lya[i], 2) * np.exp(y_Lya[i] - x_Lya[i]) + k0 \
              + k * x_Lya[i] * (alpha + 3 * beta * pow(phi_Lya[i], 2)) \
                + Input_Signal_In[i]
        y_Lya[i+1] = a * y_Lya[i] - b * x_Lya[i] + c
        phi_Lya[i+1] = k1 * x_Lya[i] - k2 * phi_Lya[i]

        Lyapunov_Lambda = x_Lya[i] - x[i]
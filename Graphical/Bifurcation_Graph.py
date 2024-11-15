import numpy as np
import scipy.linalg
import time

#電磁束下Chialvoのパラメータ
a = 0.5
b = 0.4
c = 0.89
k0 = -0.44
k1 = 0.1
k2 = 0.2
alpha = 0.1
beta = 0.1
k = 0

#入力信号
InputSingal = 0

setting = {
    "dx"    : 1e-05,
    "dx_R"  : 1e+05,
    "round" : 5
}

#計測間隔
dx = setting["dx"]
print(f"計測間隔：{dx}")
dx_R = int(setting["dx_R"])
print(f"計測間隔の逆数：{dx_R}")
round_Pre = setting["round"]

#開始, 終了地点
Plot_Start = -10
Plot_End = 10

#交点を求める関数
Derive_of_Intersections = True
#固定点の導出
Derive_of_FixedPoint = True

#固定点を加えるためのリスト
FixedPoint_List = list()

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
                                [-b, a, 0],\
                                    [k1, 0, -k2]])
        
        Eig_Val = np.linalg.eigvals(Jac_Matrix)
        print(f"Eigenvalue = {Eig_Val}\n")
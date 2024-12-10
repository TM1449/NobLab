#====================================================================
import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d, Axes3D


#====================================================================
#入力信号
InputSingal = 0

#None, sin, cos（ローレンツ方程式）
InputSingal_def = None

#====================================================================
#パラメータ
sigma = 10
rho = 28
beta = 8/3

#--------------------------------------------------------------------
#時間刻み幅
dt = 1/1000
#時間刻み幅の逆数
Reciprocal_dt = 1 / dt
#時間刻み幅の逆数の四捨五入
Reciprocal_dt_round = round(Reciprocal_dt)

#--------------------------------------------------------------------
print("\n====================================================================")
print("====================================================================")

#時間刻み幅の表示
print(f"\n時間刻み幅: {dt}")
#時間刻み幅の逆数表示
print(f"時間刻み幅の逆数: {Reciprocal_dt}")
#時間刻み幅の逆数表示＋四捨五入
print(f"時間刻み幅の逆数＋四捨五入: {Reciprocal_dt_round}")

#--------------------------------------------------------------------
#空走時間
Burn_in_time = 100 * Reciprocal_dt_round

#計算回数（計算時間）
Calculation_Number = 1000

#何時刻差のリアプノフ指数を計算するか
Step = 200

#--------------------------------------------------------------------
#総合時間
All_Number = Calculation_Number * Step


#--------------------------------------------------------------------
print("\n--------------------------------------------------------------------")
print(f"\n空走時間: {Burn_in_time / Reciprocal_dt_round}, 計算回数: {Calculation_Number} 回")
print(f"空走ステップ数: {Burn_in_time}, 計算時間: {Calculation_Number * dt}")
print(f"\n何時間先のリアプノフ指数を計算するか: {Step / Reciprocal_dt_round}, ステップ数: {Step}")

#====================================================================
print("\n====================================================================")
print("====================================================================")
print("\n電磁束下Chialvoニューロン: 最大リアプノフ指数の導出")

#--------------------------------------------------------------------
#初期値の指定
Initial_x = None #0.0448
Initial_y = None #2.3011
Initial_z = None #0.0037

#--------------------------------------------------------------------
#基準軌道xと初期値について
x = np.zeros(Burn_in_time + All_Number + 1)
if Initial_x == None:
    x[0] = np.random.random() * 0.2 - 0.1
else:
    x[0] = Initial_x

#--------------------------------------------------------------------
#基準軌道yと初期値について
y = np.zeros(Burn_in_time + All_Number + 1)
if Initial_y == None:
    y[0] = np.random.random() * 0.2 - 0.1
else:
    y[0] = Initial_y

#--------------------------------------------------------------------
#基準軌道phiと初期値について
z = np.zeros(Burn_in_time + All_Number + 1)
if Initial_z == None:
    z[0] = np.random.random() * 0.2 - 0.1
else:
    z[0] = Initial_z

#====================================================================
print("\n--------------------------------------------------------------------")
#規準軌道の計算部
print("\n基準軌道の計算")
for i in range(0, Burn_in_time + All_Number):
    print("\r%d / %d" % (i, All_Number + Burn_in_time), end="")

    x[i+1] = x[i] + dt * sigma * (y[i] - x[i])
    y[i+1] = y[i] + dt * (x[i] * (rho - z[i]) - y[i])
    z[i+1] = z[i] + dt * (x[i] * y[i] - beta * z[i])

print("\n\n--------------------------------------------------------------------")
print("\n空走時間を除外した、ローレンツ方程式の基準軌道")

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection = '3d')
ax.plot(x[Burn_in_time:], y[Burn_in_time:], z[Burn_in_time:])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.grid()
plt.show()

#====================================================================
#最大リアプノフ指数を取る配列
Lyapunov = np.zeros(Calculation_Number)
#最大リアプノフ指数の総合
Lyapunov_Sum = 0

#摂動軌道を取る配列
x_D = np.zeros(Calculation_Number+1)
y_D = np.zeros(Calculation_Number+1)
z_D = np.zeros(Calculation_Number+1)

#摂動軌道の初期値を確保する配列
delta_0_x = np.zeros(Calculation_Number)
delta_0_y = np.zeros(Calculation_Number)
delta_0_z = np.zeros(Calculation_Number)

delta_0_x[0] = 1e-08
delta_0_y[0] = 1e-08
delta_0_z[0] = 1e-08

#摂動軌道のを発展先の値を確保する配列
delta_tau_x = np.zeros(Calculation_Number)
delta_tau_y = np.zeros(Calculation_Number)
delta_tau_z = np.zeros(Calculation_Number)

delta_tau_x[0] = 1e-08
delta_tau_y[0] = 1e-08
delta_tau_z[0] = 1e-08


#====================================================================
#初期値をぶち込んで、Step後の値を出す
def lorenz(x, y, z):
    x_n = x
    y_n = y
    z_n = z

    for i in range(0, Step):
        x_n1 = x_n + dt * (sigma * (y_n - x_n))
        y_n1 = y_n + dt * (x_n * (rho - z_n) - y_n)
        z_n1 = z_n + dt * (x_n * y_n - beta * z_n)

        x_n = x_n1
        y_n = y_n1
        z_n = z_n1
    
    return x_n1, y_n1, z_n1

#====================================================================
#基準軌道に摂動を与える
x_D[0] = x[Burn_in_time] + delta_0_x[0]
y_D[0] = y[Burn_in_time] + delta_0_y[0]
z_D[0] = z[Burn_in_time] + delta_0_z[0]

#関数に初期値をぶち込み、Step数だけ発展させた値を算出
x_D[1], y_D[1], z_D[1] = lorenz(x_D[0], \
                                y_D[0], \
                                z_D[0])

#基準軌道の初期値の点
Standard_0 = np.array([x[Burn_in_time], y[Burn_in_time], z[Burn_in_time]])
#基準軌道の時間発展させた点
Standard_tau = np.array([x[Burn_in_time + Step], y[Burn_in_time + Step], z[Burn_in_time + Step]])

#摂動軌道の初期値の点
Pert_0 = np.array([x_D[0], y_D[0], z_D[0]])
#摂動軌道の時間発展させた点
Pert_tau = np.array([x_D[1], y_D[1], z_D[1]])

print("\n--------------------------------------------------------------------")
print(f"\n基準軌道の初期値の点: {Standard_0}\n摂動軌道の初期値の点: {Pert_0}")
print(f"基準軌道の発展先の点: {Standard_tau}\n摂動軌道の初期値の点: {Pert_tau}")

#初期値の差ベクトル
Vector_0 = (Pert_0 - Standard_0)
#時間発展させた差ベクトル
Vector_tau = (Pert_tau - Standard_tau)

#初期値の差ベクトルの大きさ
Norm_0 = np.linalg.norm(Vector_0)
#時間発展させた差ベクトルの大きさ
Norm_tau = np.linalg.norm(Vector_tau)

Lyapunov[0] = np.log(Norm_tau / Norm_0) / (Step * dt)
print(f"\n1回目の最大リアプノフ指数: {Lyapunov[0]}")
Lyapunov_Sum = Lyapunov_Sum + Lyapunov[0]
print("\n====================================================================")

#====================================================================
for i in range(1, Calculation_Number):
    #時間発展させた差ベクトルの向きに対して、摂動を与える
    delta_0_x[i] =  (Vector_tau[0] / Norm_tau) * delta_0_x[0]
    delta_0_y[i] =  (Vector_tau[1] / Norm_tau) * delta_0_y[0]
    delta_0_z[i] =  (Vector_tau[2] / Norm_tau) * delta_0_z[0]

    #基準軌道に初期摂動を与える
    x_D[i] = x[Burn_in_time + (i*Step)] + delta_0_x[i]
    y_D[i] = y[Burn_in_time + (i*Step)] + delta_0_y[i]
    z_D[i] = z[Burn_in_time + (i*Step)] + delta_0_z[i]
    
    #関数に初期値をぶち込み、Step数だけ発展させた値を算出
    x_D[i+1], y_D[i+1], z_D[i+1] = lorenz(x_D[i], \
                                    y_D[i], \
                                    z_D[i])

    #基準軌道の初期値の点
    Standard_0 = np.array([x[Burn_in_time + (i * Step)], y[Burn_in_time + (i * Step)], z[Burn_in_time + (i * Step)]])
    #基準軌道の時間発展させた点
    Standard_tau = np.array([x[Burn_in_time + ((i+1) * Step)], y[Burn_in_time + ((i+1) * Step)], z[Burn_in_time + ((i+1) * Step)]])

    #摂動軌道の初期値の点
    Pert_0 = np.array([x_D[i], y_D[i], z_D[i]])
    #摂動軌道の時間発展させた点
    Pert_tau = np.array([x_D[i+1], y_D[i+1], z_D[i+1]])

    print("\n--------------------------------------------------------------------")
    print(f"\n基準軌道の初期値の点: {Standard_0}\n摂動軌道の初期値の点: {Pert_0}")
    print(f"基準軌道の発展先の点: {Standard_tau}\n摂動軌道の初期値の点: {Pert_tau}")

    #初期値の差ベクトル
    Vector_0 = (Pert_0 - Standard_0)
    #時間発展させた差ベクトル
    Vector_tau = (Pert_tau - Standard_tau)

    #初期値の差ベクトルの大きさ
    Norm_0 = np.linalg.norm(Vector_0)
    #時間発展させた差ベクトルの大きさ
    Norm_tau = np.linalg.norm(Vector_tau)

    Lyapunov[i] = np.log(Norm_tau / Norm_0) / (Step * dt)
    print(f"\n{i+2}回目の最大リアプノフ指数: {Lyapunov[i]}")
    Lyapunov_Sum = Lyapunov_Sum + Lyapunov[i]
    print("\n====================================================================")


Lyapunov_Ave = Lyapunov_Sum / Calculation_Number
print(Lyapunov_Ave)
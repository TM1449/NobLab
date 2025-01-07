#====================================================================
import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt


#====================================================================
#ロジスティック写像のパラメータ
a = 4 #0.89 0.6

##ロジスティック写像のパラメータ配列
a_list = np.arange(0,4.001,0.1)
a_list_result = np.copy(a_list)
#print(a_list)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#空走時間
Burn_in_time = 500

#実行時間
Runtime = 10000

#全体時間
Alltime = Burn_in_time + Runtime

#一回のみ
Lyapunov_expotent = True
#リストの分
Lyapunov_expotent_List = False

#====================================================================
if Lyapunov_expotent:
    print("\nロジスティック写像: リアプノフ指数の導出")
    #初期値の指定
    Initial_x = None #0.0448
    
    #--------------------------------------------------------------------
    #基準軌道xと初期値について
    x = np.zeros(Alltime)

    if Initial_x == None:
        x[0] = np.random.random()
    else:
        x[0] = Initial_x

    #--------------------------------------------------------------------
    #規準軌道の計算部
    print("\n基準軌道の計算")
    for i in range(Alltime-1):
        print("\r%d / %d"%(i, Alltime), end = "")
        x[i+1] = a * x[i] *(1 - x[i])
    
    #--------------------------------------------------------------------
    #リアプノフ指数を取る配列
    x_Lyapunov = np.zeros(Runtime-1)
    
    #初期摂動が与えられた軌道
    x_D = np.zeros(Runtime)
    
    #総和を取る配列
    x_sum = 0
    
    #初期摂動を確保する配列
    delta_0_x = np.zeros(Runtime)
    delta_0_x[0] = 1e-06
    
    #初期摂動を1step発展させる配列
    delta_tau_x = np.zeros(Runtime)
    delta_tau_x[0] = 1e-06
    
    #--------------------------------------------------------------------
    #規準軌道に初期摂動を与える
    x_D[0] = x[Burn_in_time] + delta_0_x[0]
    
    #初期摂動を与えたものを1step時間発展させた
    x_D[1] = a * x_D[0] * (1 - x_D[0])

    #時間発展させた2つの軌道の差の算出
    delta_tau_x[0] = x_D[1] - x[Burn_in_time+1]
    
    #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
    #（負ならば差が縮まる、正なら差が広がる）
    x_Lyapunov[0] = np.log(abs(delta_tau_x[0]) / abs(delta_0_x[0]))
    x_sum += x_Lyapunov[0]
    
    print("\n\n摂動軌道の計算")
    #摂動軌道の計算部
    for i in range(1,Runtime-1):
        print("\r%d / %d"%(i, Runtime), end = "")
        delta_0_x[i] = (delta_tau_x[i-1] / abs(delta_tau_x[i-1])) * delta_0_x[0]
    
        #規準軌道に初期摂動を与える
        x_D[i] = x[Burn_in_time+i] + delta_0_x[i]
        
        #初期摂動を与えたものを1step時間発展させた
        x_D[i+1] = a * x_D[i] * (1 - x_D[i])
    
        #時間発展させた2つの軌道の差の算出
        delta_tau_x[i] = x_D[i+1] - x[Burn_in_time+i+1]
    
        #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
        #（負ならば差が縮まる、正なら差が広がる）
        x_Lyapunov[i] = np.log(abs(delta_tau_x[i]) / abs(delta_0_x[i]))
        x_sum += x_Lyapunov[i]

    #時間平均でのリアプノフ指数
    x_ave = x_sum / len(x_Lyapunov)
    
    print("\nロジスティック写像のリアプノフ指数")
    print(x_ave)

if Lyapunov_expotent_List:
    #a_listに入っている変数で実行
    print("\nロジスティック写像: 各パラメータにおけるリアプノフ指数の導出")
    z = 0
    for a in a_list:
        print({f"{z+1} / {len(a_list_result)} 回"})
        print({f"aの値: {a}"})
        #初期値の指定
        Initial_x = None
        
        #--------------------------------------------------------------------
        #基準軌道xと初期値について
        x = np.zeros(Alltime+1)

        if Initial_x == None:
            x[0] = np.random.random()
        else:
            x[0] = Initial_x

        #--------------------------------------------------------------------
        #規準軌道の計算部
        
        #空走時間の計算部
        for i in range(Alltime):
            print("\r%d / %d"%(i, Alltime), end = "")
            x[i+1] = a * x[i] *(1 - x[i])
        
        #--------------------------------------------------------------------
        #リアプノフ指数を取る配列
        x_Lyapunov = np.zeros(Runtime-1)
        
        #初期摂動が与えられた軌道
        x_D = np.zeros(Runtime+1)
        
        #総和を取る配列
        x_sum = 0
        
        #初期摂動を確保する配列
        delta_0_x = np.zeros(Runtime)
        delta_0_x[0] = 1e-06
        
        #初期摂動を1step発展させる配列
        delta_tau_x = np.zeros(Runtime)
        delta_tau_x[0] = 1e-06
        
        #--------------------------------------------------------------------
        #規準軌道に初期摂動を与える
        x_D[0] = x[Burn_in_time] + delta_0_x[0]
        
        #初期摂動を与えたものを1step時間発展させた
        x_D[1] = a * x_D[0] * (1 - x_D[0])
        x_D[2] = a * x_D[1] * (1 - x_D[1])

        #時間発展させた2つの軌道の差の算出
        delta_tau_x[0] = x_D[2] - x[Burn_in_time+2]
        
        #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
        #（負ならば差が縮まる、正なら差が広がる）
        x_Lyapunov[0] = np.log(abs(delta_tau_x[0]) / abs(delta_0_x[0]))
        x_sum += x_Lyapunov[0]
        
        print("\n\n摂動軌道の計算")
        #摂動軌道の計算部
        for i in range(1,Runtime-1):
            print("\r%d / %d"%(i, Runtime), end = "")
            delta_0_x[i] = (delta_tau_x[i-1] / abs(delta_tau_x[i-1])) * delta_0_x[0]
        
            #規準軌道に初期摂動を与える
            x_D[i] = x[Burn_in_time+i] + delta_0_x[i]
            
            #初期摂動を与えたものを1step時間発展させた
            x_D[i+1] = a * x_D[i] * (1 - x_D[i])
            x_D[i+2] = a * x_D[i+1] * (1 - x_D[i+1])
        
            #時間発展させた2つの軌道の差の算出
            delta_tau_x[i] = x_D[i+2] - x[Burn_in_time+i+2]
        
            #絶対値を取り、初期摂動の差と時間発展した差がどれくらいあるかをlogで算出
            #（負ならば差が縮まる、正なら差が広がる）
            x_Lyapunov[i] = np.log(abs(delta_tau_x[i]) / abs(delta_0_x[i]))
            x_sum += x_Lyapunov[i]

        x_ave = 0.5 * x_sum / len(x_Lyapunov)
        
        print("\nロジスティック写像のリアプノフ指数")
        print(x_ave)

        a_list_result[z] = x_ave
        z = z+1
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.grid()
    ax.plot(a_list_result)
    
    plt.show()


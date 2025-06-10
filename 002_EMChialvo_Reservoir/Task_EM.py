#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大 田中勝規
#作成日：2023/05/24
"""
本体

maru
"""

#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・ノイズ入りタスクは未移植＠Ver1
#・少数入力次元が影響力のあるタスク未実装
#・分類タスク未実装

#====================================================================
import numpy as np
import signalz
from scipy.integrate import solve_ivp
import ddeint
import bisect
import os

#====================================================================
#タスク

#********************************************************************
#継承元
class Task:
    """
    タスク
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        self.Param = param

        self.Evaluation = evaluation                    #親評価オブジェクト

        #パラメータ取得
        self.D_u = self.Param["Task_D_u"]               #入力信号次元
        self.D_y = self.Param["Task_D_y"]               #出力信号次元
        self.Length = self.Param["Task_Length"]         #データ生成期間
        self.Noise = self.Param["Task_Noise"]             #ノイズの有無
        self.NoiseScale = self.Param["Task_Noise_Scale"]   #ノイズのスケール

    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple: pass

    #データ生成
    def makeData(self): pass
    
#********************************************************************
#利用可能タスク

#Sin波タスク
class Task_SinCurve(Task):
    """
    sinカーブ予測タスク
    ルンゲクッタ法を使用
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.h = self.Param["Task_SinCurve_RK_h"]       #ルンゲクッタ法刻み幅
        self.Tau = param["Task_SinCurve_Tau"]                    #どれくらい先を予測するか

        self.makeData()

    #導関数
    def f(self, x : np.ndarray, t : np.ndarray) -> np.ndarray:
        return self.Amplitude * self.Frequency * np.cos(self.Frequency * t + self.Phase)

    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple: 
        return self.Y[t], self.Y[t + self.Tau]

    #データ生成
    def makeData(self):
        self.X = np.zeros([self.Length + self.Tau, self.D_u])
        self.Y = np.zeros([self.Length + self.Tau, self.D_y])

        #位相と周波数，振幅をランダムに決定
        #[-π,π]
        self.Phase =  0 #np.random.rand(self.D_u) * 2 * np.pi - np.pi
        #[1,5]
        self.Frequency =  2 #np.random.rand(self.D_u)
        #[0.5,1.5]
        self.Amplitude = 1 #np.random.rand(self.D_u) * 1 + 0.5

        #初期値（積分定数C）
        x = self.Amplitude * np.sin(self.Phase)
        #ルンゲクッタ法
        for ts in range(self.Length + self.Tau):
            self.X[ts] = x
            t = np.ones([self.D_u]) * ts * self.h
            k1 = self.h * self.f(x, t)
            k2 = self.h * self.f(x + 0.5 * k1, t + 0.5 * self.h)
            k3 = self.h * self.f(x + 0.5 * k2, t + 0.5 * self.h)
            k4 = self.h * self.f(x + k3, t + self.h)
            x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.Y[ts] = x
        
        self.Y = self.Y + 1
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#MCタスク
class Task_MC(Task):
    """
    MCタスク（正式名称ではない？）
    N次元ランダム時系列記憶タスク（正式名称ではない？）
    Tau後に入力されたものを出力する課題
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.Tau = self.Param["Task_MC_Tau"]               #遅延量，MCのτ
        
        self.makeData()
        
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        if t - self.Tau >= 0:
            return self.X[t], self.X[t - self.Tau]
        else:
            return self.X[t], np.zeros([self.D_y])
    
    #時刻tの入出力データ取得（MC用）
    def getDataTau(self, t: int, tau: int) -> tuple:
        if t - tau >= 0:
            return self.X[t], self.X[t - tau]
        else:
            return self.X[t], np.zeros([self.D_y])

    #データ生成
    def makeData(self):

        np.random.seed(seed=32)
        self.X = np.random.rand(self.Length, self.D_u) * 2 - 1
    
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#レスラー方程式タスク
class Task_NormalRosslor(Task):
    """
    レスラー方程式
    1次元のみ!!!!
    事前に入力信号作成プログラム完成
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.Scale = param["Task_NormalRosslor_Scale"]                #信号のスケール    
        self.Dt = param["Task_NormalRosslor_Dt"]                      #時間スケール
        self.Tau = param["Task_NormalRosslor_Tau"]                    #どれくらい先を予測するか
        self.InitTerm = param["Task_NormalRosslor_InitTerm"]          #初期状態排除期間

        self.a = param["Task_NormalRosslor_a"]                  #レスラー方程式パラメータ
        self.b = param["Task_NormalRosslor_b"]                  #レスラー方程式パラメータ
        self.c = param["Task_NormalRosslor_c"]                  #レスラー方程式パラメータ（μ）

        self.loadData()
        
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        if self.Noise == True:
            #入力信号にノイズを付与、教師信号はノイズなし
            return self.X_Noise[t], self.X[t + self.Tau]
        else:
            return self.X[t], self.X[t + self.Tau]
    
    #レスラー方程式の関数
    def Rosslor(self, old_x, old_y, old_z):
        x = - (old_y + old_z)
        y = old_x + self.a * old_y
        z = self.b + old_z * (old_x - self.c)

        return x, y, z
    
    #データを読み込む or 生成する
    def loadData(self):
        os.makedirs("./EMChialvo_Reservoir/Input_Data/Rosslor", exist_ok=True)  # ディレクトリがなければ作成
        if os.path.exists("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_X.npy"):
            #print("データをロードしています...")
            self.X = np.load("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_X.npy")
            self.Y = np.load("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_Y.npy")
            self.Z = np.load("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_Z.npy")
            self.X_Noise = np.load("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_X_Noise.npy")
            self.Y_Noise = np.load("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_Y_Noise.npy")
            self.Z_Noise = np.load("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_Z_Noise.npy")
        else:
            #print("データが見つかりません。新しく生成します...")
            self.makeData()

    #データ生成
    def makeData(self):
        self.X = np.zeros([self.Length + self.Tau, 1])
        self.Y = np.zeros([self.Length + self.Tau, 1])
        self.Z = np.zeros([self.Length + self.Tau, 1])

        np.random.seed(seed=99)
        
        s = (np.random.rand(1, 3) - 0.5) * 10
        
        for t in range(self.InitTerm + self.Length + self.Tau):

            if self.InitTerm <= t:
                #self.X[t - self.InitTerm] = s.reshape([-1])[:self.D_u] * self.Scale
                self.X[t - self.InitTerm] = s[0][0] * self.Scale        #1次元の信号に限り、列要素を変えることで、成分を変更できる。
                self.Y[t - self.InitTerm] = s[0][1] * self.Scale
                self.Z[t - self.InitTerm] = s[0][2] * self.Scale

            s_old = s
            s = np.zeros([1, 3])

            x, y, z = s_old[0,:]

            k1 = np.array(self.Rosslor(x, y, z))
            k2 = np.array(self.Rosslor(x + 0.5 * self.Dt * k1[0], y + 0.5 * self.Dt * k1[1], z + 0.5 * self.Dt * k1[2]))
            k3 = np.array(self.Rosslor(x + 0.5 * self.Dt * k2[0], y + 0.5 * self.Dt * k2[1], z + 0.5 * self.Dt * k2[2]))
            k4 = np.array(self.Rosslor(x + self.Dt * k3[0], y + self.Dt * k3[1], z + self.Dt * k3[2]))

            s[0, :] = [x + (self.Dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                       y + (self.Dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
                       z + (self.Dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])]
        
        np.save("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_X.npy", self.X)
        np.save("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_Y.npy", self.Y)
        np.save("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_Z.npy", self.Z)
        
        #ノイズ付与
        self.X_Noise = self.X + np.random.normal(0, self.NoiseScale, self.X.shape)
        self.Y_Noise = self.Y + np.random.normal(0, self.NoiseScale, self.Y.shape)
        self.Z_Noise = self.Z + np.random.normal(0, self.NoiseScale, self.Z.shape)

        np.save("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_X_Noise.npy", self.X_Noise)
        np.save("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_Y_Noise.npy", self.Y_Noise)
        np.save("./EMChialvo_Reservoir/Input_Data/Rosslor/NormalRosslor_data_Z_Noise.npy", self.Z_Noise)

#--------------------------------------------------------------------
#ローレンツ方程式タスク
class Task_NormalLorenz(Task):
    """
    ローレンツ方程式
    1次元のみ!!!!
    """
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        self.Scale = param["Task_NormalLorenz_Scale"]
        self.Dt = param["Task_NormalLorenz_Dt"]
        self.Tau = param["Task_NormalLorenz_Tau"]
        self.InitTerm = param["Task_NormalLorenz_InitTerm"]

        self.sigma = param["Task_NormalLorenz_Sigma"]
        self.beta = param["Task_NormalLorenz_Beta"]
        self.rho = param["Task_NormalLorenz_Rho"]

        self.loadData()
        
    def getData(self, t: int) -> tuple:
        if self.Noise == True:
            #入力信号にノイズを付与、教師信号はノイズなし
            return self.X_Noise[t], self.X[t + self.Tau]
        else:
            return self.X[t], self.X[t + self.Tau]
    
    def Lorenz(self, x, y, z):
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return dx, dy, dz
    
    #データを読み込む or 生成する
    def loadData(self):
        os.makedirs("./EMChialvo_Reservoir/Input_Data/Lorenz", exist_ok=True)  # ディレクトリがなければ作成
        if os.path.exists("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_X.npy"):
            #print("データをロードしています...")
            self.X = np.load("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_X.npy")
            self.Y = np.load("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_Y.npy")
            self.Z = np.load("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_Z.npy")
            self.X_Noise = np.load("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_X_Noise.npy")
            self.Y_Noise = np.load("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_Y_Noise.npy")
            self.Z_Noise = np.load("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_Z_Noise.npy")
        else:
            #print("データが見つかりません。新しく生成します...")
            self.makeData()

    def makeData(self):
        self.X = np.zeros([self.Length + self.Tau, 1])
        self.Y = np.zeros([self.Length + self.Tau, 1])
        self.Z = np.zeros([self.Length + self.Tau, 1])

        np.random.seed(seed=99)
        
        s = (np.random.rand(1, 3) - 0.5) * 10
        
        for t in range(self.InitTerm + self.Length + self.Tau):
            if self.InitTerm <= t:
                self.X[t - self.InitTerm] = s[0][0] * self.Scale
                self.Y[t - self.InitTerm] = s[0][1] * self.Scale
                self.Z[t - self.InitTerm] = s[0][2] * self.Scale
            
            s_old = s.copy()
            s = np.zeros([1, 3])
            
            x, y, z = s_old[0,:]
            
            k1 = np.array(self.Lorenz(x, y, z))
            k2 = np.array(self.Lorenz(x + 0.5 * self.Dt * k1[0], y + 0.5 * self.Dt * k1[1], z + 0.5 * self.Dt * k1[2]))
            k3 = np.array(self.Lorenz(x + 0.5 * self.Dt * k2[0], y + 0.5 * self.Dt * k2[1], z + 0.5 * self.Dt * k2[2]))
            k4 = np.array(self.Lorenz(x + self.Dt * k3[0], y + self.Dt * k3[1], z + self.Dt * k3[2]))

            s[0, :] = [x + (self.Dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                       y + (self.Dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
                       z + (self.Dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])]
            
        np.save("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_X.npy", self.X)
        np.save("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_Y.npy", self.Y)
        np.save("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_Z.npy", self.Z)

        #ノイズ付与
        self.X_Noise = self.X + np.random.normal(0, self.NoiseScale, self.X.shape)
        self.Y_Noise = self.Y + np.random.normal(0, self.NoiseScale, self.Y.shape)
        self.Z_Noise = self.Z + np.random.normal(0, self.NoiseScale, self.Z.shape)
        
        np.save("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_X_Noise.npy", self.X_Noise)
        np.save("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_Y_Noise.npy", self.Y_Noise)
        np.save("./EMChialvo_Reservoir/Input_Data/Lorenz/NormalLorenz_data_Z_Noise.npy", self.Z_Noise)

#--------------------------------------------------------------------
#ローレンツ96タスク
class Task_Lorenz96(Task):
    """
    ローレンツ96
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.Scale = param["Task_Lorenz96_Scale"]                 #信号のスケール
        self.Dt = param["Task_Lorenz96_Dt"]                       #時間スケール
        self.Tau = param["Task_Lorenz96_Tau"]                     #どれくらい先を予測するか
        self.InitTerm = param["Task_Lorenz96_InitTerm"]          #初期状態排除期間

        self.N = param["Task_Lorenz96_N"]                       #次元数
        self.F = param["Task_Lorenz96_F"]                       #外部強制力

        self.loadData()
    
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        if self.Noise == True:
            #入力信号にノイズを付与、教師信号はノイズなし
            return self.X_Noise[t], self.X[t + self.Tau]
        else:
            return self.X[t], self.X[t + self.Tau]
    
    def Lorenz96(self, x):
        dx = np.zeros(self.N)
        for i in range(self.N):
            dx[i] = (
                (x[(i + 1) % self.N] - x[(i - 2) % self.N])
                * x[(i - 1) % self.N]
                - x[i]
                + self.F
            )
        return dx
    
    def rk4_step(self, x, dt):
        k1 = self.Lorenz96(x)
        k2 = self.Lorenz96(x + dt * k1 / 2)
        k3 = self.Lorenz96(x + dt * k2 / 2)
        k4 = self.Lorenz96(x + dt * k3)
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    def loadData(self):
        os.makedirs("./EMChialvo_Reservoir/Input_Data/Lorenz96", exist_ok=True)
        if os.path.exists("./EMChialvo_Reservoir/Input_Data/Lorenz96/Lorenz96_data_X.npy"):
            self.X = np.load("./EMChialvo_Reservoir/Input_Data/Lorenz96/Lorenz96_data_X.npy")
            self.X_Noise = np.load("./EMChialvo_Reservoir/Input_Data/Lorenz96/Lorenz96_data_X_Noise.npy")
        else:
            self.makeData()

    #データ生成
    def makeData(self):
        np.random.seed(seed=999)
        total_steps = self.InitTerm + self.Length + self.Tau
        self.X = np.zeros((self.Length + self.Tau, self.D_u))
        self.s = (np.random.rand(self.N) - 0.5) * 10

        for i in range(total_steps):
            if i >= self.InitTerm:
                self.X[i - self.InitTerm] = self.s[:self.D_u] * self.Scale
            self.s = self.rk4_step(self.s, self.Dt)

        # ノイズを含むデータも保存
        if self.Noise:
            noise = np.random.normal(scale=self.NoiseScale, size=self.X.shape)
            self.X_Noise = self.X + noise
        else:
            self.X_Noise = self.X.copy()

        np.save("./EMChialvo_Reservoir/Input_Data/Lorenz96/Lorenz96_data_X.npy", self.X)
        np.save("./EMChialvo_Reservoir/Input_Data/Lorenz96/Lorenz96_data_X_Noise.npy", self.X_Noise)


#--------------------------------------------------------------------
#--------------------------------------------------------------------
class Task_Zeros(Task):
    """
    入力0
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        
        self.makeData()
    
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        return self.X[t], self.X[t + 1]
    
    #データ生成
    def makeData(self):
        self.X = np.zeros([self.Length + 1, self.D_u])

#--------------------------------------------------------------------
#--------------------------------------------------------------------
class Task_MackeyGlass_DDE(Task):
    """
    DDEマッキー・グラス方程式
    ただし、全く最適化されていないため、非常に重い
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        self.Dt = param["Task_MackeyGlassDDE_Dt"]                       #時間スケール
        self.PredictTau = param["Task_PredictDDE_Tau"]                     #どれくらい先を予測するか

        self.Beta = param["Task_MackeyGlassDDE_Beta"]                      #βの役割
        self.Gamma = param["Task_MackeyGlassDDE_Gamma"]                    #γの役割
        self.N = param["Task_MackeyGlassDDE_N"]                      #乗数
        self.MackeyTau = param["Task_MackeyGlassDDE_Tau"]                     #どれくらい先を予測するか
        self.InitTerm = param["Task_MackeyGlassDDE_InitTerm"]          #初期状態排除期間
        
        self.loadData()
    
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        if self.Noise == True:
            #入力信号にノイズを付与、教師信号はノイズなし
            return self.Y_Noise[t], self.Y[t + self.PredictTau]
        else:
            return self.Y[t], self.Y[t + self.PredictTau]
    
    # データを読み込む or 生成する
    def loadData(self):
        os.makedirs("./EMChialvo_Reservoir/Input_Data/MackeyGlass", exist_ok=True)  # ディレクトリがなければ作成
        if os.path.exists("./EMChialvo_Reservoir/Input_Data/MackeyGlass/mackey_glass_data.npy"):
            #print("データをロードしています...")
            self.Y = np.load("./EMChialvo_Reservoir/Input_Data/MackeyGlass/mackey_glass_data.npy")
            self.Y_Noise = np.load("./EMChialvo_Reservoir/Input_Data/MackeyGlass/mackey_glass_data_Noise.npy")
        else:
            #print("データが見つかりません。新しく生成します...")
            self.makeData()

    #データ生成
    def makeData(self):
        t_max = (self.InitTerm + self.Length + self.PredictTau)
        
        # 時間配列
        times = np.linspace(0, t_max * self.Dt, t_max)  # dt = 0.001, ステップ数 20000

        def MackeyGlass(X, t, tau):
            x_tau = X(t - tau) if t - tau >= 0 else 0
            # Mackey-Glass方程式の定義
            return (self.Beta * x_tau) / (1 + pow(x_tau, self.N)) - self.Gamma * X(t)
    
        def DLE(t):
            return 0.1
        
        solution = ddeint.ddeint(MackeyGlass, DLE, times, fargs=(self.MackeyTau,))
    
        self.Y = np.zeros((self.Length + self.PredictTau, self.D_u))
        
        for i in range(self.InitTerm + self.Length + self.PredictTau):
            if self.InitTerm <= i:
                self.Y[i - self.InitTerm] = solution[i]

        #データを外部ファイルに保存
        np.save("./EMChialvo_Reservoir/Input_Data/MackeyGlass/mackey_glass_data.npy", self.Y)
        self.Y_Noise = self.Y + np.random.normal(0, self.NoiseScale, self.Y.shape)
        np.save("./EMChialvo_Reservoir/Input_Data/MackeyGlass/mackey_glass_data_Noise.npy", self.Y_Noise)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
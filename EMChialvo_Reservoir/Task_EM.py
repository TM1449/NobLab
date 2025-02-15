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

    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple: pass

    #データ生成
    def makeData(self): pass
    
#********************************************************************
#利用可能タスク
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
        
        self.makeData()

    #導関数
    def f(self, x : np.ndarray, t : np.ndarray) -> np.ndarray:
        return self.Amplitude * self.Frequency * np.cos(self.Frequency * t + self.Phase)

    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple: 
        return self.X[t], self.Y[t]

    #データ生成
    def makeData(self):
        self.X = np.zeros([self.Length, self.D_u])
        self.Y = np.zeros([self.Length, self.D_y])

        #位相と周波数，振幅をランダムに決定
        #[-π,π]
        self.Phase = np.random.rand(self.D_u) * 2 * np.pi - np.pi
        #[1,5]
        self.Frequency = np.random.rand(self.D_u) * 4 + 1
        #[0.5,1.5]
        self.Amplitude = np.random.rand(self.D_u) * 1 + 0.5

        #初期値（積分定数C）
        x = self.Amplitude * np.sin(self.Phase)
        #ルンゲクッタ法
        for ts in range(self.Length):
            self.X[ts] = x
            t = np.ones([self.D_u]) * ts * self.h
            k1 = self.h * self.f(x, t)
            k2 = self.h * self.f(x + 0.5 * k1, t + 0.5 * self.h)
            k3 = self.h * self.f(x + 0.5 * k2, t + 0.5 * self.h)
            k4 = self.h * self.f(x + k3, t + self.h)
            x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.Y[ts] = x
        
#--------------------------------------------------------------------
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
class Task_Parity(Task):
    """
    パリティタスク（正式名称ではない？）
    Tau後に入力された多次元二値の偶奇（二値，1次元）を出力する課題
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.Tau = self.Param["Task_Parity_Tau"]                #遅延量
        
        self.MinTerm = param["Task_Parity_MinTerm"]             #同じ状態を維持する最小期間
        self.MaxTerm = param["Task_Parity_MaxTerm"]             #同じ状態を維持する最大期間

        self.makeData()
        
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        if t - self.Tau >= 0:
            return self.X[t], self.Y[t - self.Tau]
        else:
            return self.X[t], np.zeros([self.D_y])
    
    #時刻tの入出力データ取得（MC用，使わないかも）
    def getDataTau(self, t: int, tau: int) -> tuple:
        if t - tau >= 0:
            return self.X[t], self.Y[t - tau]
        else:
            return self.X[t], np.zeros([self.D_y])

    #データ生成
    def makeData(self):
        self.X = np.zeros([self.Length, self.D_u])
        self.Y = np.zeros([self.Length, self.D_y])

        x = np.random.randint(0, 2, [self.D_u])
        term = np.random.randint(self.MinTerm, self.MaxTerm + 1, [self.D_u])
        for t in range(self.Length):
            self.X[t] = x
            self.Y[t] = np.sum(x) % 2

            term -= 1
            for i in range(self.D_u):
                if term[i] <= 0:
                    x[i] = 1 if x[i] <= 0 else 0
                    term[i] = np.random.randint(self.MinTerm, self.MaxTerm + 1, [1])[0]
        
#--------------------------------------------------------------------
class Task_NDRosslor(Task):
    """
    N次元レスラー結合系時系列予測タスク
    複数のレスラー方程式のyをギャップジャンクションでリング状に結合
    ＠ノイズ入り入力信号を未実装
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.Scale = param["Task_Rosslor_Scale"]                #信号のスケール
        self.Mu = param["Task_Rosslor_Mu"]                      #レスラー方程式パラメータ
        self.A = param["Task_Rosslor_A"]                        #ギャップジャンクションパラメータ
        self.Dt = param["Task_Rosslor_Dt"]                      #時間スケール
        self.Tau = param["Task_Rosslor_Tau"]                    #どれくらい先を予測するか
        self.InitTerm = param["Task_Rosslor_InitTerm"]          #初期状態排除期間

        self.Systems = self.D_u // 3 + (0 if self.D_u % 3 == 0 else 1)#レスラー系の数

        self.makeData()
        
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        return self.X[t], self.X[t + self.Tau]
    
    #データ生成
    def makeData(self):
        self.X = np.zeros([self.Length + self.Tau, self.D_u])

        np.random.seed(seed=99)
        
        s = (np.random.rand(self.Systems, 3) - 0.5) * 10
        
        for t in range(self.InitTerm + self.Length + self.Tau):
            if self.InitTerm <= t:
                #self.X[t - self.InitTerm] = s.reshape([-1])[:self.D_u] * self.Scale
                self.X[t - self.InitTerm] = s[0][2] * self.Scale        #1次元の信号に限り、列要素を変えることで、成分を変更できる。

            s_old = s
            s = np.zeros([self.Systems, 3])

            for s_i in range(self.Systems):
                x = s_old[s_i][0]
                y = s_old[s_i][1]
                z = s_old[s_i][2]

                prev_y = s_old[s_i - 1][1] if s_i != 0 else s_old[self.Systems - 1][1]
                next_y = s_old[s_i + 1][1] if s_i != self.Systems - 1 else s_old[0][1]

                s[s_i][0] = x + (-(y + z)) * self.Dt
                s[s_i][1] = y + (x + 0.2 * y + self.A * (prev_y + next_y - 2 * y)) * self.Dt
                s[s_i][2] = z + (0.2 + z * (x - self.Mu)) * self.Dt
            
#--------------------------------------------------------------------
class Task_NDLorenz(Task):
    """
    N次元ローレンツ結合系時系列予測タスク
    複数のローレンツ方程式のyをギャップジャンクションでリング状に結合
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.Scale = param["Task_Lorenz_Scale"]                 #信号のスケール
        self.Sigma = param["Task_Lorenz_Sigma"]                 #ローレンツ方程式パラメータ
        self.Gamma = param["Task_Lorenz_Gamma"]                 #ローレンツ方程式パラメータ
        self.Const_B = param["Task_Lorenz_Const_B"]             #ローレンツ方程式パラメータ
        self.Dt = param["Task_Lorenz_Dt"]                       #時間スケール
        self.A = param["Task_Lorenz_A"]                         #ギャップジャンクションパラメータ
        self.Tau = param["Task_Lorenz_Tau"]                     #どれくらい先を予測するか
        self.InitTerm = param["Task_Lorenz_InitTerm"]          #初期状態排除期間

        self.Systems = self.D_u // 3 + (0 if self.D_u % 3 == 0 else 1)#ローレンツ系の数

        self.makeData()
    
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        return self.X[t], self.X[t + self.Tau]
    
    #データ生成
    def makeData(self):
        self.X = np.zeros([self.Length + self.Tau, self.D_u])
        self.Z = np.zeros([self.Length + self.Tau, self.D_u])

        np.random.seed(seed=999)

        s = (np.random.rand(self.Systems, 3) - 0.5) * 10
        
        for t in range(self.InitTerm + self.Length + self.Tau):
            if self.InitTerm <= t:
                #self.X[t - self.InitTerm] = s.reshape([-1])[:self.D_u] * self.Scale
                self.X[t - self.InitTerm] = s[0][0] * self.Scale        #1次元の信号に限り、列要素を変えることで、成分を変更できる。
                #self.Z[t - self.InitTerm] = s[0][2] * self.Scale

            
            s_old = s
            s = np.zeros([self.Systems, 3])

            for s_i in range(self.Systems):
                x = s_old[s_i][0]
                y = s_old[s_i][1]
                z = s_old[s_i][2]

                prev_y = s_old[s_i - 1][1] if s_i != 0 else s_old[self.Systems - 1][1]
                next_y = s_old[s_i + 1][1] if s_i != self.Systems - 1 else s_old[0][1]

                s[s_i][0] = x + (-self.Sigma * (x - y)) * self.Dt
                s[s_i][1] = y + (-x * z + self.Gamma * x - (y)) * self.Dt
                s[s_i][2] = z + (x * y - self.Const_B * z) * self.Dt

class Task_tcVDP(Task):
    """
    van der Pol振動子の結合タスク
    入力4の出力4でのみ動く
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.Mu = param["Task_vandelPol_Mu"]                        #振動の係数
        self.C = param["Task_vandelPol_c"]                          #結合係数
        self.TimeScale = param["Task_vandelPol_TimeScale"]          #信号の時間スケール
        self.Init = param["Task_vandelPol_Init"]                    #初期状態
        
        self.Dt = param["Task_vandelPol_Dt"]                        #時間スケール
        self.Tau = param["Task_vandelPol_Tau"]                      #どれくらい先を予測するか
        self.InitTerm = param["Task_vandelPol_InitTerm"]            #初期状態排除期間

        self.makeData()
        
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        return self.X[t], self.Y[t + self.Tau]
    
    #データ生成
    def makeData(self):
        self.X = np.zeros([self.Length + self.Tau, self.D_u])
        self.Y = np.zeros([self.Length + self.Tau, self.D_y])
        # データの軌跡を保存する配列
        x1s = np.empty(self.Length + self.Tau)
        y1s = np.empty(self.Length + self.Tau)
        x2s = np.empty(self.Length + self.Tau)
        y2s = np.empty(self.Length + self.Tau)

        x1s[0], y1s[0], x2s[0], y2s[0] = self.Init

        for i in range (self.Length + self.Tau - 1):
            x1s[i + 1] = x1s[i] +  ( ((y1s[i] + self.C[0] * x2s[i]) / self.TimeScale[0]) * self.Dt )
            y1s[i + 1] = y1s[i] +  ( ((self.Mu[0] * (1 - x1s[i] ** 2) * y1s[i] - x1s[i]) / self.TimeScale[0]) * self.Dt )
            x2s[i + 1] = x2s[i] +  ( ((y2s[i] + self.C[1] * x1s[i]) / self.TimeScale[1]) * self.Dt )
            y2s[i + 1] = y2s[i] +  ( ((self.Mu[1] * (1 - x2s[i] ** 2) * y2s[i] - x2s[i]) / self.TimeScale[1]) * self.Dt )
        
        x1s = x1s.reshape([len(x1s),-1])
        y1s = y1s.reshape([len(y1s),-1])
        x2s = x2s.reshape([len(x2s),-1])
        y2s = y2s.reshape([len(y2s),-1])
        
        self.X = np.concatenate([x1s, y1s, x2s, y2s],1)
        self.Y = np.concatenate([x1s, y1s, x2s, y2s],1)

class Task_LogisticEquation(Task):
    """
    ロジスティック写像
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.A = self.Param["Task_LogisticEquation_A"]       #ルンゲクッタ法刻み幅
        self.Tau = param["Task_LogisticEquation_Tau"]        #どれくらい先を予測するか
        
        self.makeData()

    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple: 
        return self.X[t], self.X[t + self.Tau]

    #データ生成
    def makeData(self):
        
        #ロジスティック写像の配列
        self.X = np.zeros([self.Length + self.Tau, self.D_u])
        
        #シード値設定（無効にしても可）
        np.random.seed(seed=999)
        
        #初期値生成
        self.X[0] = np.random.rand(1)

        for ts in range(self.Length + self.Tau - 1):
            self.X[ts + 1] = self.Logistic(self.X[ts])

    def Logistic(self, old_x):
        #ロジスティック写像の更新式
        self.next_x = self.A * old_x * (1 - old_x)

        return self.next_x
    

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

        self.makeData()
    
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        return self.X[t], self.X[t + self.Tau]
    
    def Lorenz96(self, x):
        dx = np.zeros(self.N)
        for i in range((self.N)):
            dx[i] =  x[i] +  (((x[(i + 1) % self.N] - x[(i - 2)]) * x[i - 1]) - x[i] + self.F) * self.Dt

        return dx

    #データ生成
    def makeData(self):
        # 乱数シードの設定
        np.random.seed(seed=999)
        self.X = np.zeros((self.Length + self.Tau, self.D_u))

        self.s = (np.random.rand(self.N) - 0.5) * 10

        for i in range(self.InitTerm + self.Length + self.Tau):
            if self.InitTerm <= i:
                self.X[i - self.InitTerm] = self.s[-1] * self.Scale

            self.s = self.Lorenz96(self.s)


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


class Task_MackeyGlass(Task):
    """
    マッキー・グラス方程式
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        self.Scale = param["Task_MackeyGlass_Scale"]                 #信号のスケール
        
        self.PredictTau = param["Task_Predict_Tau"]                     #どれくらい先を予測するか
        self.MackeyTau = param["Task_MackeyGlass_Tau"]                     #どれくらい先を予測するか
        self.InitTerm = param["Task_MackeyGlass_InitTerm"]          #初期状態排除期間

        self.makeData()
    
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        return self.X[t], self.X[t + self.PredictTau]
    
    #データ生成
    def makeData(self):

        # 乱数シードの設定
        np.random.seed(seed=999)
        
        N = self.InitTerm + self.Length + self.PredictTau
        self.XmI = signalz.mackey_glass(N, 0.2, 0.8, 0.9, 23, 10, 0.1)
        self.Xm = np.delete(self.XmI, np.s_[:self.InitTerm])
        self.X = np.expand_dims(self.Xm, 1)


class Task_MackeyGlass_DDE(Task):
    """
    DDEマッキー・グラス方程式
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        self.Scale = param["Task_MackeyGlassDDE_Scale"]                 #信号のスケール
        self.Dt = param["Task_MackeyGlassDDE_Dt"]                       #時間スケール
        self.PredictTau = param["Task_PredictDDE_Tau"]                     #どれくらい先を予測するか

        self.MackeyA = param["Task_MackeyGlassDDE_A"]                      #γの役割
        self.MackeyB = param["Task_MackeyGlassDDE_B"]                      #βの役割
        self.MackeyN = param["Task_MackeyGlassDDE_N"]                      #乗数
        self.MackeyTau = param["Task_MackeyGlassDDE_Tau"]                     #どれくらい先を予測するか
        self.InitTerm = param["Task_MackeyGlassDDE_InitTerm"]          #初期状態排除期間

        self.makeData()
    
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        return self.Y[t], self.Y[t + self.PredictTau]
    
    #データ生成
    def makeData(self):

        np.random.seed(seed=999)
        self.X = np.zeros([self.MackeyTau + self.Length + self.PredictTau, self.D_u])
        self.Y = np.zeros([self.Length + self.PredictTau, self.D_u])
        self.X[0] = np.random.rand(1) 
        
        for t in range(self.MackeyTau, self.InitTerm + self.Length + self.PredictTau - 1):
            if self.InitTerm <= t:
                self.Y[t - self.InitTerm] = self.X[t] * self.Scale

            self.X[t + 1] = ((self.MackeyB * self.X[t - self.MackeyTau]) / (1 + pow(self.X[t - self.MackeyTau], self.MackeyN)) - self.MackeyA * self.X[t]) * self.Dt + self.X[t]
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

        np.random.seed(seed=728)
        
        s = (np.random.rand(self.Systems, 3) - 0.5) * 10
        for t in range(self.InitTerm + self.Length + self.Tau):
            if self.InitTerm <= t:
                self.X[t - self.InitTerm] = s.reshape([-1])[:self.D_u] * self.Scale
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
        self.InitTerm = param["Task_Rosslor_InitTerm"]          #初期状態排除期間

        self.Systems = self.D_u // 3 + (0 if self.D_u % 3 == 0 else 1)#ローレンツ系の数

        self.makeData()
    
    #時刻tの入出力データ取得
    def getData(self, t: int) -> tuple:
        return self.X[t], self.X[t + self.Tau]
    
    #データ生成
    def makeData(self):
        self.X = np.zeros([self.Length + self.Tau, self.D_u])

        np.random.seed(seed=728)

        s = (np.random.rand(self.Systems, 3) - 0.5) * 10
        
        for t in range(self.InitTerm + self.Length + self.Tau):
            if self.InitTerm <= t:
                self.X[t - self.InitTerm] = s.reshape([-1])[:self.D_u] * self.Scale
            
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


#####################################################################
#ESNプロジェクト，Chialvoニューロンマップのグラフ描写用
#制作者：田中勝規
#作成日：2024/09/24
#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・
#・
#・
#====================================================================


#====================================================================
#外部ライブラリ
import numpy as np
import random

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#内部プログラム



#====================================================================
class Model:
    """
    モデル

    このクラスで，どんなモジュール（どんな複雑な構造）を使っても，
    評価のクラスでは同一のインタフェイスになるようにする．
    つまり，モデルの違いはここで全て吸収する．
    これにより，評価はモデルごとに作成しなくて良くなる（はず．）
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        self.Param = param
        
        self.Parent = parent                            #親オブジェクト
    
    def __call__(self) -> dict: pass


class Model_Chialvo(Model):

    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #従来Chialvo変数
        self.a = self.Param["Chialvo_a"]
        self.b = self.Param["Chialvo_b"]
        self.c = self.Param["Chialvo_c"]
        self.k0 = self.Param["Chialvo_k0"]

        #電磁束下におけるChialvo変数
        self.k = self.Param["Chialvo_k"]
        self.k1 = self.Param["Chialvo_k1"]
        self.k2 = self.Param["Chialvo_k2"]
        self.alpha = self.Param["Chialvo_alpha"]
        self.beta = self.Param["Chialvo_beta"]

        #入力信号
        self.Input_Signal_Amp = self.Param["Input_Signal_Amp"]
        self.Input_Signal_Def = self.Param["Input_Signal_Def"]

        #空走時間
        self.Length_Burnin = self.Param["Length_Burnin"]
        #評価時間
        self.Length_Eva = self.Param["Length_Eva"]
        #プロット時間
        self.Length_Plot = self.Param["Length_Plot"]
        
        #総合時間
        self.Length_Total = self.Length_Burnin + self.Length_Eva +self.Length_Plot

        #--------------------------------------------------------------------
        #初期値Xについて
        self.Initial_Value_X = self.Param["Initial_Value_X"]

        if self.Initial_Value_X == None:
            self.x = np.random.uniform(-0.1,0.1,self.Length_Total)
        else:
            self.x = np.zeros(self.Length_Total)
            self.x[0] = self.Initial_Value_X

        #--------------------------------------------------------------------
        #初期値Yについて
        self.Initial_Value_Y = self.Param["Initial_Value_Y"]

        if self.Initial_Value_Y == None:
            self.y = np.random.uniform(-0.1,0.1,self.Length_Total)
        else:
            self.y = np.zeros(self.Length_Total)
            self.y[0] = self.Initial_Value_Y
        
        #--------------------------------------------------------------------
        #初期値Phiについて
        self.Initial_Value_Phi = self.Param["Initial_Value_Phi"]

        if self.Initial_Value_Phi == None:
            self.phi = np.random.uniform(-0.1,0.1,self.Length_Total)
        else:
            self.phi = np.zeros(self.Length_Total)
            self.phi[0] = self.Initial_Value_Phi
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self):
        #入力信号プロットの作成
        self.Input_Signal = np.zeros(self.Length_Total)

        #恒等関数の場合
        if self.Input_Signal_Def == None:
            for n in range(self.Length_Burnin, self.Length_Total - 1):
                self.Input_Signal[n+1] = self.Input_Signal_Amp
        
        #sin波の場合
        elif self.Input_Signal_Def == np.sin:
            for n in range(self.Length_Burnin, self.Length_Total - 1):
                self.Input_Signal[n+1] = 0.1 * self.Input_Signal_Amp * self.Input_Signal_Def(4 * n * np.pi / 180)

        #Chialvoニューロンの差分方程式の計算部
        for i in range(self.Length_Total - 1):
            print("\r%d / %d"%(i, self.Length_Total), end = "")

            self.x[i+1] = pow(self.x[i], 2) * np.exp(self.y[i] - self.x[i]) + self.k0 \
                + self.k * self.x[i] * (self.alpha + 3 * self.beta * pow(self.phi[i], 2)) + self.Input_Signal[i]
            self.y[i+1] = self.a * self.y[i] - self.b * self.x[i] + self.c 
            self.phi[i+1] = self.k1 * self.x[i] - self.k2 * self.phi[i]

        return self.x[self.Length_Burnin :], \
            self.y[self.Length_Burnin :], \
            self.phi[self.Length_Burnin :], \
                self.Input_Signal[self.Length_Burnin :]

class Model_Chialvo_OldNullcline(Model):

    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #従来Chialvo変数
        self.a = self.Param["Chialvo_a"]
        self.b = self.Param["Chialvo_b"]
        self.c = self.Param["Chialvo_c"]
        self.k0 = self.Param["Chialvo_k0"]

        #実行時間
        self.RunTime = self.Param["RunTime"]
        self.Plot_Start = self.Param["Plot_Start"]
        self.Plot_End = self.Param["Plot_End"]

        #入力信号
        self.Input_Signal = self.Param["Input_Signal"]

        #ベクトル場の間隔
        self.Vdt = self.Param["Vdt"]
        
        #計算用の間隔
        self.dt = self.Param["dt"]

        #ベクトル場の点分布の範囲
        self.Plot_x_Start = self.Param["Plot_x_Start"]
        self.Plot_x_End = self.Param["Plot_x_End"]

        self.Plot_y_Start = self.Param["Plot_y_Start"]
        self.Plot_y_End = self.Param["Plot_y_End"]
        #====================================================================
        
    def __call__(self):
        #2D-ChialvoニューロンマップのNullclineとベクトル場の計算部
        print("Old_Nullcline")
        #--------------------------------------------------------------------
        """
        Vt, Vx, Vyはベクトル場の変数.
        """    
        self.Vdx = np.arange(self.Plot_x_Start, self.Plot_x_End, self.Vdt)
        self.Vdy = np.arange(self.Plot_y_Start, self.Plot_y_End, self.Vdt)

        self.X, self.Y = np.meshgrid(self.Vdx, self.Vdy)
        
        self.Vx = (pow(self.X, 2) * np.exp(self.Y - self.X) + self.k0) + self.Input_Signal - self.X
        self.Vy = (self.a * self.Y  - self.b * self.X + self.c) - self.Y

        #--------------------------------------------------------------------
        """
        dt, dx, dyはNullclineの変数.
        """    
        self.dx = np.arange(self.Plot_x_Start, self.Plot_x_End, self.dt)
        self.dy = np.arange(self.Plot_y_Start, self.Plot_y_End, self.dt)

        self.fx = np.log(self.dx - self.k0 - self.Input_Signal) - 2 * np.log(self.dx) + self.dx
        self.fy = (self.a * self.dy + self.c - self.dy) / self.b

        #--------------------------------------------------------------------
        """
        x, yは相平面の変数.
        """
        self.Initial_Value_X = self.Param["Initial_Value_X"]

        if self.Initial_Value_X == None:
            self.x = np.random.uniform(-1,1,self.RunTime)
        else:
            self.x = np.zeros(self.RunTime)
            self.x[0] = self.Initial_Value_X
        
        #--------------------------------------------------------------------
        #初期値Yについて
        self.Initial_Value_Y = self.Param["Initial_Value_Y"]

        if self.Initial_Value_Y == None:
            self.y = np.random.uniform(-1,1,self.RunTime)
        else:
            self.y = np.zeros(self.RunTime)
            self.y[0] = self.Initial_Value_Y
        #--------------------------------------------------------------------
        #入力信号プロットの作成
        self.Input_Signal_In = np.zeros(self.RunTime)

        for n in range(self.RunTime - 1):
            self.Input_Signal_In[n+1] = self.Input_Signal
        #--------------------------------------------------------------------
        for n in range(self.RunTime - 1):
            self.x[n+1] = pow(self.x[n], 2) * np.exp(self.y[n] - self.x[n]) + self.k0 + self.Input_Signal_In[n]
            self.y[n+1] = self.a * self.y[n] - self.b * self.x[n] + self.c

        return self.X, self.Y, self.Vx, self.Vy, \
                self.dx, self.dy, self.fx, self.fy, \
                        self.x[self.Plot_Start : self.Plot_End - 1], self.y[self.Plot_Start : self.Plot_End - 1]


class Model_Chialvo_NewNullcline(Model):

    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #従来Chialvo変数
        self.a = self.Param["Chialvo_a"]
        self.b = self.Param["Chialvo_b"]
        self.c = self.Param["Chialvo_c"]
        self.k0 = self.Param["Chialvo_k0"]

        #電磁束下におけるChialvo変数
        self.k = self.Param["Chialvo_k"]
        self.k1 = self.Param["Chialvo_k1"]
        self.k2 = self.Param["Chialvo_k2"]
        self.alpha = self.Param["Chialvo_alpha"]
        self.beta = self.Param["Chialvo_beta"]

        #実行時間
        self.RunTime = self.Param["RunTime"]
        self.Plot_Start = self.Param["Plot_Start"]
        self.Plot_End = self.Param["Plot_End"]
        
        #入力信号
        self.Input_Signal = self.Param["Input_Signal"]

        #ベクトル場の間隔
        self.Vdt = self.Param["Vdt"]
        
        #計算用の間隔
        self.dt = self.Param["dt"]

        #ベクトル場の点分布の範囲
        self.Plot_x_Start = self.Param["Plot_x_Start"]
        self.Plot_x_End = self.Param["Plot_x_End"]

        self.Plot_y_Start = self.Param["Plot_y_Start"]
        self.Plot_y_End = self.Param["Plot_y_End"]
        #====================================================================
        
    def __call__(self):
        #3D-ChialvoニューロンマップのNullclineとベクトル場の計算部
        print("New_Nullcline")
        #--------------------------------------------------------------------
        """
        Vt, Vx, Vyはベクトル場の変数.
        """    
        self.Vdx = np.arange(self.Plot_x_Start, self.Plot_x_End, self.Vdt)
        self.Vdy = np.arange(self.Plot_y_Start, self.Plot_y_End, self.Vdt)

        self.X, self.Y = np.meshgrid(self.Vdx, self.Vdy)
        
        self.Vx = (pow(self.X, 2) * np.exp(((self.b  - self.a + 1) * self.X - self.c) / (self.a - 1)) + self.k0 \
            + ((3 * self.k * self.beta * pow(self.k1, 2)) / pow((1 + self.k2), 2)) * pow(self.X, 3) \
                + self.X * self.k * self.alpha) + self.Input_Signal - self.X
        self.Vy = self.X - self.Y
        
        #--------------------------------------------------------------------
        """
        dt, dx, dyはNullclineの変数.
        """    
        self.dx = np.arange(self.Plot_x_Start, self.Plot_x_End, self.dt)
        self.dy = np.arange(self.Plot_y_Start, self.Plot_y_End, self.dt)

        self.fx = pow(self.dx, 2) * np.exp(((self.b  - self.a + 1) * self.dx - self.c) / (self.a - 1)) + self.k0 \
            + ((3 * self.k * self.beta * pow(self.k1, 2)) / pow((1 + self.k2), 2)) * pow(self.dx, 3) \
                + self.dx * self.k * self.alpha + self.Input_Signal
        self.fy = self.dx

        #--------------------------------------------------------------------
        """
        x, y, phiは相平面の変数.
        """
        self.Initial_Value_X = self.Param["Initial_Value_X"]

        if self.Initial_Value_X == None:
            self.x = np.random.uniform(-0.1,0.1,self.RunTime)
        else:
            self.x = np.zeros(self.RunTime)
            self.x[0] = self.Initial_Value_X
        
        #--------------------------------------------------------------------
        #初期値Yについて
        self.Initial_Value_Y = self.Param["Initial_Value_Y"]

        if self.Initial_Value_Y == None:
            self.y = np.random.uniform(-0.1,0.1,self.RunTime)
        else:
            self.y = np.zeros(self.RunTime)
            self.y[0] = self.Initial_Value_Y
        
        #--------------------------------------------------------------------
        #初期値Phiについて
        self.Initial_Value_Phi = self.Param["Initial_Value_Phi"]

        if self.Initial_Value_Phi == None:
            self.phi = np.random.uniform(-0.1,0.1,self.RunTime)
        else:
            self.phi = np.zeros(self.RunTime)
            self.phi[0] = self.Initial_Value_Phi
        
        #--------------------------------------------------------------------
        #入力信号プロットの作成
        self.Input_Signal_In = np.zeros(self.RunTime)

        for n in range(self.RunTime - 1):
            self.Input_Signal_In[n+1] = self.Input_Signal
        #--------------------------------------------------------------------
        for n in range(self.RunTime - 1):
            self.x[n+1] = pow(self.x[n], 2) * np.exp(self.y[n] - self.x[n]) + self.k0 \
                + self.k * self.x[n] * (self.alpha + 3 * self.beta * pow(self.phi[n], 2)) + self.Input_Signal_In[n]
            self.y[n+1] = self.a * self.y[n] - self.b * self.x[n] + self.c
            self.phi[n+1] = self.k1 * self.x[n] - self.k2 * self.phi[n]

        return self.X, self.Y, self.Vx, self.Vy, \
                self.dx, self.dy, self.fx, self.fy, \
                        self.x[self.Plot_Start : self.Plot_End - 1], self.y[self.Plot_Start : self.Plot_End - 1]


class Model_ChialvoNeuronMap(Model):

    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #従来Chialvo変数
        self.a = self.Param["Chialvo_a"]
        self.b = self.Param["Chialvo_b"]
        self.c = self.Param["Chialvo_c"]
        self.k0 = self.Param["Chialvo_k0"]

        #電磁束下におけるChialvo変数
        self.k = self.Param["Chialvo_k"]
        self.k1 = self.Param["Chialvo_k1"]
        self.k2 = self.Param["Chialvo_k2"]
        self.alpha = self.Param["Chialvo_alpha"]
        self.beta = self.Param["Chialvo_beta"]

        #電磁束下Chialvoニューロンネットワークのパラメータ
        self.N = self.Param["Chialvo_Neurons"]
        self.Mu = self.Param["Chialvo_Mu"]
        self.Sigma = self.Param["Chialvo_Sigma"]
        self.R = self.Param["Chialvo_R"]

        self.Random = self.Param["Chialvo_Neurons_Random"]
        self.Rho = self.Param["Chialvo_Rho"]

        #入力信号
        self.Input_Signal_Amp = self.Param["Input_Signal_Amp"]
        self.Input_Signal_Def = self.Param["Input_Signal_Def"]

        #空走時間
        self.Length_Burnin = self.Param["Length_Burnin"]
        #評価時間
        self.Length_Eva = self.Param["Length_Eva"]
        #プロット時間
        self.Length_Plot = self.Param["Length_Plot"]
        
        #総合時間
        self.Length_Total = self.Length_Burnin + self.Length_Eva +self.Length_Plot

        #--------------------------------------------------------------------
        #初期値Xについて
        self.x = np.zeros((self.Length_Total, self.N))
        self.x[0, :] = np.random.uniform(-1,1)

        #--------------------------------------------------------------------
        #初期値Yについて
        self.y = np.zeros((self.Length_Total, self.N))
        self.y[0, :] = np.random.uniform(-1,1)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #初期値Phiについて
        self.phi = np.zeros((self.Length_Total, self.N))
        self.phi[0, :] = np.random.uniform(-1,1)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self):
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #入力信号プロットの作成
        self.Input_Signal = np.zeros(self.Length_Total)

        #恒等関数の場合
        if self.Input_Signal_Def == None:
            for n in range(self.Length_Burnin, self.Length_Total - 1):
                self.Input_Signal[n+1] = self.Input_Signal_Amp
        
        #sin波の場合
        elif self.Input_Signal_Def == np.sin:
            for n in range(self.Length_Burnin, self.Length_Total - 1):
                self.Input_Signal[n+1] = 0.1 * self.Input_Signal_Amp * self.Input_Signal_Def(4 * n * np.pi / 180)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #Chialvoニューロンの差分方程式の計算部
        #--------------------------------------------------------------------
        if self.Random == True:
            self.LW = np.random.randn(self.N, self.N)
            self.sw, _ = np.linalg.eig(self.LW)

            self.Matrix = self.Rho * (self.LW / np.max(np.abs(self.sw)))

        else:
            #--------------------------------------------------------------------
            #self.ring_matrixは、リングネットワークの隣接行列を生成する
            self.ring_matrix = np.identity(self.N)
            self.ring_matrix[0,0] = 0

            for i in range(1,self.N):
                #これが対角成分周辺の1を生成する
                self.ring_matrix[i:i+self.R+1 , i:i+self.R+1] = 1

                #これらが範囲外でも対応させる部分（左下や右上の1を生成する）
                self.ring_matrix[self.N-self.R+i-1: , i] = 1
                self.ring_matrix[i , self.N-self.R+i-1:] = 1
            for i in range(1,self.N):
                self.ring_matrix[i,i] = -2 * self.R

            self.ring_matrix *= (self.Sigma / (2 * self.R))

            #--------------------------------------------------------------------
            #self.star_matrixは、スターネットワークの隣接行列を生成する
            self.star_matrix = np.zeros((self.N, self.N))

            self.star_matrix[0,0] = -self.Mu * (self.N-1)
            self.star_matrix[0,1:] = self.Mu
            self.star_matrix[1:,0] = -self.Mu

            for i in range(1,self.N):
                self.star_matrix[i,i] += self.Mu
            
            #--------------------------------------------------------------------
            #self.Ring_Star_matrixは、リングスターネットワークの隣接行列を生成する（ただ足し合わせた）
            self.Ring_Star_matrix = (self.ring_matrix + self.star_matrix)

            self.Matrix = self.Ring_Star_matrix

    #--------------------------------------------------------------------

        def M(phi):
            return self.alpha + 3 * self.beta * pow(phi, 2)

        for n in range(self.Length_Total - 1):

            print("\r%d / %d"%(n, self.Length_Total), end = "")

            self.x[n+1, :] = pow(self.x[n, :], 2) * np.exp(self.y[n, :] - self.x[n, :]) + self.k0 \
                + self.k * self.x[n, :] * M(self.phi[n, :]) + self.x[n,:] @ self.Matrix
            self.y[n+1, :] = self.a * self.y[n, :] - self.b * self.x[n, :] + self.c
            self.phi[n+1, :] = self.k1 * self.x[n, :] - self.k2 * self.phi[n, :]

        return self.x[self.Length_Burnin: ,:], \
            self.y[self.Length_Burnin: ,:], \
            self.phi[self.Length_Burnin: ,:], _

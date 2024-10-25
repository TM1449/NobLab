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
        self.Input_Signal = self.Param["Input_Signal"]

        #実行時間
        self.RunTime = self.Param["RunTime"]
        self.Plot_Start = self.Param["Plot_Start"]
        self.Plot_End = self.Param["Plot_End"]

        #--------------------------------------------------------------------
        #初期値Xについて
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
        #初期値Phiについて
        self.Initial_Value_Phi = self.Param["Initial_Value_Phi"]

        if self.Initial_Value_Phi == None:
            self.phi = np.random.uniform(-1,1,self.RunTime)
        else:
            self.phi = np.zeros(self.RunTime)
            self.phi[0] = self.Initial_Value_Phi
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self):
        #入力信号プロットの作成
        self.Input_Signal = np.zeros(self.RunTime)
        self.Input_Signal_X = np.zeros(self.RunTime)

        for j in range(400):
            self.Input_Signal[self.RunTime * j // 400 : self.RunTime * (j+1) // 400] = np.random.randint(0, 2)
        
        #Chialvoニューロンの差分方程式の計算部
        for i in range(self.RunTime - 1):

            print("\r%d / %d"%(i, self.RunTime), end = "")

            self.x[i+1] = pow(self.x[i], 2) * np.exp(self.y[i] - self.x[i]) + self.k0 \
                + self.k * self.x[i] * (self.alpha + 3 * self.beta * pow(self.phi[i], 2)) + self.Input_Signal[i]
            self.y[i+1] = self.a * self.y[i] - self.b * self.x[i] + self.c
            self.phi[i+1] = self.k1 * self.x[i] - self.k2 * self.phi[i]

            self.Input_Signal_X[i+1] = self.Input_Signal[i]

        return self.x[self.Plot_Start : self.Plot_End - 1], \
            self.y[self.Plot_Start : self.Plot_End - 1], \
            self.phi[self.Plot_Start : self.Plot_End - 1], \
                self.Input_Signal_X[self.Plot_Start : self.Plot_End - 1]


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

        for n in range(self.RunTime - 1):
            self.x[n+1] = pow(self.x[n], 2) * np.exp(self.y[n] - self.x[n]) + self.k0
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
        #初期値Phiについて
        self.Initial_Value_Phi = self.Param["Initial_Value_Phi"]

        if self.Initial_Value_Phi == None:
            self.phi = np.random.uniform(-1,1,self.RunTime)
        else:
            self.phi = np.zeros(self.RunTime)
            self.phi[0] = self.Initial_Value_Phi
        
        #--------------------------------------------------------------------
        for n in range(self.RunTime - 1):
            self.x[n+1] = pow(self.x[n], 2) * np.exp(self.y[n] - self.x[n]) + self.k0 \
                + self.k * self.x[n] * (self.alpha + 3 * self.beta * pow(self.phi[n], 2))
            self.y[n+1] = self.a * self.y[n] - self.b * self.x[n] + self.c
            self.phi[n+1] = self.k1 * self.x[n] - self.k2 * self.phi[n]

        return self.X, self.Y, self.Vx, self.Vy, \
                self.dx, self.dy, self.fx, self.fy, \
                        self.x[self.Plot_Start : self.Plot_End - 1], self.y[self.Plot_Start : self.Plot_End - 1]


class Model_Chialvo_NewNullcline3D(Model):

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

        #ベクトル場の間隔
        self.Vdt = self.Param["Vdt"]
        
        #計算用の間隔
        self.dt = self.Param["dt"]

        #ベクトル場の点分布の範囲
        self.Plot_x_Start = self.Param["Plot_x_Start"]
        self.Plot_x_End = self.Param["Plot_x_End"]

        self.Plot_y_Start = self.Param["Plot_y_Start"]
        self.Plot_y_End = self.Param["Plot_y_End"]

        self.Plot_phi_Start = self.Param["Plot_phi_Start"]
        self.Plot_phi_End = self.Param["Plot_phi_End"]
        #====================================================================
        
    def __call__(self):
        #3D-ChialvoニューロンマップのNullclineとベクトル場の計算部
        print("New_Nullcline_3D")
        #--------------------------------------------------------------------
        """
        Vt, Vx, Vy, Vphiはベクトル場の変数.
        """    
        self.Vdx = np.arange(self.Plot_x_Start, self.Plot_x_End, self.Vdt)
        self.Vdy = np.arange(self.Plot_y_Start, self.Plot_y_End, self.Vdt)
        self.Vdphi = np.arange(self.Plot_phi_Start, self.Plot_phi_End, self.Vdt)

        self.X, self.Y, self.Phi = np.meshgrid(self.Vdx, self.Vdy, self.Vdphi)
        
        self.Vx = (pow(self.X, 2) * np.exp(self.Y - self.X) + self.k0 + self.k * self.X * (self.alpha + 3 * self.beta * pow(self.Phi, 2))) - self.X
        self.Vy = (self.a * self.Y - self.b * self.X + self.c) - self.Y
        self.Vphi = (self.k1 * self.X - self.k2 * self.Phi) - self.Phi
        
        #--------------------------------------------------------------------
        """
        dt, dx, dy, dphiはNullclineの変数.
        """    
        self.dx = np.arange(self.Plot_x_Start, self.Plot_x_End, self.dt)
        self.dy = np.arange(self.Plot_y_Start, self.Plot_y_End, self.dt)
        self.dphi = np.arange(self.Plot_phi_Start, self.Plot_phi_End, self.dt)

        self.fx = (pow(self.dx, 2) * np.exp(self.dy - self.dx) + self.k0 + self.k * self.dx * (self.alpha + 3 * self.beta * pow(self.dphi, 2)))
        self.fy = (self.b * self.dx - self.c) / (self.a - 1)
        self.fphi = (self.k1 * self.dx) / (self.k2 + 1)

        #--------------------------------------------------------------------
        """
        x, y, phiは相平面の変数.
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
        #初期値Phiについて
        self.Initial_Value_Phi = self.Param["Initial_Value_Phi"]

        if self.Initial_Value_Phi == None:
            self.phi = np.random.uniform(-1,1,self.RunTime)
        else:
            self.phi = np.zeros(self.RunTime)
            self.phi[0] = self.Initial_Value_Phi
        
        #--------------------------------------------------------------------
        for n in range(self.RunTime - 1):
            self.x[n+1] = pow(self.x[n], 2) * np.exp(self.y[n] - self.x[n]) + self.k0 \
                + self.k * self.x[n] * (self.alpha + 3 * self.beta * pow(self.phi[n], 2))
            self.y[n+1] = self.a * self.y[n] - self.b * self.x[n] + self.c
            self.phi[n+1] = self.k1 * self.x[n] - self.k2 * self.phi[n]

        return self.X, self.Y, self.Phi, \
            self.Vx, self.Vy, self.Vphi, \
                self.dx, self.dy, self.dphi, \
                    self.fx, self.fy, self.fphi, \
                        self.x[self.Plot_Start : self.Plot_End - 1], self.y[self.Plot_Start : self.Plot_End - 1], self.phi[self.Plot_Start : self.Plot_End - 1]
    

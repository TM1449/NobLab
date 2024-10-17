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

#====================================================================
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

        #実行時間
        self.RunTime = self.Param["RunTime"]
        self.Plot_Start = self.Param["Plot_Start"]
        self.Plot_End = self.Param["Plot_End"]

        #====================================================================
        #初期値Xについて
        self.Initial_Value_X = self.Param["Initial_Value_X"]

        if self.Initial_Value_X == None:
            self.x = np.random.uniform(-1,1,self.RunTime)
        else:
            self.x = np.zeros(self.RunTime)
            self.x[0] = self.Initial_Value_X
        
        #初期値Yについて
        self.Initial_Value_Y = self.Param["Initial_Value_Y"]

        if self.Initial_Value_Y == None:
            self.y = np.random.uniform(-1,1,self.RunTime)
        else:
            self.y = np.zeros(self.RunTime)
            self.y[0] = self.Initial_Value_Y
        
        #初期値Phiについて
        self.Initial_Value_Phi = self.Param["Initial_Value_Phi"]

        if self.Initial_Value_Phi == None:
            self.phi = np.random.uniform(-1,1,self.RunTime)
        else:
            self.phi = np.zeros(self.RunTime)
            self.phi[0] = self.Initial_Value_Phi
        #====================================================================

    def __call__(self):
        #Chialvoニューロンの差分方程式の計算部
        for i in range(self.RunTime - 1):

            print("\r%d / %d"%(i, self.RunTime), end = "")

            self.x[i+1] = pow(self.x[i], 2) * np.exp(self.y[i] - self.x[i]) + self.k0 \
                + self.k * self.x[i] * (self.alpha + 3 * self.beta * pow(self.phi[i], 2))
            self.y[i+1] = self.a * self.y[i] - self.b * self.x[i] + self.c
            self.phi[i+1] = self.k1 * self.x[i] - self.k2 * self.phi[i]

        return self.x[self.Plot_Start : self.Plot_End - 1], \
            self.y[self.Plot_Start : self.Plot_End - 1], \
            self.phi[self.Plot_Start : self.Plot_End - 1]

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

        #ベクトル場の間隔
        self.Et = self.Param["Et"]
        
        #計算用の間隔
        self.dt = self.Param["dt"]

        self.Plot_x_Start = self.Param["Plot_x_Start"]
        self.Plot_x_End = self.Param["Plot_x_End"]
        
        self.Plot_y_Start = self.Param["Plot_y_Start"]
        self.Plot_y_End = self.Param["Plot_y_End"]
        #====================================================================
        
    def __call__(self):
        #2D-ChialvoニューロンマップのNullclineとベクトル場の計算部
        print("Old_Nullcline")
        """
        Et, Ex, Eyはベクトル場の変数.
        """    
        self.Ex = np.arange(self.Plot_x_Start, self.Plot_x_End, self.Et)
        self.Ey = np.arange(self.Plot_y_Start, self.Plot_y_End, self.Et)

        self.X, self.Y = np.meshgrid(self.Ex, self.Ey)
        
        self.Ex = (pow(self.X, 2) * np.exp(self.Y - self.X) + self.k0) - self.X
        self.Ey = (self.a * self.Y  - self.b * self.X + self.c) - self.Y

        """
        dt, dx, dyはNullclineの変数.
        """    
        self.dx = np.arange(self.Plot_x_Start, self.Plot_x_End, self.dt)
        self.dy = np.arange(self.Plot_y_Start, self.Plot_y_End, self.dt)

        self.fx = np.log(self.dx - self.k0) - 2 * np.log(self.dx) + self.dx
        self.fy = (self.a * self.dy + self.c - self.dy) / self.b

        """
        x, yは相平面の変数.
        """
        self.x = np.random.uniform(1,1.01,self.RunTime)
        self.y = np.random.uniform(1,1.01,self.RunTime)

        for n in range(self.RunTime - 1):
            self.x[n+1] = pow(self.x[n], 2) * np.exp(self.y[n] - self.x[n]) + self.k0
            self.y[n+1] = self.a * self.y[n] - self.b * self.x[n] + self.c

        return self.dx, self.dy, self.X, self.Y, \
                self.Ex, self.Ey, self.fx, self.fy, \
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

        #ベクトル場の間隔
        self.Et = self.Param["Et"]
        
        #計算用の間隔
        self.dt = self.Param["dt"]

        self.Plot_x_Start = self.Param["Plot_x_Start"]
        self.Plot_x_End = self.Param["Plot_x_End"]
        
        self.Plot_y_Start = self.Param["Plot_y_Start"]
        self.Plot_y_End = self.Param["Plot_y_End"]
        #====================================================================
        
    def __call__(self):
        #3D-ChialvoニューロンマップのNullclineとベクトル場の計算部
        print("Old_Nullcline")
        """
        Et, Ex, Eyはベクトル場の変数.
        """    
        self.Ex = np.arange(self.Plot_x_Start, self.Plot_x_End, self.Et)
        self.Ey = np.arange(self.Plot_y_Start, self.Plot_y_End, self.Et)

        self.X, self.Y = np.meshgrid(self.Ex, self.Ey)
        
        self.Ex = pow(self.X, 2) * np.exp(self.Y - self.X) + self.k0 - self.X
        self.Ey = (self.a * self.Y  - self.b * self.X + self.c - self.Y)

        """
        dt, dx, dyはNullclineの変数.
        """    
        self.dx = np.arange(self.Plot_x_Start, self.Plot_x_End, self.dt)
        self.dy = np.arange(self.Plot_y_Start, self.Plot_y_End, self.dt)

        self.fx = np.log(self.dx - self.k0) - 2 * np.log(self.dx) + self.dx
        self.fy = (self.a * self.dy + self.c - self.dy) / self.b

        """
        px, pyは相平面の変数.
        """

        self.x = np.random.uniform(-1,1,self.RunTime)
        self.y = np.random.uniform(-1,1,self.RunTime)

        for n in range(self.RunTime - 1):
            self.x[n+1] = pow(self.x[n], 2) * np.exp(self.y[n] - self.x[n]) + self.k0
            self.y[n+1] = self.a * self.y[n] - self.b * self.x[n] + self.c

        return self.dx, self.dy, \
            self.X, self.Y, \
                self.Ex, self.Ey, \
                    self.fx, self.fy, \
                        self.x[self.Plot_Start : self.Plot_End - 1], self.y[self.Plot_Start : self.Plot_End - 1]
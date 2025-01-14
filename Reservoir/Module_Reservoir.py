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
#・DeepAESN未移植＠Ver2
#・IP未移植＠Ver1
#・ChESN未移植＠Ver1
#・リングAESN未実装
#・２次元リザバー未実装

#====================================================================
import numpy as np

import Module

#====================================================================
#モジュール

#********************************************************************
#リザバー

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#継承元
class Module_Reservoir(Module.Module):
    """
    リザバーモジュール
    全てのリザバーは以下のインタフェイスを備える．
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: pass

    #時間発展
    def update(self): pass
        
    #初期化
    def reset(self): pass

    #ディープコピー
    def clone(self): pass
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#利用可能モジュール
class Module_SubReservoir(Module_Reservoir):
    """
    サブリザバーモジュール
    どのモデルのリザバーもこのクラスを使うようにできたら良いなあ．
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)

        #パラメータ取得
        self.D_u = self.Param["SubReservoir_D_u"]                       #入力信号次元＋バイアス
        self.D_x = self.Param["SubReservoir_D_x"]                       #ニューロン数
        self.LeakingRate = self.Param["SubReservoir_LeakingRate"]       #リーク率
        self.InputScale = self.Param["SubReservoir_InputScale"]         #入力スケーリング
        self.Rho = self.Param["SubReservoir_Rho"]                       #スペクトル半径
        self.Density = self.Param["SubReservoir_Density"]               #結合密度
        self.ActivationFunc = self.Param["SubReservoir_ActivationFunc"] #活性化関数
        self.Bias = np.ones([1])                                        #バイアス

        #変数
        self.s = np.zeros([self.D_x])                           #状態ベクトル
        self.s_old = self.s                                     #1step前の状態ベクトル
        #重み初期化
        self.W_in = self._makeInputWeight()                     #入力重み
        self.W_rec = self._makeRecurrentWeight()                #再帰重み

    #入力重み生成
    def _makeInputWeight(self) -> np.ndarray:
        return np.concatenate([
            np.random.rand(1, self.D_x),                        #バイアスに掛かる重み（正）
            np.random.rand(self.D_u, self.D_x) * 2 - 1]         #入力信号に掛かる重み（正負）
                              ) * self.InputScale               #入力スケールをかける
    
    #再帰重み生成
    def _makeRecurrentWeight(self) -> np.ndarray:
        W = np.random.rand(self.D_x, self.D_x) * 2 - 1
        W = self._makeWSparse(W)
        w, v = np.linalg.eig(W)
        return self.Rho * W / np.amax(w.real)
        
    #疎行列化
    def _makeWSparse(self, w: np.ndarray) -> np.ndarray:
        s_w = w.reshape([-1])
        s_w[np.random.choice(len(s_w), (int)(len(s_w) * (1 - self.Density)), replace = False)] = 0.
        return s_w.reshape(w.shape[0], w.shape[1])
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        self.s = (1 - self.LeakingRate) * self.s_old + \
            self.LeakingRate * self.ActivationFunc(
                np.dot(np.concatenate([self.Bias, u]), self.W_in) + 
                np.dot(self.s_old, self.W_rec))
        return self.s

    #時間発展
    def update(self):
        self.s_old = self.s

    #リザバーの初期化
    def reset(self):
        self.s = np.zeros([self.D_x])
        self.s_old = self.s
        
    #ディープコピー
    def clone(self):
        new = Module_SubReservoir(self.Param.copy(), self.Parent)
        
        new.s = self.s
        new.s_old = self.s_old
        new.W_in = self.W_in
        new.W_rec = self.W_rec

        return new
    
#--------------------------------------------------------------------
class Module_AESNReservoir(Module_Reservoir):
    """
    AESNReservoirモジュール
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)

        #パラメータ取得
        self.D_u = self.Param["AESNReservoir_D_u"]          #入力信号次元
        self.D_x = self.Param["AESNReservoir_D_x"]          #サブリザバーニューロン数
        
        #AESNwithHub用
        if "AESNReservoir_F_Use_x2AESNConnection" in self.Param:
            self.F_Use_x2AESNConnection = self.Param["AESNReservoir_F_Use_x2AESNConnection"]
            self.D_Hub_x = self.Param["AESNReservoir_D_Hub_x"]
        else:
            self.F_Use_x2AESNConnection = False

        #サブリザバーインスタンス
        self.L_SubReservoir = [None for _ in range(self.D_u)]
        for i in range(self.D_u):
            param = self.Param.copy()
            param.update({
                "SubReservoir_Index" : i,
                "SubReservoir_D_u" : 1 if not self.F_Use_x2AESNConnection else (1 + self.D_Hub_x),
                "SubReservoir_D_x" : self.D_x
                })
            self.L_SubReservoir[i] = Module_SubReservoir(param, self)
        
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray:
        U = np.array([[u[i]] for i in range(self.D_u)])
        if self.F_Use_x2AESNConnection:
            U = np.array([np.concatenate([np.array([u[i]]), self.Parent.HubReservoir.s_old]) for i in range(self.D_u)])

        return np.array([sub_esn.forward(U[i]) for i, sub_esn in enumerate(self.L_SubReservoir)]).reshape([-1])
            
    #時間発展
    def update(self):
        for sub_esn in self.L_SubReservoir: 
            sub_esn.update()

    #リザバーの初期化
    def reset(self):
        for sub_esn in self.L_SubReservoir: 
            sub_esn.reset()

    #ディープコピー
    def clone(self):
        new = Module_AESNReservoir(self.Param.copy(), self.Parent)
        new.L_SubReservoir = [sub_esn.clone() for sub_esn in self.L_SubReservoir]
            
        return new

    def get_S_old(self) -> np.ndarray:
        return np.array([sub_esn.s_old for sub_esn in self.L_SubReservoir])

#--------------------------------------------------------------------
class Module_DifferentUpdateESNReservoir(Module_Reservoir):
    """
    更新速度の異なるESNReservoirモジュール
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)

        #パラメータ取得
        self.D_u = self.Param["DifferentUpdateESNReservoir_D_u"]                            #入力信号次元
        self.D_x = self.Param["DifferentUpdateESNReservoir_D_x"]                            #サブリザバーニューロン数
        self.leakingRate_list = self.Param["DifferentUpdateESNReservoir_LeakingRate"]       #リーク率配列->サブリザバーの数(必ずリスト型)
        self.N_SubReservoirs = len(self.leakingRate_list)                                        #サブリザバー数
        self.InputScale = self.Param["DifferentUpdateESNReservoir_InputScale"]              #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["DifferentUpdateESNReservoir_Rho"]                            #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["DifferentUpdateESNReservoir_Density"]                    #結合密度配列（リスト型可能，None可能）
        
        #サブリザバーインスタンス
        self.L_SubReservoir = [None for _ in range(self.N_SubReservoirs)]
        for i in range(self.N_SubReservoirs):
            param = self.Param.copy()
            param.update({
                "SubReservoir_Index" : i,
                "SubReservoir_D_u" : self.D_u,
                "SubReservoir_D_x" : self.inputParam_NumOrList(i, self.D_x),
                "SubReservoir_LeakingRate" : self.leakingRate_list[i]
                })
            if self.InputScale is not None: 
                param.update({"SubReservoir_InputScale" : self.inputParam_NumOrList(i, self.InputScale)})
            if self.Rho is not None: 
                param.update({"SubReservoir_Rho" : self.inputParam_NumOrList(i, self.Rho)})
            if self.Density is not None: 
                param.update({"SubReservoir_Density" : self.inputParam_NumOrList(i, self.Density)})

            self.L_SubReservoir[i] = Module_SubReservoir(param, self)
        
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray:
        #リードアウトに掛ける形に変形 reshape(-1)
        return np.array([sub_esn.forward(u) for sub_esn in self.L_SubReservoir]).reshape([-1])
            
    #時間発展
    def update(self):
        for sub_esn in self.L_SubReservoir: 
            sub_esn.update()

    #リザバーの初期化
    def reset(self):
        for sub_esn in self.L_SubReservoir: 
            sub_esn.reset()

    #ディープコピー
    def clone(self):
        new = Module_DifferentUpdateESNReservoir(self.Param.copy(), self.Parent)
        new.L_SubReservoir = [sub_esn.clone() for sub_esn in self.L_SubReservoir]
            
        return new

    def get_S_old(self) -> np.ndarray:
        return np.array([sub_esn.s_old for sub_esn in self.L_SubReservoir])

class Module_DifferentUpdateAESNReservoir(Module_Reservoir):
    """
    更新速度の異なるAESNReservoirモジュール
    入力を次元ごとに分ける
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)

        #パラメータ取得
        self.D_u = self.Param["DifferentUpdateAESNReservoir_D_u"]                            #入力信号次元
        self.D_x = self.Param["DifferentUpdateAESNReservoir_D_x"]                            #サブリザバーニューロン数
        self.leakingRate_list = self.Param["DifferentUpdateAESNReservoir_LeakingRate"]       #リーク率配列->サブリザバーの数(必ずリスト型)
        self.N_SubReservoirs = len(self.leakingRate_list)                                        #サブリザバー数
        self.InputScale = self.Param["DifferentUpdateAESNReservoir_InputScale"]              #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["DifferentUpdateAESNReservoir_Rho"]                            #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["DifferentUpdateAESNReservoir_Density"]                    #結合密度配列（リスト型可能，None可能）
        
        #サブリザバーインスタンス
        self.L_SubReservoir = [None for _ in range(self.N_SubReservoirs)]
        for i in range(self.N_SubReservoirs):
            param = self.Param.copy()
            param.update({
                "SubReservoir_Index" : i,
                "SubReservoir_D_u" : 1,
                "SubReservoir_D_x" : self.inputParam_NumOrList(i, self.D_x),
                "SubReservoir_LeakingRate" : self.leakingRate_list[i]
                })
            if self.InputScale is not None: 
                param.update({"SubReservoir_InputScale" : self.inputParam_NumOrList(i, self.InputScale)})
            if self.Rho is not None: 
                param.update({"SubReservoir_Rho" : self.inputParam_NumOrList(i, self.Rho)})
            if self.Density is not None: 
                param.update({"SubReservoir_Density" : self.inputParam_NumOrList(i, self.Density)})

            self.L_SubReservoir[i] = Module_SubReservoir(param, self)
        
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray:
        #リードアウトに掛ける形に変形 reshape(-1)
        return np.array([sub_esn.forward([u[i]]) for i, sub_esn in enumerate(self.L_SubReservoir)]).reshape([-1])
            
    #時間発展
    def update(self):
        for sub_esn in self.L_SubReservoir: 
            sub_esn.update()

    #リザバーの初期化
    def reset(self):
        for sub_esn in self.L_SubReservoir: 
            sub_esn.reset()

    #ディープコピー
    def clone(self):
        new = Module_DifferentUpdateAESNReservoir(self.Param.copy(), self.Parent)
        new.L_SubReservoir = [sub_esn.clone() for sub_esn in self.L_SubReservoir]
            
        return new

    def get_S_old(self) -> np.ndarray:
        return np.array([sub_esn.s_old for sub_esn in self.L_SubReservoir])

#--------------------------------------------------------------------
class Module_AESNwithHubReservoir(Module_Reservoir):
    """
    Hubを持ったAESNモジュール
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)
        
        #パラメータ取得
        self.D_u = self.Param["AESNwithHubReservoir_D_u"]       #入力信号次元
        self.D_h = self.Param["AESNwithHubReservoir_D_h"]       #サブリザバーニューロン数
        self.D_H = self.D_u * self.D_h                          #AESN総ニューロン数
        self.D_x = self.Param["AESNwithHubReservoir_D_x"]       #ハブリザバーニューロン数
        self.F_Use_U2HubConnection = self.Param["AESNwithHubReservoir_F_Use_U2HubConnection"]   #Hubへの入力信号使用の有無
        self.F_Use_x2zConnection = self.Param["AESNwithHubReservoir_F_Use_x2zConnection"]       #出力へのHub状態使用の有無
        self.F_Use_x2AESNConnection = self.Param["AESNwithHubReservoir_F_Use_x2AESNConnection"] #AESNへのHub信号使用の有無
        self.F_Use_AverageHInHub = self.Param["AESNwithHubReservoir_F_Use_AverageHInHub"]       #HubでHの平均をするか
        self.Hub_LeakingRate = self.Param["AESNwithHubReservoir_Hub_LeakingRate"]               #Hubリーク率配列（None可能）
        self.Hub_InputScale = self.Param["Model_AESNwithHub_Hub_InputScale"]                    #Hub入力スケーリング配列（None可能）
        self.Hub_Rho = self.Param["Model_AESNwithHub_Hub_Rho"]                                  #Hubスペクトル半径配列（None可能）
        self.Hub_Density = self.Param["Model_AESNwithHub_Hub_Density"]                          #Hub結合密度配列（None可能）

        #Hub入力次元算出
        if self.F_Use_AverageHInHub:
            self.D_Hub_input = self.D_h
        else:
            self.D_Hub_input = self.D_H

        if self.F_Use_U2HubConnection:
            self.D_Hub_input += self.D_u

        #リザバーモジュールインスタンス
        #AESN
        param = self.Param.copy()
        param.update({
            "AESNReservoir_D_u" : self.D_u,
            "AESNReservoir_D_x" : self.D_h,
            "AESNReservoir_F_Use_x2AESNConnection" : self.F_Use_x2AESNConnection,
            "AESNReservoir_D_Hub_x" : self.D_x
            })
        self.AESNReservoir = Module_AESNReservoir(param, self)
        
        #Hub
        param = self.Param.copy()
        param.update({
            "SubReservoir_D_u" : self.D_Hub_input,
            "SubReservoir_D_x" : self.D_x
            })
        if self.Hub_LeakingRate is not None: 
            param.update({"SubReservoir_LeakingRate" : self.Hub_LeakingRate})
        if self.Hub_InputScale is not None: 
            param.update({"SubReservoir_InputScale" : self.Hub_InputScale})
        if self.Hub_Rho is not None: 
            param.update({"SubReservoir_Rho" : self.Hub_Rho})
        if self.Hub_Density is not None: 
            param.update({"SubReservoir_Density" : self.Hub_Density})
        self.HubReservoir = Module_SubReservoir(param, self)
        
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        H = self.AESNReservoir.forward(u)

        H_old = self.AESNReservoir.get_S_old()
        if self.F_Use_AverageHInHub:
            Hub_input = np.mean(H_old, axis = 0)
        else:
            Hub_input = H_old.reshape([-1])

        if self.F_Use_U2HubConnection:
            x = self.HubReservoir.forward(np.concatenate([Hub_input, u])) 
        else:
            x = self.HubReservoir.forward(Hub_input)
        
        if self.F_Use_x2zConnection:
            return np.concatenate([H, x])
        else:
            return H
        
    #時間発展
    def update(self):
        self.AESNReservoir.update()
        self.HubReservoir.update()

    #リザバーの初期化
    def reset(self):
        self.AESNReservoir.reset()
        self.HubReservoir.reset()

    #ディープコピー
    def clone(self):
        new = Module_AESNwithHubReservoir(self.Param)
        new.AESNReservoir = self.AESNReservoir.clone()
        new.HubReservoir = self.HubReservoir.clone()

        return new
    
#--------------------------------------------------------------------
class Module_ModifiedDeepReservoir(Module_Reservoir):
    """
    変形DeepReservoirモジュール
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)

        #パラメータ取得
        self.D_u = self.Param["ModifiedDeepReservoir_D_u"]  #入力信号次元
        self.D_x = self.Param["ModifiedDeepReservoir_D_x"]  #各層ニューロン数（リスト型）
        self.D_l = len(self.D_x)                            #層の数
        self.LeakingRate = self.Param["ModifiedDeepReservoir_LeakingRate"]  #リーク率配列（リスト型可能，None可能）
        self.InputScale = self.Param["ModifiedDeepReservoir_InputScale"]    #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["ModifiedDeepReservoir_Rho"]                  #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["ModifiedDeepReservoir_Density"]          #結合密度配列（リスト型可能，None可能）

        #サブリザバーインスタンス
        self.L_SubReservoir = [None for _ in range(self.D_l)]
        for i in range(self.D_l):
            param = self.Param.copy()
            param.update({
                "SubReservoir_Index" : i,
                "SubReservoir_D_u" : self.D_u if i == 0 else self.D_x[i - 1],
                "SubReservoir_D_x" : self.D_x[i]
                })
            if self.LeakingRate is not None: 
                param.update({"SubReservoir_LeakingRate" : self.inputParam_NumOrList(i, self.LeakingRate)})
            if self.InputScale is not None: 
                param.update({"SubReservoir_InputScale" : self.inputParam_NumOrList(i, self.InputScale)})
            if self.Rho is not None: 
                param.update({"SubReservoir_Rho" : self.inputParam_NumOrList(i, self.Rho)})
            if self.Density is not None: 
                param.update({"SubReservoir_Density" : self.inputParam_NumOrList(i, self.Density)})
            self.L_SubReservoir[i] = Module_SubReservoir(param, self)

    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = np.zeros([0])
        u_i = u
        for sub_esn in self.L_SubReservoir:
            u_i = sub_esn.forward(u_i)
            s = np.concatenate([s, u_i])
        return s

    #時間発展
    def update(self):
        for sub_esn in self.L_SubReservoir:
            sub_esn.update()
        
    #リザバーの初期化
    def reset(self):
        for sub_esn in self.L_SubReservoir:
            sub_esn.reset()

    #ディープコピー
    def clone(self):
        new = Module_ModifiedDeepReservoir(self.Param.copy(), self.Parent)
        new.L_SubReservoir = [sub_esn.clone() for sub_esn in self.L_SubReservoir]

        return new

#--------------------------------------------------------------------
#Sishu提案モデル

def ReLU(x):
    return np.where(x > 0, x , 0.1 * x)

class Module_SishuReservoir(Module_Reservoir):
    """
    Sishu提案モデル
    ニューロン間の結合形態は指定、結合強度は乱数で与える
    スケーリングサイズを組み込む
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)

        #パラメータ取得
        self.D_u = self.Param["SishuReservoir_D_u"]                       #入力信号次元＋バイアス
        self.D_x = self.Param["SishuReservoir_D_x"]                       #ニューロン数
        self.InputScale = self.Param["SishuReservoir_InputScale"]         #入力スケーリング
        
        self.sigma = self.Param["SishuReservoir_sigma"]                     #リング・ネットワークの有無
        self.mu = self.Param["SishuReservoir_mu"]                           #スター・ネットワークの有無
        self.k = self.Param["SishuReservoir_k"]                             #k
        self.Rho = self.Param["SishuReservoir_Rho"]                         #スペクトル半径
        self.Density = self.Param["SishuReservoir_Density"]
        self.Bias = np.ones([1])                                        #バイアス

        #変数
        self.x = np.zeros([self.D_x])                           #xの状態ベクトル
        self.x_old = self.x                                     #1step前のxの状態ベクトル
        self.y = np.ones([self.D_x])                           #xの状態ベクトル
        self.y_old = self.y                                     #1step前のxの状態ベクトル
        self.phi = np.ones([self.D_x])                           #xの状態ベクトル
        self.phi_old = self.phi                                     #1step前のxの状態ベクトル
        
        #重み初期化
        self.W_in = self._makeInputWeight()                     #入力重み
        self.W_rec = self._makeRecurrentWeight()                #再帰重み

    #入力重み生成
    def _makeInputWeight(self) -> np.ndarray:
        return np.concatenate([
            np.random.rand(1, self.D_x),                        #バイアスに掛かる重み（正）
            np.random.rand(self.D_u, self.D_x) * 2 - 1]         #入力信号に掛かる重み（正負）
                              ) * self.InputScale               #入力スケールをかける
    
    #再帰重み生成
    def _makeRecurrentWeight(self) -> np.ndarray:
        #リングスターネットワークの隣接行列
        #使用するハイパーパラメータ
        N = 100
        R = 10

        if (self.sigma == "True") and (self.mu == "False"):
            #リングネットワークを乱数で生成
            Ring_Matrix = np.identity(self.D_x)
            Ring_Matrix[0,0] = 0
            for i in range(1,self.D_x):
                Ring_Matrix[i:i+R+1,i:i+R+1] = 1
                Ring_Matrix[N-R+i-1:,i] = 1
                Ring_Matrix[i,N-R+i-1:] = 1
            for i in range(1,self.D_x):
                Ring_Matrix[i,i] = 1
            
            Matrix = Ring_Matrix
        
        elif (self.sigma == "False") and (self.mu == "True"):
            #スターネットワークの生成
            Star_Matrix = np.zeros((self.D_x,self.D_x))
            Star_Matrix[0,0] = 1
            Star_Matrix[0,1:] = 1
            Star_Matrix[1:,0] = 1
            for i in range(1,N):
                Star_Matrix[i,i] += 1
            
            Matrix = Star_Matrix

        elif (self.sigma == "True") and (self.mu == "True"):
            #リングネットワークの生成
            Ring_Matrix = np.identity(self.D_x)
            Ring_Matrix[0,0] = 0
            for i in range(1,self.D_x):
                Ring_Matrix[i:i+R+1,i:i+R+1] = 1
                Ring_Matrix[N-R+i-1:,i] = 1
                Ring_Matrix[i,N-R+i-1:] = 1
            for i in range(1,self.D_x):
                Ring_Matrix[i,i] = 0

            #スターネットワークの生成
            Star_Matrix = np.zeros((self.D_x,self.D_x))
            Star_Matrix[0,0] = 1
            Star_Matrix[0,1:] = 1
            Star_Matrix[1:,0] = 1
            for i in range(1,self.D_x):
                Star_Matrix[i,i] += 1

            #リングスターネットワークの生成
            Matrix = (Ring_Matrix + Star_Matrix)

        elif (self.sigma == "False") and (self.mu == "False"):
            
            #乱数で重み調整
            np.random.seed(seed=None)
        
            W = (np.random.rand(self.D_x,self.D_x) * 2 - 1)
            W = self._makeWSparse(W)
            w , v = np.linalg.eig(W)

            Matrix = self.Rho * W / np.amax(w.real)

        else:
            #リングネットワークの生成
            Ring_Matrix = np.identity(self.D_x)
            Ring_Matrix[0,0] = 0
            for i in range(1,self.D_x):
                Ring_Matrix[i:i+R+1,i:i+R+1] = 1
                Ring_Matrix[N-R+i-1:,i] = 1
                Ring_Matrix[i,N-R+i-1:] = 1
            for i in range(1,self.D_x):
                Ring_Matrix[i,i] = -2 * R

            Ring_Matrix *= (self.sigma / (2 * R))

            #スターネットワークの生成
            Star_Matrix = np.zeros((self.D_x,self.D_x))
            Star_Matrix[0,0] = -self.mu * (self.D_x-1)
            Star_Matrix[0,1:] = -self.mu
            Star_Matrix[1:,0] = self.mu
            for i in range(1,self.D_x):
                Star_Matrix[i,i] += self.mu

            #リングスターネットワークの生成
            Matrix = Ring_Matrix + Star_Matrix
        
        return Matrix
        
    #疎行列化
    def _makeWSparse(self, w: np.ndarray) -> np.ndarray:
        s_w = w.reshape([-1])
        s_w[np.random.choice(len(s_w), (int)(len(s_w) * (1 - self.Density)), replace = False)] = 0.
        return s_w.reshape(w.shape[0], w.shape[1])
    
    def ReLU(x):
        #return np.where(x > 0, x , 0.0 * x)
        return x

    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        #使用するハイパーパラメータ
        a = 0.89
        b = 0.6
        c = 0.28
        k0 = 0.04
        k1 = 0.1
        k2 = 0.2
        Alph_Chialvo = 0.1
        Beta_Chialvo = 0.2
        
        self.x = ReLU(self.x_old ** 2 * np.exp(self.y_old - self.x_old) + k0 \
                        + self.k * self.x_old * (Alph_Chialvo + 3 * Beta_Chialvo * self.phi_old ** 2) \
                                + np.dot(self.x_old, self.W_rec)) + np.dot(np.concatenate([self.Bias, u]), self.W_in)
        self.y = a * self.y_old - b * self.x_old + c
        self.phi = k1 * self.x_old - k2 * self.phi_old

        return self.x

    #時間発展
    def update(self):
        self.x_old = self.x
        self.y_old = self.y
        self.phi_old = self.phi

    #リザバーの初期化
    def reset(self):
        self.x = np.zeros([self.D_x])
        self.x_old = self.x
        self.y = np.ones([self.D_x])
        self.y_old = self.y
        self.phi = np.ones([self.D_x])
        self.phi_old = self.phi

    #ディープコピー
    def clone(self):
        new = Module_SishuReservoir(self.Param.copy(), self.Parent)
        
        new.x = self.x
        new.x_old = self.x_old
        new.y = self.y
        new.y_old = self.y_old
        new.phi = self.phi
        new.phi_old = self.phi_old

        new.W_in = self.W_in
        new.W_rec = self.W_rec

        return new
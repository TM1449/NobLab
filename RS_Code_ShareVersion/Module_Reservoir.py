#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大
#作成日：2023/05/24
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
import torch

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
    
    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        pass
    
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
        self.W_res = self._makeRecurrentWeight()                #再帰重み

    #入力重み生成
    def _makeInputWeight(self) -> np.ndarray:
        return np.concatenate([
            np.random.rand(self.D_u + 1, self.D_x) * 2 - 1]         #入力信号に掛かる重み（正負）
                              ) * self.InputScale               #入力スケールをかける
        #return np.concatenate([
        #    np.random.rand(1, self.D_x),                        #バイアスに掛かる重み（正）
        #    np.random.rand(self.D_u, self.D_x) * 2 - 1]         #入力信号に掛かる重み（正負）
        #                      ) * self.InputScale               #入力スケールをかける
    
    #再帰重み生成
    def _makeRecurrentWeight(self) -> np.ndarray:
        W = np.random.rand(self.D_x, self.D_x) * 2 - 1
        W = self._makeWSparse(W)
        w, v = np.linalg.eig(W)
        return self.Rho * W / np.amax(np.abs(w))
        
    #疎行列化
    def _makeWSparse(self, w: np.ndarray) -> np.ndarray:
        s_w = w.reshape([-1])
        s_w[np.random.choice(len(s_w), (int)(len(s_w) * (1 - self.Density)), replace = False)] = 0.
        return s_w.reshape(w.shape[0], w.shape[1])
    
    #順伝播
    def forward(self, u: np.ndarray or torch.tensor) -> np.ndarray or torch.tensor: 
        self.s = (1 - self.LeakingRate) * self.s_old + \
            self.LeakingRate * self.ActivationFunc(
                np.dot(np.concatenate([self.Bias, u]), self.W_in) + 
                np.dot(self.s_old, self.W_res))
        return self.s

    #時間発展
    def update(self):
        self.s_old = self.s.copy()

    #リザバーの初期化
    def reset(self):
        self.s = np.zeros([self.D_x])
        self.s_old = self.s.copy()
        
    #ディープコピー
    def clone(self):
        new = Module_SubReservoir(self.Param.copy(), self.Parent)
        
        new.s = self.s.copy()
        new.s_old = self.s_old.copy()
        new.W_in = self.W_in.copy()
        new.W_res = self.W_res.copy()

        return new
    
    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        return [self]

    #偏微分（通常ESN以外のモデルでは偏微分の数式が違う場合があり確認が必要．かつ，活性化関数はtanh限定．）
    def Jacobian(self, u: np.ndarray or torch.tensor) -> np.ndarray or torch.tensor: 
        return (1 - self.LeakingRate) * np.eye(self.D_x) + \
            self.LeakingRate * np.dot(np.diag(1 - np.tanh(
                np.dot(np.concatenate([self.Bias, u]), self.W_in) + 
                np.dot(self.s_old, self.W_res))**2), self.W_res)
    
class Module_Torch_SubReservoir(Module_SubReservoir):
    """
    Torch版サブリザバーモジュール
    どのモデルのリザバーもこのクラスを使うようにできたら良いなあ．
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super(Module_SubReservoir, self).__init__(param, parent)

        #パラメータ取得
        self.D_u = self.Param["SubReservoir_D_u"]                       #入力信号次元＋バイアス
        self.D_x = self.Param["SubReservoir_D_x"]                       #ニューロン数
        self.LeakingRate = self.Param["SubReservoir_LeakingRate"]       #リーク率
        self.InputScale = self.Param["SubReservoir_InputScale"]         #入力スケーリング
        self.Rho = self.Param["SubReservoir_Rho"]                       #スペクトル半径
        self.Density = self.Param["SubReservoir_Density"]               #結合密度
        
        #ネットワーク生成
        self.Network = Network_SubReservoir(param, self)
        
    #順伝播
    def forward(self, u: np.ndarray or torch.tensor) -> np.ndarray or torch.tensor: 
        return self.Network.forward(u)
        
    #時間発展
    def update(self):
        self.Network.update()
        
    #リザバーの初期化
    def reset(self):
        self.Network.reset()
        
    #ディープコピー
    def clone(self):
        return self.Network.clone()
        
class Network_SubReservoir(Module.Network):
    """
    Torchで実装したサブリザバーネットワーク
    """
    def __init__(self, param: dict, parent = None) -> None:
        super().__init__(param, parent)
        
        #パラメータ取得
        self.DeviceCode = self.Param["Project_DeviceCode"]              #CPU/GPUを使うか（CPU -> cpu, GPU -> gpu:n（nはデバイス番号，無くてもいい））
        self.DataType = self.Param["Project_DataType"]                  #Pytorchのデータ型

        self.D_u = self.Param["SubReservoir_D_u"]                       #入力信号次元＋バイアス
        self.D_x = self.Param["SubReservoir_D_x"]                       #ニューロン数
        self.LeakingRate = self.Param["SubReservoir_LeakingRate"]       #リーク率
        self.InputScale = self.Param["SubReservoir_InputScale"]         #入力スケーリング
        self.Rho = self.Param["SubReservoir_Rho"]                       #スペクトル半径
        self.Density = self.Param["SubReservoir_Density"]               #結合密度
        self.ActivationFunc = self.Param["SubReservoir_ActivationFunc"] #活性化関数
        self.Bias = np.ones([1])                                        #バイアス
        
        #ハイパーパラメータのtorchヘの変換
        #活性化関数
        if self.ActivationFunc is np.tanh:#tanhの場合
            self.Torch_ActivationFunc = torch.tanh
        self.Torch_LeakingRate = torch.tensor(self.LeakingRate, device = self.DeviceCode, dtype = self.DataType)
        self.Torch_Bias = torch.tensor(self.Bias, device = self.DeviceCode, dtype = self.DataType)

        #変数
        self.s = torch.zeros([self.D_x], device = self.DeviceCode, dtype = self.DataType)                       #状態ベクトル
        self.s_old = self.s                                                                                     #1step前の状態ベクトル
        #重み初期化
        self.W_in = torch.tensor(self.Parent._makeInputWeight(), device = self.DeviceCode, dtype = self.DataType)      #入力重み
        self.W_res = torch.tensor(self.Parent._makeRecurrentWeight(), device = self.DeviceCode, dtype = self.DataType) #再帰重み
        
    #順伝播
    def forward(self, u: torch.tensor) -> torch.tensor:
        self.s = (1 - self.Torch_LeakingRate) * self.s_old + \
            self.Torch_LeakingRate * self.Torch_ActivationFunc(
                torch.matmul(torch.cat([self.Torch_Bias, u]), self.W_in) + 
                torch.matmul(self.s_old, self.W_res))
        return self.s
    
    #時間発展
    def update(self):
        self.s_old = self.s.copy()

    #初期化
    #未実装#################################
    def reset(self):
        print("ERROR")

    #ディープコピー
    #未実装#################################
    def clone(self) -> any:
        print("ERROR")
    
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
                "ModuleName" : "SubRes",
                "ModuleShowName" : "Sub Reservoir No." + str(i),
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

    #状態古いベクトル取得
    def get_S_old(self) -> np.ndarray:
        return np.array([sub.s_old for sub in self.getSubReservoirs()])
    
    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        out = []
        for sub in self.L_SubReservoir:
            out += sub.getSubReservoirs()
        return out

#--------------------------------------------------------------------
class Module_HetAESNReservoir(Module_Reservoir):
    """
    リーク率の異なるマルチリザバーReservoirモジュール
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)

        #パラメータ取得
        self.D_u = self.Param["HetAESNReservoir_D_u"]                            #入力信号次元
        self.D_x = self.Param["HetAESNReservoir_D_x"]                            #サブリザバーニューロン数
        self.L_LeakingRate = self.Param["HetAESNReservoir_LeakingRate"]       #リーク率配列->サブリザバーの数(必ずリスト型)
        self.N_SubReservoirs = len(self.L_LeakingRate)                                    #サブリザバー数
        self.InputScale = self.Param["HetAESNReservoir_InputScale"]              #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["HetAESNReservoir_Rho"]                            #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["HetAESNReservoir_Density"]                    #結合密度配列（リスト型可能，None可能）
        
        #サブリザバーインスタンス
        self.L_SubReservoir = [None for _ in range(self.N_SubReservoirs)]
        for i in range(self.N_SubReservoirs):
            param = self.Param.copy()
            param.update({
                "ModuleName" : "SubRes",
                "ModuleShowName" : "Sub Reservoir No." + str(i),
                "SubReservoir_Index" : i,
                "SubReservoir_D_u" : self.D_u,
                "SubReservoir_D_x" : self.inputParam_NumOrList(i, self.D_x),
                "SubReservoir_LeakingRate" : self.L_LeakingRate[i]
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
        new = Module_HetAESNReservoir(self.Param.copy(), self.Parent)
        new.L_SubReservoir = [sub_esn.clone() for sub_esn in self.L_SubReservoir]
            
        return new
    
    #状態古いベクトル取得（未使用）
    def get_S_old(self) -> np.ndarray:
        return np.array([sub.s_old for sub in self.getSubReservoir()])
    
    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        out = []
        for sub in self.L_SubReservoir:
            out += sub.getSubReservoirs()
        return out

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
            "ModuleName" : "AESNRes",
            "ModuleShowName" : "AESN",
            "AESNReservoir_D_u" : self.D_u,
            "AESNReservoir_D_x" : self.D_h,
            "AESNReservoir_F_Use_x2AESNConnection" : self.F_Use_x2AESNConnection,
            "AESNReservoir_D_Hub_x" : self.D_x
            })
        self.AESNReservoir = Module_AESNReservoir(param, self)
        
        #Hub
        param = self.Param.copy()
        param.update({
            "ModuleName" : "HubRes",
            "ModuleShowName" : "Hub",
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
        new = Module_AESNwithHubReservoir(self.Param, self.Parent)
        new.AESNReservoir = self.AESNReservoir.clone()
        new.HubReservoir = self.HubReservoir.clone()

        return new
    
    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        return self.AESNReservoir.getSubReservoirs() + self.HubReservoir.getSubReservoirs()

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
                "ModuleName" : "SubRes",
                "ModuleShowName" : "Sub Reservoir No." + str(i),
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
    
    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        out = []
        for sub in self.L_SubReservoir:
            out += sub.getSubReservoirs()
        return out

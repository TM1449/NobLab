#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大
#作成日：2023/05/24
#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・２次元リザバー未実装
#・リングAESN未実装
#・DeepAESN未移植＠Ver2
#・オンライン学習未移植＠Ver1

#====================================================================
import numpy as np
import torch

import Module_Reservoir as Reservoir
import Module_Readout as Readout

#====================================================================
#モデル

#********************************************************************
#継承元
class Model:
    """
    モデル
    このクラスで，どんなモジュール（どんな複雑な構造）を使っても，
    評価のクラスでは同一のインタフェイスになるようにする．
    つまり，モデルの違いはここで全て吸収する．
    これにより，評価はモデルごとに作成しなくて良くなる（はず．）
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        self.Param = param
        
        self.Evaluation = evaluation                            #親オブジェクト
        
    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray: pass
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray: pass
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: pass
    
    #初期化
    def reset(self): pass

    #ディープコピー
    def clone(self) -> any: pass
    
    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        pass
    
#********************************************************************
#利用可能モデル

#--------------------------------------------------------------------
#通常ESN
class Model_NormalESN(Model):
    """
    通常ESNモデル（Torch対応済み）
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.F_UsePytorch = self.Param["Project_F_UsePytorch"]  #Pytorchを使うか（多層リードアウトでは強制的に使用）
        
        self.D_u = self.Param["Model_NormalESN_D_u"]            #入力信号次元
        self.D_x = self.Param["Model_NormalESN_D_x"]            #ニューロン数（リスト型）
        self.D_y = self.Param["Model_NormalESN_D_y"]            #出力信号次元
        self.D_z = self.D_x + self.D_u                          #特徴ベクトル次元
        
        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "Res",
            "ModuleShowName" : "Reservoir",
            "SubReservoir_D_u" : self.D_u,
            "SubReservoir_D_x" : self.D_x
            })
        if self.F_UsePytorch: self.SubReservoir = Reservoir.Module_Torch_SubReservoir(param, self)
        else: self.SubReservoir = Reservoir.Module_SubReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "LTRead",
            "ModuleShowName" : "LT Readout",
            "LinearTransformer_D_z" : self.D_z,
            "LinearTransformer_D_y" : self.D_y
            })
        if self.F_UsePytorch: self.Readout_LinearTransformer = Readout.Module_Torch_Readout_LinearTransformer(param, self)
        else: self.Readout_LinearTransformer = Readout.Module_Readout_LinearTransformer(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray or torch.tensor) -> np.ndarray or torch.tensor:
        s = self.SubReservoir.forward(u)
        z = torch.cat([s, u]) if self.F_UsePytorch else np.concatenate([s, u])
        self.SubReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray or torch.tensor) -> np.ndarray or torch.tensor:
        return self.Readout_LinearTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray or torch.tensor) -> np.ndarray or torch.tensor: 
        s = self.SubReservoir.forward(u)
        z = torch.cat([s, u]) if self.F_UsePytorch else np.concatenate([s, u])
        y = self.Readout_LinearTransformer.forward(z)
        self.SubReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray or torch.tensor, y_d: np.ndarray or torch.tensor) -> tuple: 
        s = self.SubReservoir.forward(u)
        z = torch.cat([s, u]) if self.F_UsePytorch else np.concatenate([s, u])
        y = self.Readout_LinearTransformer.forward(z)
        e = self.Readout_LinearTransformer.RMSE(z, y_d)
        self.SubReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray or torch.tensor, Y_d: np.ndarray or torch.tensor):
        self.Readout_LinearTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.SubReservoir.reset()
        self.Readout_LinearTransformer.reset()

    #ディープコピー
    def clone(self) -> any:
        new = Model_NormalESN(self.Param.copy(), self.Evaluation)
        new.SubReservoir = self.SubReservoir.clone()
        new.Readout_LinearTransformer = self.Readout_LinearTransformer.clone()
        return new

    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        return self.SubReservoir.getSubReservoirs()
        
class Model_ESN_DNNReadout(Model):
    """
    多層リードアウトESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_NormalESN_D_u"]            #入力信号次元
        self.D_x = self.Param["Model_NormalESN_D_x"]            #ニューロン数（リスト型）
        self.D_y = self.Param["Model_NormalESN_D_y"]            #出力信号次元
        self.D_z = self.D_x + self.D_u                          #特徴ベクトル次元
        
        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "Res",
            "ModuleShowName" : "Reservoir",
            "SubReservoir_D_u" : self.D_u,
            "SubReservoir_D_x" : self.D_x
            })
        self.SubReservoir = Reservoir.Module_SubReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "DNNRead",
            "ModuleShowName" : "DNNReadout",
            "DNN_D_z" : self.D_z,
            "DNN_D_y" : self.D_y
            })
        self.Readout_DNN = Readout.Module_Readout_DNN(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.SubReservoir.forward(u)
        z = np.concatenate([s, u])
        self.SubReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_DNN.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.SubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        self.SubReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.SubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        e = self.Readout_DNN.RMSE(z, y_d)
        self.SubReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_DNN.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.SubReservoir.reset()
        self.Readout_DNN.reset()

    #ディープコピー
    def clone(self) -> any:
        new = Model_ESN_DNNReadout(self.Param.copy(), self.Evaluation)
        new.SubReservoir = self.SubReservoir.clone()
        new.Readout_DNN = self.Readout_DNN.clone()
        return new

    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        return self.SubReservoir.getSubReservoirs()
        
#--------------------------------------------------------------------
#通常AESNモデル
class Model_NormalAESN(Model):
    """
    通常AESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_NormalAESN_D_u"]           #入力信号次元
        self.D_x = self.Param["Model_NormalAESN_D_x"]           #サブリザバーニューロン数
        self.D_s = self.D_u * self.D_x                          #総ニューロン数
        self.D_y = self.Param["Model_NormalAESN_D_y"]           #出力信号次元
        self.D_z = self.D_s + self.D_u                          #特徴ベクトル次元
        
        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "ASENRes",
            "ModuleShowName" : "AESN",
            "AESNReservoir_D_u" : self.D_u,
            "AESNReservoir_D_x" : self.D_x
            })
        self.AESNReservoir = Reservoir.Module_AESNReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "LTRead",
            "ModuleShowName" : "LT Readout",
            "LinearTransformer_D_z" : self.D_z,
            "LinearTransformer_D_y" : self.D_y
            })
        self.Readout_LinearTransformer = Readout.Module_Readout_LinearTransformer(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.AESNReservoir.forward(u)
        z = np.concatenate([s, u])
        self.AESNReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinearTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.AESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinearTransformer.forward(z)
        self.AESNReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.AESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinearTransformer.forward(z)
        e = self.Readout_LinearTransformer.RMSE(z, y_d)
        self.AESNReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_LinearTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.AESNReservoir.reset()
        self.Readout_LinearTransformer.reset()

    #ディープコピー
    def clone(self) -> any:
        new = Model_NormalAESN(self.Param.copy(), self.Evaluation)
        new.AESNReservoir = self.AESNReservoir.clone()
        new.Readout_LinearTransformer = self.Readout_LinearTransformer.clone()
        return new

    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        return self.AESNReservoir.getSubReservoirs()
        
class Model_AESN_DNNReadout(Model):
    """
    多層リードアウトAESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_NormalAESN_D_u"]           #入力信号次元
        self.D_x = self.Param["Model_NormalAESN_D_x"]           #サブリザバーニューロン数
        self.D_s = self.D_u * self.D_x                          #総ニューロン数
        self.D_y = self.Param["Model_NormalAESN_D_y"]           #出力信号次元
        self.D_z = self.D_s + self.D_u                          #特徴ベクトル次元
        
        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "ASENRes",
            "ModuleShowName" : "AESN Reservoir",
            "AESNReservoir_D_u" : self.D_u,
            "AESNReservoir_D_x" : self.D_x
            })
        self.AESNReservoir = Reservoir.Module_AESNReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "DNNRead",
            "ModuleShowName" : "DNNReadout",
            "DNN_D_z" : self.D_z,
            "DNN_D_y" : self.D_y
            })
        self.Readout_DNN = Readout.Module_Readout_DNN(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.AESNReservoir.forward(u)
        z = np.concatenate([s, u])
        self.AESNReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_DNN.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.AESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        self.AESNReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.AESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        e = self.Readout_DNN.RMSE(z, y_d)
        self.AESNReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_DNN.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.AESNReservoir.reset()
        self.Readout_DNN.reset()

    #ディープコピー
    def clone(self) -> any:
        new = Model_AESN_DNNReadout(self.Param.copy(), self.Evaluation)
        new.AESNReservoir = self.AESNReservoir.clone()
        new.Readout_DNN = self.Readout_DNN.clone()
        return new

    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        return self.AESNReservoir.getSubReservoirs()
        
#--------------------------------------------------------------------
#リーク率の異なるマルチリザバーモデル
class Model_HetAESN(Model):
    """
    リーク率の異なるマルチリザバーモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_HetAESN_D_u"]                   #入力信号次元
        self.D_x = self.Param["Model_HetAESN_D_x"]                   #サブリザバーニューロン数
        self.LeakingRate = self.Param["Model_HetAESN_LeakingRate"]   #リーク率配列->サブリザバーの数(必ずリスト型)
        self.N_SubReservoirs = len(self.LeakingRate)                        #サブリザバー数
        self.D_y = self.Param["Model_HetAESN_D_y"]                   #出力信号次元
        self.D_z = self.N_SubReservoirs * self.D_x + self.D_u               #特徴ベクトル次元
        self.InputScale = self.Param["Model_HetAESN_InputScale"]     #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["Model_HetAESN_Rho"]                   #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["Model_HetAESN_Density"]           #結合密度配列（リスト型可能，None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "DifferentLRRes",
            "ModuleShowName" : "DLR-ESN",
            "HetAESNReservoir_D_u" : self.D_u,
            "HetAESNReservoir_D_x" : self.D_x,
            "HetAESNReservoir_LeakingRate" : self.LeakingRate,
            "HetAESNReservoir_InputScale" : self.InputScale,
            "HetAESNReservoir_Rho" : self.Rho,
            "HetAESNReservoir_Density" : self.Density
            })
        self.HetAESNReservoir = Reservoir.Module_HetAESNReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "LTRead",
            "ModuleShowName" : "LT Readout",
            "LinearTransformer_D_z" : self.D_z,
            "LinearTransformer_D_y" : self.D_y
            })
        self.Readout_LinearTransformer = Readout.Module_Readout_LinearTransformer(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.HetAESNReservoir.forward(u)
        z = np.concatenate([s, u])
        self.HetAESNReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinearTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.HetAESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinearTransformer.forward(z)
        self.HetAESNReservoir.update()
        return y
    
    #順伝播（エラー出力付き）?
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.HetAESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinearTransformer.forward(z)
        e = self.Readout_LinearTransformer.RMSE(z, y_d)
        self.HetAESNReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_LinearTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.HetAESNReservoir.reset()
        self.Readout_LinearTransformer.reset()

    #ディープコピー
    def clone(self) -> any:
        new = Model_HetAESN(self.Param.copy(), self.Evaluation)
        new.HetAESNReservoir = self.HetAESNReservoir.clone()
        new.Readout_LinearTransformer = self.Readout_LinearTransformer.clone()
        return new

    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        return self.HetAESNReservoir.getSubReservoirs()
        
class Model_HetAESN_DNNReadout(Model):
    """
    多層リードアウトのリーク率の異なるマルチリザバーモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_HetAESN_D_u"]                   #入力信号次元
        self.D_x = self.Param["Model_HetAESN_D_x"]                   #サブリザバーニューロン数
        self.LeakingRate = self.Param["Model_HetAESN_LeakingRate"]   #リーク率配列->サブリザバーの数(必ずリスト型)
        self.N_SubReservoirs = len(self.LeakingRate)                        #サブリザバー数
        self.D_y = self.Param["Model_HetAESN_D_y"]                   #出力信号次元
        self.D_z = self.N_SubReservoirs * self.D_x + self.D_u               #特徴ベクトル次元
        self.InputScale = self.Param["Model_HetAESN_InputScale"]     #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["Model_HetAESN_Rho"]                   #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["Model_HetAESN_Density"]           #結合密度配列（リスト型可能，None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "DifferentLRRes",
            "ModuleShowName" : "DLR-ESN",
            "HetAESNReservoir_D_u" : self.D_u,
            "HetAESNReservoir_D_x" : self.D_x,
            "HetAESNReservoir_LeakingRate" : self.LeakingRate,
            "HetAESNReservoir_InputScale" : self.InputScale,
            "HetAESNReservoir_Rho" : self.Rho,
            "HetAESNReservoir_Density" : self.Density
            })
        self.HetAESNReservoir = Reservoir.Module_HetAESNReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "DNNRead",
            "ModuleShowName" : "DNNReadout",
            "DNN_D_z" : self.D_z,
            "DNN_D_y" : self.D_y
            })
        self.Readout_DNN = Readout.Module_Readout_DNN(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.HetAESNReservoir.forward(u)
        z = np.concatenate([s, u])
        self.HetAESNReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_DNN.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.HetAESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        self.HetAESNReservoir.update()
        return y
    
    #順伝播（エラー出力付き）?
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.HetAESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        e = self.Readout_DNN.RMSE(z, y_d)
        self.HetAESNReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_DNN.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.HetAESNReservoir.reset()
        self.Readout_DNN.reset()

    #ディープコピー
    def clone(self) -> any:
        new = Model_HetAESN_DNNReadout(self.Param.copy(), self.Evaluation)
        new.HetAESNReservoir = self.HetAESNReservoir.clone()
        new.Readout_DNN = self.Readout_DNN.clone()
        return new

    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        return self.HetAESNReservoir.getSubReservoirs()

#--------------------------------------------------------------------
#Hubを持つAESNモデル
class Model_AESNwithHub(Model):
    """
    Hubを持つAESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_AESNwithHub_D_u"]          #入力信号次元
        self.D_h = self.Param["Model_AESNwithHub_D_h"]          #サブリザバーニューロン数
        self.D_H = self.D_u * self.D_h                          #総ニューロン数
        self.D_x = self.Param["Model_AESNwithHub_D_x"]          #ハブリザバーニューロン数
        self.D_y = self.Param["Model_AESNwithHub_D_y"]          #出力信号次元
        self.F_Use_U2HubConnection = self.Param["Model_AESNwithHub_F_Use_U2HubConnection"]   #Hubへの入力信号使用の有無
        self.F_Use_x2zConnection = self.Param["Model_AESNwithHub_F_Use_x2zConnection"]       #出力へのHub状態使用の有無
        self.F_Use_x2AESNConnection = self.Param["Model_AESNwithHub_F_Use_x2AESNConnection"] #AESNへのHub信号使用の有無
        self.F_Use_AverageHInHub = self.Param["Model_AESNwithHub_F_Use_AverageHInHub"]       #HubでHの平均をするか
        if self.F_Use_x2zConnection:
            self.D_z = self.D_H + self.D_x + self.D_u           #特徴ベクトル次元
        else:
            self.D_z = self.D_H + self.D_u                      #特徴ベクトル次元
        self.Hub_LeakingRate = self.Param["Model_AESNwithHub_Hub_LeakingRate"]  #Hubリーク率配列（None可能）
        self.Hub_InputScale = self.Param["Model_AESNwithHub_Hub_InputScale"]    #Hub入力スケーリング配列（None可能）
        self.Hub_Rho = self.Param["Model_AESNwithHub_Hub_Rho"]                  #Hubスペクトル半径配列（None可能）
        self.Hub_Density = self.Param["Model_AESNwithHub_Hub_Density"]          #Hub結合密度配列（None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "AESNHRes",
            "ModuleShowName" : "AESNwithHub",
            "AESNwithHubReservoir_D_u" : self.D_u,
            "AESNwithHubReservoir_D_h" : self.D_h,
            "AESNwithHubReservoir_D_x" : self.D_x,
            "AESNwithHubReservoir_F_Use_U2HubConnection" : self.F_Use_U2HubConnection,
            "AESNwithHubReservoir_F_Use_x2zConnection" : self.F_Use_x2zConnection,
            "AESNwithHubReservoir_F_Use_x2AESNConnection" : self.F_Use_x2AESNConnection,
            "AESNwithHubReservoir_F_Use_AverageHInHub" : self.F_Use_AverageHInHub,
            "AESNwithHubReservoir_Hub_LeakingRate" : self.Hub_LeakingRate,
            "AESNwithHubReservoir_Hub_InputScale" : self.Hub_InputScale,
            "AESNwithHubReservoir_Hub_Rho" : self.Hub_Rho,
            "AESNwithHubReservoir_Hub_Density" : self.Hub_Density,
            })
        self.AESNwithHubReservoir = Reservoir.Module_AESNwithHubReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "LTRead",
            "ModuleShowName" : "LT Readout",
            "LinearTransformer_D_z" : self.D_z,
            "LinearTransformer_D_y" : self.D_y
            })
        self.Readout_LinearTransformer = Readout.Module_Readout_LinearTransformer(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.AESNwithHubReservoir.forward(u)
        z = np.concatenate([s, u])
        self.AESNwithHubReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinearTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.AESNwithHubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinearTransformer.forward(z)
        self.AESNwithHubReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.AESNwithHubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinearTransformer.forward(z)
        e = self.Readout_LinearTransformer.RMSE(z, y_d)
        self.AESNwithHubReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_LinearTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.AESNwithHubReservoir.reset()
        self.Readout_LinearTransformer.reset()

    #ディープコピー
    def clone(self) -> any:
        new = Model_AESNwithHub(self.Param.copy(), self.Evaluation)
        new.AESNwithHubReservoir = self.AESNwithHubReservoir.clone()
        new.Readout_LinearTransformer = self.Readout_LinearTransformer.clone()
        return new

    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        return self.AESNwithHubReservoir.getSubReservoirs()
        
class Model_AESNwithHub_DNNReadout(Model):
    """
    多層リードアウトHubを持つAESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)
        
        #パラメータ取得
        self.D_u = self.Param["Model_AESNwithHub_D_u"]          #入力信号次元
        self.D_h = self.Param["Model_AESNwithHub_D_h"]          #サブリザバーニューロン数
        self.D_H = self.D_u * self.D_h                          #総ニューロン数
        self.D_x = self.Param["Model_AESNwithHub_D_x"]          #ハブリザバーニューロン数
        self.D_y = self.Param["Model_AESNwithHub_D_y"]          #出力信号次元
        self.F_Use_U2HubConnection = self.Param["Model_AESNwithHub_F_Use_U2HubConnection"]   #Hubへの入力信号使用の有無
        self.F_Use_x2zConnection = self.Param["Model_AESNwithHub_F_Use_x2zConnection"]       #出力へのHub状態使用の有無
        self.F_Use_x2AESNConnection = self.Param["Model_AESNwithHub_F_Use_x2AESNConnection"] #AESNへのHub信号使用の有無
        self.F_Use_AverageHInHub = self.Param["Model_AESNwithHub_F_Use_AverageHInHub"]       #HubでHの平均をするか
        if self.F_Use_x2zConnection:
            self.D_z = self.D_H + self.D_x + self.D_u           #特徴ベクトル次元
        else:
            self.D_z = self.D_H + self.D_u                      #特徴ベクトル次元
        self.Hub_LeakingRate = self.Param["Model_AESNwithHub_Hub_LeakingRate"]  #Hubリーク率配列（None可能）
        self.Hub_InputScale = self.Param["Model_AESNwithHub_Hub_InputScale"]    #Hub入力スケーリング配列（None可能）
        self.Hub_Rho = self.Param["Model_AESNwithHub_Hub_Rho"]                  #Hubスペクトル半径配列（None可能）
        self.Hub_Density = self.Param["Model_AESNwithHub_Hub_Density"]          #Hub結合密度配列（None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "AESNHRes",
            "ModuleShowName" : "AESNwithHub",
            "AESNwithHubReservoir_D_u" : self.D_u,
            "AESNwithHubReservoir_D_h" : self.D_h,
            "AESNwithHubReservoir_D_x" : self.D_x,
            "AESNwithHubReservoir_F_Use_U2HubConnection" : self.F_Use_U2HubConnection,
            "AESNwithHubReservoir_F_Use_x2zConnection" : self.F_Use_x2zConnection,
            "AESNwithHubReservoir_F_Use_x2AESNConnection" : self.F_Use_x2AESNConnection,
            "AESNwithHubReservoir_F_Use_AverageHInHub" : self.F_Use_AverageHInHub,
            "AESNwithHubReservoir_Hub_LeakingRate" : self.Hub_LeakingRate,
            "AESNwithHubReservoir_Hub_InputScale" : self.Hub_InputScale,
            "AESNwithHubReservoir_Hub_Rho" : self.Hub_Rho,
            "AESNwithHubReservoir_Hub_Density" : self.Hub_Density,
            })
        self.AESNwithHubReservoir = Reservoir.Module_AESNwithHubReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "DNNRead",
            "ModuleShowName" : "DNNReadout",
            "DNN_D_z" : self.D_z,
            "DNN_D_y" : self.D_y
            })
        self.Readout_DNN = Readout.Module_Readout_DNN(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.AESNwithHubReservoir.forward(u)
        z = np.concatenate([s, u])
        self.AESNwithHubReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_DNN.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.AESNwithHubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        self.AESNwithHubReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.AESNwithHubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        e = self.Readout_DNN.RMSE(z, y_d)
        self.AESNwithHubReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_DNN.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.AESNwithHubReservoir.reset()
        self.Readout_DNN.reset()

    #ディープコピー
    def clone(self) -> any:
        new = Model_AESNwithHub_DNNReadout(self.Param.copy(), self.Evaluation)
        new.AESNwithHubReservoir = self.AESNwithHubReservoir.clone()
        new.Readout_DNN = self.Readout_DNN.clone()
        return new

    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        return self.AESNwithHubReservoir.getSubReservoirs()
        
#--------------------------------------------------------------------
#変形DeepESNモデル
class Model_ModifiedDeepESN(Model):
    """
    変形DeepESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_ModifiedDeepESN_D_u"]      #入力信号次元
        self.D_x = self.Param["Model_ModifiedDeepESN_D_x"]      #各層ニューロン数（リスト型）
        self.D_l = len(self.D_x)                                #層の数
        self.D_y = self.Param["Model_ModifiedDeepESN_D_y"]      #出力信号次元
        self.D_s = sum(self.D_x)                                #状態ベクトル次元
        self.D_z = self.D_s + self.D_u                          #特徴ベクトル次元
        self.LeakingRate = self.Param["Model_ModifiedDeepESN_LeakingRate"]  #リーク率配列（リスト型可能，None可能）
        self.InputScale = self.Param["Model_ModifiedDeepESN_InputScale"]    #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["Model_ModifiedDeepESN_Rho"]                  #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["Model_ModifiedDeepESN_Density"]          #結合密度配列（リスト型可能，None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "MDeepESNRes",
            "ModuleShowName" : "M-DeepESN",
            "ModifiedDeepReservoir_D_u" : self.D_u,
            "ModifiedDeepReservoir_D_x" : self.D_x,
            "ModifiedDeepReservoir_LeakingRate" : self.LeakingRate,
            "ModifiedDeepReservoir_InputScale" : self.InputScale,
            "ModifiedDeepReservoir_Rho" : self.Rho,
            "ModifiedDeepReservoir_Density" : self.Density
            })
        self.ModifiedDeepReservoir = Reservoir.Module_ModifiedDeepReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "LTRead",
            "ModuleShowName" : "LT Readout",
            "LinearTransformer_D_z" : self.D_z,
            "LinearTransformer_D_y" : self.D_y
            })
        self.Readout_LinearTransformer = Readout.Module_Readout_LinearTransformer(param, self)
        
    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.ModifiedDeepReservoir.forward(u)
        z = np.concatenate([s, u])
        self.ModifiedDeepReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinearTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.ModifiedDeepReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinearTransformer.forward(z)
        self.ModifiedDeepReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.ModifiedDeepReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinearTransformer.forward(z)
        e = self.Readout_LinearTransformer.RMSE(z, y_d)
        self.ModifiedDeepReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_LinearTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.ModifiedDeepReservoir.reset()
        self.Readout_LinearTransformer.reset()

    #ディープコピー
    def clone(self) -> any:
        new = Model_ModifiedDeepESN(self.Param.copy(), self.Evaluation)
        new.ModifiedDeepReservoir = self.ModifiedDeepReservoir.clone()
        new.Readout_LinearTransformer = self.Readout_LinearTransformer.clone()
        return new

    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        return self.ModifiedDeepReservoir.getSubReservoirs()

class Model_ModifiedDeepESN_DNNReadout(Model):
    """
    多層リードアウト変形DeepESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)
        
        #パラメータ取得
        self.D_u = self.Param["Model_ModifiedDeepESN_D_u"]      #入力信号次元
        self.D_x = self.Param["Model_ModifiedDeepESN_D_x"]      #各層ニューロン数（リスト型）
        self.D_l = len(self.D_x)                                #層の数
        self.D_y = self.Param["Model_ModifiedDeepESN_D_y"]      #出力信号次元
        self.D_s = sum(self.D_x)                                #状態ベクトル次元
        self.D_z = self.D_s + self.D_u                          #特徴ベクトル次元
        self.LeakingRate = self.Param["Model_ModifiedDeepESN_LeakingRate"]  #リーク率配列（リスト型可能，None可能）
        self.InputScale = self.Param["Model_ModifiedDeepESN_InputScale"]    #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["Model_ModifiedDeepESN_Rho"]                  #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["Model_ModifiedDeepESN_Density"]          #結合密度配列（リスト型可能，None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "MDeepESNRes",
            "ModuleShowName" : "M-DeepESN",
            "ModifiedDeepReservoir_D_u" : self.D_u,
            "ModifiedDeepReservoir_D_x" : self.D_x,
            "ModifiedDeepReservoir_LeakingRate" : self.LeakingRate,
            "ModifiedDeepReservoir_InputScale" : self.InputScale,
            "ModifiedDeepReservoir_Rho" : self.Rho,
            "ModifiedDeepReservoir_Density" : self.Density
            })
        self.ModifiedDeepReservoir = Reservoir.Module_ModifiedDeepReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "ModuleName" : "DNNRead",
            "ModuleShowName" : "DNNReadout",
            "DNN_D_z" : self.D_z,
            "DNN_D_y" : self.D_y
            })
        self.Readout_DNN = Readout.Module_Readout_DNN(param, self)
        
    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.ModifiedDeepReservoir.forward(u)
        z = np.concatenate([s, u])
        self.ModifiedDeepReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_DNN.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.ModifiedDeepReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        self.ModifiedDeepReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.ModifiedDeepReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        e = self.Readout_DNN.RMSE(z, y_d)
        self.ModifiedDeepReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_DNN.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.ModifiedDeepReservoir.reset()
        self.Readout_DNN.reset()

    #ディープコピー
    def clone(self) -> any:
        new = Model_ModifiedDeepESN_DNNReadout(self.Param.copy(), self.Evaluation)
        new.ModifiedDeepReservoir = self.ModifiedDeepReservoir.clone()
        new.Readout_DNN = self.Readout_DNN.clone()
        return new
        
    #サブリザバー取得
    def getSubReservoirs(self) -> list:
        return self.ModifiedDeepReservoir.getSubReservoirs()
